# Dynamic visual-KV routing with FlexAttention

## Conclusion

FlexAttention is a practical backend for **rapidly experimenting with** the
two requested algorithms:

1. select visual tokens at an arbitrary prefill layer; and
2. let every decode query independently select the visual KV it may read in
   all later layers.

The implementation does not patch vLLM source, rewrite its block table, or
delete entries from its KV cache.  The algorithm is ordinary PyTorch and the
vLLM-specific code is isolated in one plugin.  This is substantially easier to
modify and debug than physical FlashAttention compaction.  The limitation is
equally important: the current Flex mask is **logical sparsity**, not physical
KV slicing, so it is an algorithm-development implementation and does not yet
claim decode acceleration or KV-memory reduction.

The static prefill implementation and its Flex-versus-compact performance
results are documented in
[`flex-attention-layerwise-implementation.md`](flex-attention-layerwise-implementation.md).
This document covers the new query-dependent decode path.

## The two experiment modes

### Arbitrary-layer prefill selection

Set `prune_after_layer=N` and choose one of these selector inputs:

- `vision_embedding`: select once from vision-tower output;
- `decoder_key`: select once from the real decoder Q/K/V tensors at layer
  `N`.

Layers `0..N` see the complete prompt.  Layers after `N` use the selected
visual tokens.  The selection is transported to the actor and exactly replayed
during training.  Custom selector functions receive normal PyTorch tensors,
so an experiment can score by K norm, QK, or another learned/statistical rule
without importing vLLM cache types.

### Per-query decode selection

Set `selector_input=decode_query`.  At the anchor layer `N`, the plugin keeps a
sidecar copy of that layer's complete K states.  For the final prefill query and
then for every generated-token query, it performs a fresh selection:

1. compute QK logits against the complete visible context;
2. apply the model scale and temperature softmax;
3. sum the visual probability mass per head and take the head maximum;
4. either derive a dynamic budget from that mass, or use a fixed budget;
5. average visual probabilities over heads and select the top-K visual keys;
6. apply this query-specific visual mask in every layer after `N`;
7. keep the full paged KV cache and repeat the decision for the next query.

The temperature-softmax denominator covers **all visible context keys**:

\[
p_{h,j}=\operatorname{softmax}_{j\in\text{all visible KV}}
\left(\frac{q_h k_{h,j}^{T}}{\sqrt{d}\,\tau}\right).
\]

This detail is required.  Normalizing only over visual keys would make their
total probability mass exactly one and would collapse the dynamic budget to
all visual tokens.  The implemented dynamic budget is

\[
K_t=\left\lceil N_v\max_h\sum_{i\in visual}p_{t,h,i}\right\rceil,
\qquad
s_{t,i}=\frac{1}{H}\sum_h p_{t,h,i}.
\]

Grouped-query attention is supported by expanding KV heads to their query-head
groups before scoring.

## Configuration

Fixed 5% budget, with a different top-K set for every query:

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.05
  prune_after_layer: 15
  layerwise_backend: flex
  selector_input: decode_query
  selector: vision_pulse
  selector_kwargs:
    budget_mode: fixed
    temperature: 0.1
    capture_capacity: 16
```

VisionPulse-style dynamic budget:

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.10
  prune_after_layer: 15
  layerwise_backend: flex
  selector_input: decode_query
  selector: vision_pulse
  selector_kwargs:
    budget_mode: visual_mass
    temperature: 0.4
    min_keep_ratio: 0.0
    max_keep_ratio: 1.0
    capture_capacity: 64
```

`keep_ratio` is the exact budget only in `fixed` mode.  In `visual_mass` mode
it is nominal experiment metadata; `min_keep_ratio` and `max_keep_ratio`
control the actual budget range.  `capture_capacity` must be at least the
largest possible selected count because it is the fixed-width temporary
transport used to return routing decisions from vLLM.  Unknown or invalid
dynamic options fail during configuration rather than being silently ignored.

## Why this is easier to experiment with

| Concern | Flex research path | Physical compact-Flash path |
|---|---|---|
| vLLM source fork | none | version-sensitive low-level integration |
| Paged-cache block table | unchanged | compact lengths/layout must be maintained |
| Per-query selection | mask changes each forward | requires dynamic gather/compact kernels |
| Algorithm code | pure PyTorch QK/softmax/top-K | coupled to cache and kernel layout |
| Full KV retained | yes | depends on compaction design |
| Current speed claim | none | measured faster for static pruning |

Flex does not eliminate all vLLM knowledge: the adapter still depends on
vLLM's attention metadata, physical slot mapping, and plugin lifecycle, so a
vLLM upgrade needs the GPU lifecycle tests.  It does, however, move that work
out of the algorithm.  A researcher changing temperature, budget, scoring, or
anchor layer does not need to change the block table or write a cache kernel.

## Real H20 results

The end-to-end runs used Qwen2.5-VL-3B-Instruct, PyTorch 2.10.0+cu129,
vLLM 0.18.0, Transformers 4.57.6, BF16, TP=1, anchor layer 15, and a 224×224
image that becomes 64 visual tokens.

### Fixed 5%

Twelve decode queries each retained 3/64 visual tokens.  They produced 11
different selected sets, proving that the mask is query-dependent rather than
request-static:

```text
[0,56,57] [9,16,19] [5,21,29] [8,9,13]
[5,9,29] [10,25,30] [5,9,24] [13,20,28]
[5,21,29] [22,25,34] [5,21,23] [0,8,57]
```

The generated answer was `The shapes are a red square and a blue circle.`

### Dynamic visual-mass budget

With `temperature=0.4`, the per-query budgets were:

```text
[2, 1, 18, 1, 1, 1, 1, 1, 20, 7, 4, 1, 3, 2, 3, 1]
```

Both the budget and selected token identities changed with the query.  With
`temperature=0.1`, all observed steps selected one token because the sharper
softmax assigned most mass to non-visual self/text keys at this layer.  That is
a useful algorithmic observation, not an implementation failure: temperature
and anchor layer need calibration for this model.

### Training replay

The fixed-5% rollout was replayed by the Transformers actor for three real
forward/backward/optimizer steps:

```text
student_visual_tokens = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
loss_before           = 0.0324364081
loss_after            = 0.0316218324
final_grad_norm       = 0.0766602
parameter_delta_norm  = 0.0259703
E2E_STAGE_TRAIN_STEP  = PASS
```

The rollout protocol stores a variable-length visual-index row for every
causal query.  The actor reconstructs a `[batch, query, key]` mask; the
remove-padding path converts it to a packed block-diagonal mask.  The teacher
still receives the unpruned input.

### vLLM versus Transformers logits

The vLLM Flex routing history was replayed one query at a time in Transformers
over one prefill logit and seven decode logits:

- all 8 greedy token IDs matched exactly;
- seven steps had 20/20 top-20 overlap and one had 19/20;
- sampled-token log-probability differences ranged from `0.0001988` to
  `0.0678258`;
- the declared cross-kernel BF16 tolerance `0.08` passed.

This is numerical parity, not bitwise identity.  vLLM FlexAttention and
Transformers SDPA have different BF16 reduction orders.  The acceptance rule
therefore requires identical greedy tokens plus an explicit normalized-logit
tolerance.

### Test result

The complete GPU directory passed:

```text
81 passed, 1 skipped
```

It includes pure-PyTorch equations, full-context normalization, GQA, dynamic
budget clamps, per-query reselection, paged-cache lifecycle, protocol
round-trips, multi-sample replay, packed masks, and static regressions.

Raw JSON outputs are stored under
[`artifacts/dynamic-decode-h20`](../artifacts/dynamic-decode-h20/README.md).

## Performance interpretation

The dynamic runs prove correctness and end-to-end usability, not speed.  The
current score modifier logically hides unselected KV but native FlexAttention
still traverses the relevant physical cache blocks; an additional full-K
sidecar is also kept at the anchor layer.  Therefore 5% or 10% selection must
not be reported as 20× or 10× decode acceleration.

This is intentional for the first research implementation.  Once an algorithm
is stable, the same backend-neutral selection protocol can feed a second,
optimized adapter that gathers selected KV or routes sparse cache blocks.  The
actor replay, selection math, and experiment records do not need to change.

## Current scope and fail-fast limits

- Qwen2.5-VL rollout integration;
- tensor parallel size 1;
- one image and no video per request;
- eager vLLM execution;
- prefix caching and chunked prefill disabled;
- dynamic actor replay does not support Ulysses sequence parallelism;
- the temporary routed-expert capture channel cannot simultaneously carry MoE
  routing replay;
- `capture_capacity` bounds the largest recorded K and exceeding it fails;
- no physical KV-cache deletion, memory reduction, or hardware speedup claim.

Within that scope, both requested algorithm stages are implemented and can be
changed without modifying vLLM itself.
