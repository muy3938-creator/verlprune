# Four FlexAttention vision-token pruning methods

## Result

The experimental backend now supports four research methods without patching
the vLLM source tree:

| Selector | Decision time | State used | Result after the boundary |
|---|---|---|---|
| `dart` | once at prefill layer `N` | layer-`N` K/V plus text K/V | one static visual subset |
| `divprune` | once at prefill layer `N` | layer-`N` visual V | one static visual subset |
| `greedy_prune` | once at prefill layer `N` | layer-`N` visual/text V | one static visual subset |
| `vision_pulse` | every decode query at layer `N` | current Q and complete layer-`N` K | a different visual-KV subset per query |

All four use the same backend-neutral selection protocol. vLLM records the
actual indices chosen during rollout and the Transformers actor replays those
indices as an ordinary PyTorch attention mask. Algorithm changes remain in
PyTorch selector code; the only vLLM-facing component is the out-of-tree
FlexAttention plugin.

This is an algorithm-development implementation. The Flex score modifier
logically masks visual keys but does not physically compact the paged KV
cache, so 5% retention is not a claim of 20x speedup or 20x lower KV memory.

## Algorithms and controlled adaptations

The implementations preserve the central selection rule of each source while
adapting it to Qwen2.5-VL, arbitrary decoder layers, and an exact-K experiment
protocol.

### DART

Source: [DART](https://arxiv.org/abs/2502.11494), *Stop Looking for Important
Tokens in Multimodal Language Models: Duplication Matters More*, and the
provided `DART/` Qwen2.5-VL implementation.

At the selected layer, DART chooses image and text pivots by K-state L1 norm.
It then uses position-free normalized V states and retains image tokens that
are least similar to those pivots, removing duplication rather than estimating
attention importance.

The released default uses four image and four text pivots. That cannot fit a
3-token 5% budget. This implementation therefore scales the image pivot count
down, preserves at least one duplication-aware selection, cycles through the
available image/text pivots, and stops at exact K. This is a documented
extreme-budget adaptation, not a claim of bitwise equivalence to the released
LLaVA/Qwen script.

Options:

```yaml
selector: dart
selector_input: decoder_key
selector_kwargs:
  pivot_image_tokens: 4
  pivot_text_tokens: 4
```

### DivPrune

Source: [DivPrune](https://arxiv.org/abs/2503.02175), *Diversity-based Visual
Token Pruning for Large Multimodal Models*, and the provided `divprune/`
implementation.

For normalized visual features `x_i`, define cosine distance

\[
d(i,j)=1-\hat{x}_i^T\hat{x}_j.
\]

The first token maximizes its distance to the nearest other token. Every next
token maximizes its minimum distance to the selected set:

\[
i^*=\arg\max_i\min_{j\in S}d(i,j).
\]

The implementation is a clean-room equivalent of the provided max-min loop.
It applies the same rule to the chosen decoder layer's visual V states instead
of being fixed to layer-0 image embeddings.

Options:

```yaml
selector: divprune
selector_input: decoder_key
selector_kwargs: {}
```

### GreedyPrune

Source: [GreedyPrune](https://arxiv.org/abs/2506.13166), *Retenting Critical
Visual Token Set for Large Vision Language Models*, provided as
`2506.13166v1.pdf`.

Semantic saliency is cosine similarity between a visual token and the last
text token. Candidates are sorted by saliency. The highest-ranked active token
is retained, then candidates whose cosine similarity to it is above `tau` are
suppressed. The process repeats until K tokens are selected or candidates are
exhausted.

The current platform accepts any sorted valid subset and does not force the
final visual token. MRoPE coordinates are gathered for exactly the chosen
indices, which is also compatible with dense Qwen3-VL.
If threshold suppression empties the set before exact K, this implementation
deterministically fills from the original saliency order. That keeps retention
ratios comparable across methods.

Options:

```yaml
selector: greedy_prune
selector_input: decoder_key
selector_kwargs:
  similarity_threshold: 0.9
```

### VisionPulse

Source: [VisionPulse](https://arxiv.org/abs/2605.31457), *Dynamic Visual
Sparsity for Efficient Multimodal Reasoning*, provided as
`2605.31457v1.pdf`.

For every decode query, temperature-scaled attention is normalized over all
visible context keys, not only visual keys:

\[
p_{t,h,j}=\operatorname{softmax}_{j\in\text{visible KV}}
\left(\frac{q_{t,h}k_{h,j}^{T}}{\sqrt d\,\tau}\right).
\]

The visual-mass budget and mean-head importance are

\[
K_t=\left\lceil N_v\max_h\sum_{i\in visual}p_{t,h,i}\right\rceil,
\qquad s_{t,i}=\frac{1}{H}\sum_h p_{t,h,i}.
\]

The top-`K_t` visual keys are visible to this query in all layers after the
anchor. `budget_mode: fixed` is also available for controlled 5%/10%
experiments. Full details are in
[`dynamic-decode-visionpulse-flex.md`](dynamic-decode-visionpulse-flex.md).

## Common configuration

Static example, DART at layer 15 with 5% retention:

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.05
  prune_after_layer: 15
  layerwise_backend: flex
  selector_input: decoder_key
  selector: dart
  selector_kwargs:
    pivot_image_tokens: 4
    pivot_text_tokens: 4
```

Dynamic VisionPulse example:

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

`prune_after_layer=N` means layers `0..N` see the full prompt and layers
`N+1..end` apply the selected mask. Static selectors choose once during
prefill. VisionPulse recomputes its choice for the final prefill query and
every generated-token query. `decoder_key` and `decode_query` are intentionally
Flex-only; unsupported backend/selector combinations and misspelled built-in
options fail during configuration instead of failing deep inside a rollout.

## Numerical and end-to-end validation

The real runs used one NVIDIA H20, Qwen2.5-VL-3B-Instruct, PyTorch
2.10.0+cu129, vLLM 0.18.0, Transformers 4.57.6, BF16, TP=1, and a 224x224
image producing 64 merged visual tokens.

### Extreme 5%, layer 15

| Method | Actual K | vLLM selected indices | Greedy parity | Max sampled log-prob diff | Actor optimization |
|---|---:|---|---:|---:|---|
| DART | 3/64 | `[7,37,63]` | 8/8 | 0.0521724 | 3 steps, grad 0.06299, delta 0.02617 |
| DivPrune | 3/64 | `[1,59,63]` | 8/8 | 0.0291409 | 3 steps, grad 0.05371, delta 0.02617 |
| GreedyPrune | 3/64 | `[15,56,63]` | 8/8 | 0.0617700 | 3 steps, grad 0.04907, delta 0.02614 |
| VisionPulse fixed | 3/64 per query | 12 query-specific sets | 8/8 | 0.0678258 | 3 steps, grad 0.07666, delta 0.02597 |

Each parity run compares one prefill and seven decode logits. The acceptance
rule is identical greedy tokens and sampled-token log-probability difference
at most 0.08. The tolerance accounts for different BF16 reduction orders in
vLLM FlexAttention and Transformers FlashAttention/SDPA; it is numerical
parity, not bitwise identity.

Numerical parity means that vLLM and the PyTorch mask implement the same
pruned computation; it does not prove task accuracy. On this deliberately tiny
single-image smoke case, DivPrune, GreedyPrune, and VisionPulse preserved the
red-left/blue-right description at 5%, while DART's 3-token result swapped the
colors. DART's layer-7 10% run recovered the colors but called both shapes
circles. This is evidence that the exact-K extreme-budget DART adaptation needs
quality evaluation or a larger pivot-compatible budget before use, not an
implementation-parity failure.

The actor test loads the real 3B model, replays the exact rollout mask, computes
the student/teacher distillation loss, backpropagates through the selected
attention path, and performs three optimizer steps on 4,194,304 trainable
parameters. Every method produced finite non-zero gradients and a non-zero
parameter delta.

### Arbitrary layer, batch 4, 10%

| Method | Boundary | Actual K | Output tokens/s | Result |
|---|---:|---:|---:|---|
| DART | 7 | 6/64 | 8.58 | pass |
| DivPrune | 27 | 6/64 | 11.30 | pass |
| GreedyPrune | 34 | 6/64 | 14.79 | pass |
| VisionPulse fixed | 7 | 6/64 per query | 7.14 | pass |

These latency numbers are smoke-run observations from different layer
boundaries, not a selector speed ranking. They establish that each new method
works beyond layer 15, with concurrent requests and a second retention ratio.
The broader platform matrix already covers layers 0, 7, 15, 27, and 34 and
batches 1, 2, 4, 8, and 16.

Raw results are stored under
[`artifacts/four-method-h20`](../artifacts/four-method-h20/README.md) and
[`artifacts/dynamic-decode-h20`](../artifacts/dynamic-decode-h20/README.md).
The complete H20 pruning test directory reports `92 passed, 1 skipped`.

## Scope

- Qwen2.5-VL rollout integration, TP=1, eager vLLM;
- one image and no video per request;
- chunked prefill and prefix caching disabled;
- dynamic replay does not support Ulysses sequence parallelism;
- no physical KV deletion or sparse-block speedup claim;
- the validation is a real rollout-to-actor forward/backward/optimizer loop,
  not a long distributed benchmark-training job.

Within that scope, the four methods are implemented, independently
configurable, numerically checked against a PyTorch mask reference, and usable
for rapid algorithm experiments without maintaining a vLLM source fork.
