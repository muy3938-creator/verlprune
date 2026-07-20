# Vision-OPD layerwise FlexAttention implementation and H20 results

This document describes the original request-static prefill selection.  The
new per-decode-query VisionPulse path, training replay, and H20 results are in
[`dynamic-decode-visionpulse-flex.md`](dynamic-decode-visionpulse-flex.md).
The DART, DivPrune, GreedyPrune, and VisionPulse selector comparison is in
[`four-method-flex-pruning.md`](four-method-flex-pruning.md).

## Decision

FlexAttention is viable as a **research/debug backend** for the requested
algorithm.  It makes arbitrary-layer selection and fixed prefill-to-decode
mask reuse substantially easier to implement because vLLM continues to own
the paged KV cache.  It is not currently a performance replacement for the
existing compact FlashAttention backend: native masked FlexAttention was
about 3x slower in the measured vLLM 0.18/H20 workload and does not physically
reduce KV memory or token work.

The branch therefore keeps three explicit modes:

- `physical`: the validated layer-0 production baseline;
- `layerwise_flex`: the default layerwise algorithm-development backend;
- `layerwise_compact_flash`: the existing compact implementation retained as
  the performance/reference backend.

## Static-mode implemented semantics

With `prune_after_layer=15`:

1. Decoder layers 0 through 15 use all prompt tokens.
2. Selection runs exactly once.  It may use vision embeddings, or the boundary
   layer's Q/K/V through the pure-PyTorch selector request.
3. Layers 16 onward mask dropped image keys.  Dropped image-query attention
   outputs are also zeroed during the initial prefill.
4. Each later decoder layer records the prompt keep mask in a boolean sidecar
   indexed by vLLM's existing physical `slot_mapping`.
5. Every later decode query retains all text/generated-token KV and sees only
   the selected image KV.  Selection is not rerun during decode.

The built-in `key_norm` strategy demonstrates decoder-state selection.  A QK
score selector can use `query_states` and `key_states` from the same request
API and recompute only its required text-query × image-key slice; vLLM/Flex do
not expose or materialize a complete attention matrix.

## Why the vLLM integration is simpler

The Flex adapter does not:

- rewrite `slot_mapping` or `block_table`;
- recalculate compact `seq_lens` or `query_start_loc`;
- gather/scatter compact Q/K/V batches;
- call a custom KV-cache write kernel;
- track compact lengths by request or first physical block.

Its cache-facing logic is only:

```python
physical_slot_keep[metadata.slot_mapping] = packed_keep_mask

def score_mod(score, batch, head, query, physical_kv):
    return torch.where(
        physical_slot_keep[physical_kv],
        score,
        -float("inf"),
    )
```

New decode slots are reset to `True`; old prompt slots keep the selection.
Block reuse is reset by the next scheduled write.  Selector implementations
receive ordinary tensors and no vLLM metadata or cache types.

The tradeoff is deliberate: this is logical masking, not physical compaction.
It is easier to inspect and change, but it retains the full KV allocation and
the Flex kernel still visits the relevant physical blocks.

## Real H20 validation

Environment:

- NVIDIA H20 96 GB;
- PyTorch 2.10.0+cu129;
- vLLM 0.18.0;
- Transformers 4.57.6 (the repository-pinned version);
- Qwen2.5-VL-3B-Instruct;
- TP=1, eager execution, prefix caching and chunked prefill disabled;
- one 224×224 image, 64 post-merge image tokens, batch size 2;
- layer 15 selection, layers 16+ masked;
- one warmup plus three measured `generate` calls in the same engine.

### Steady-state rollout performance

| Backend | Keep | Image tokens | Output tokens/request | Median latency | Output tok/s |
|---|---:|---:|---:|---:|---:|
| Flex masked | 50% | 32 | 18 | 2.662 s | 13.50 |
| Compact Flash | 50% | 32 | 18 | 0.893 s | 40.26 |
| Flex masked | 10% | 6 | 13 | 2.055 s | 12.65 |
| Compact Flash | 10% | 6 | 13 | 0.664 s | 39.04 |
| Flex masked | 5% | 3 | 14 | 2.176 s | 12.86 |
| Compact Flash | 5% | 3 | 14 | 0.716 s | 38.98 |

At equal keep ratios and identical uniform selections, generated text matched
between the two backends.  Flex was 2.98x slower at 50%, 3.10x slower at 10%,
and 3.04x slower at 5%.

There is no extreme-pruning speedup in this masked implementation.  Flex
throughput stayed around 12.7–13.5 output tok/s from 50% down to 5%.  Lowering
the keep ratio changes semantics but does not compact cache blocks or reduce
the native Flex kernel's physical traversal.  The compact Flash backend is the
right choice when speed is the experiment's bottleneck.

### Boundary-state algorithm run

The actual `decoder_key`/`key_norm` path was run at layer 15 with 10% keep:

- both requests selected 6 of 64 image tokens;
- selected indices were `[20, 28, 37, 52, 58, 63]`;
- layer 63 happened to be retained in that historical selector run; current
  selectors no longer require it as an anchor;
- median latency was 2.117 s and output throughput was 12.27 tok/s;
- both requests produced: `The left shape is red and the right shape is blue.`

This is not a precomputed vision-tower mask: the selector consumed the real
decoder K tensor at the configured boundary and the resulting mask was reused
during decode.

### Expanded batch-size validation

The `decoder_key`/`key_norm` path was then exercised with concurrent batch
sizes 1, 2, 4, 8, and 16.  Every decoded request returned the expected exact
selection count and completed generation; no request consumed another
request's physical-slot mask.

| Batch | Keep | Selected/request | Median measured latency | Output tok/s | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 10% | 6/64 | 0.391 s | 33.26 | pass |
| 2 | 10% | 6/64 | 2.117 s | 12.27 | pass |
| 4 | 10% | 6/64 | 0.427 s | 121.74 | pass |
| 8 | 10% | 6/64 | 3.429 s | 30.33 | pass |
| 8 | 5% | 3/64 | 2.444 s | 42.55 | pass |
| 16 | 10% | 6/64 | 2.903 s | 71.64 | pass |

These rows were collected across several CNB scheduling windows.  The H20 is
shared, and its unrelated usage varied from almost idle to more than 60 GB;
therefore the table is strong concurrency/correctness evidence but is not a
clean batch-scaling curve.  The paired Flex-versus-compact table above remains
the controlled backend comparison.

Two runtime observations matter for future experiments:

- batch 1 and 4 were stable after one warmup and showed normal batching gain;
- batch 8 produced measured latencies of 4.343 and 2.515 seconds at 10%,
  showing an additional first-shape compile/scheduling cost.  The 5% batch-8
  runs stabilized around 2.444 seconds, again with no proportional 5% speedup.

The batch-16 run started while the shared GPU was busy, so its throughput must
not be compared numerically with the idle batch-4 run.  Its useful result is
that 16 independent prefill selections survived paged-cache storage and later
decode without mask leakage or lifecycle failure.

### Expanded layer-boundary validation

With batch 4 and 10% keep, real decoder-key selection passed at layers 0, 7,
15, 27, and 34 of the 36-layer model:

| Select after layer | Selected visual indices | Median latency | Generated description |
|---:|---|---:|---|
| 0 | `[0, 1, 19, 56, 57, 63]` | 0.454 s | red / blue shapes |
| 7 | `[50, 51, 52, 59, 62, 63]` | 0.454 s | red / blue shapes |
| 15 | `[20, 28, 37, 52, 58, 63]` | 0.427 s | red / blue shapes |
| 27 | `[13, 20, 21, 40, 49, 63]` | 0.537 s | red square / blue circle |
| 34 | `[0, 4, 27, 52, 60, 63]` | 0.736 s | red square / blue circle |

The different selections prove that the plugin consumed each configured
layer's real K state rather than replaying a vision-tower mask.  Layer 34 also
tests the near-final boundary: selection happens there and layer 35 applies
the mask.  All four requests in every run completed prefill and decode.

The output difference is algorithmically plausible and useful for research:
later selection lets more full visual processing happen before the 10% mask,
and in this prompt it preserved the more specific shape names.  It is not a
quality benchmark, but it confirms that arbitrary layer choice has observable
model semantics.

### Training smoke run

The 10% Flex rollout selection was replayed by the Vision-OPD actor for three
real forward/backward/optimizer steps:

- finite losses: `0.0200607`, `0.0200767`, `0.0200708`;
- gradient norm each step: `0.0305176`;
- trained parameter delta norm: `0.0260809`;
- trainable parameters in the focused smoke step: 4,194,304;
- result: `E2E_STAGE_TRAIN_STEP=PASS`.

The actor uses the existing Transformers/FlashAttention replay path; the OPD
teacher, reward, protocol and trainer semantics are unchanged.  The remote
image initially provided Transformers 5.3.0, which is incompatible with this
repository's Qwen monkey patch.  The successful run used the repository pin
`transformers==4.57.6`.

## Transformers ↔ vLLM numerical parity

The numerical acceptance test uses the same Qwen weights, image, prompt,
uniform visual indices `[0, 13, 25, 38, 50, 63]`, 10% keep ratio, and layer-15
boundary in both implementations:

- the Transformers reference uses the existing layerwise FlashAttention actor
  path on the full sequence;
- vLLM uses the new masked FlexAttention path and its persistent physical-slot
  sidecar;
- both run BF16 greedy decoding on the same forced history;
- step 0 is the final **prefill** logit;
- steps 1–7 are **decode** logits and therefore validate reuse of the selected
  prompt KV mask.

Results:

- all 8 greedy token IDs matched exactly:
  `[785, 2115, 6083, 374, 2518, 323, 279, 1290]`;
- all 20 top-logit token IDs matched at every step (20/20 overlap);
- maximum sampled-token log-probability absolute difference: `0.0149001`;
- the strict sampled-log-probability tolerance `0.02` passed;
- prefill step-0 sampled-log-probability difference: `0.0008495`;
- decode sampled-log-probability differences ranged from `0.0000824` to
  `0.0149001`.

The top-20 values are not bit-identical because native vLLM Flex and
Transformers FlashAttention use different BF16 kernels and reduction orders;
the largest difference among shared top-20 entries was `0.24954`.  The greedy
path, top-20 ranking set, and sampled-token normalized logits nevertheless all
met the acceptance rule.  Requiring raw bitwise equality across these kernels
would be an invalid criterion; token equality plus a declared BF16 tolerance
is the reproducible numerical test.

A second run requested `max_logprobs=-1` and compared the complete 151,936
token vocabulary for the prefill step and three decode steps:

| Phase | Compared logits | Mean abs diff | P99 abs diff | Max abs diff |
|---|---:|---:|---:|---:|
| prefill step 0 | 151,936 | 0.04169 | 0.12415 | 0.24915 |
| decode step 1 | 151,936 | 0.03446 | 0.10722 | 0.19316 |
| decode step 2 | 151,936 | 0.03390 | 0.12127 | 0.24627 |
| decode step 3 | 151,936 | 0.04481 | 0.12508 | 0.20321 |

All four greedy tokens and all top-20 token sets still matched.  The maximum
sampled-token difference in this full-vocabulary run was `0.005661`.  These
statistics characterize the entire normalized-logit distribution instead of
only the chosen path.

The reusable three-stage test is `scripts/logit_parity_smoke.py`.  Its raw
vLLM, Transformers, and comparison outputs are stored with the other H20
artifacts.

## End-to-end training parity

Flex and compact-Flash rollouts were generated with the same 10% uniform
selection and identical generated token sequence.  Each rollout was then
replayed from a fresh copy of the same pretrained actor for three independent
forward/backward/AdamW steps.

| Metric | Flex rollout | Compact-Flash rollout |
|---|---:|---:|
| step losses | `[0.0200606603, 0.0200767387, 0.0200708378]` | identical |
| step grad norms | `[0.0305175781, 0.0305175781, 0.0305175781]` | identical |
| loss after step 3 | `0.0200722739` | identical |
| raw JSD after step 3 | `0.0200722739` | identical |
| parameter delta norm | `0.0260809157` | identical |
| status | pass | pass |

This exact equality is expected and desirable: rollout backend details stop at
the exact-selection protocol boundary, and the actor replays the same indices
through one backend-neutral Transformers training path.  It demonstrates that
switching the rollout implementation from compact Flash to Flex does not
change the OPD loss, gradients, or update when rollout tokens and selected
indices are held fixed.

## Test evidence

- CPU config/strategy/launch contracts: `28 passed`.
- Native Flex paged-KV numerical tests: `2 passed`.
- Boundary selection → later prefill → decode sidecar lifecycle: `1 passed`.
- Complete GPU `tests/vision_token_pruning` suite: `71 passed, 1 skipped`.
- Real batch-2 rollouts: 50%, 10%, and 5% on both Flex and compact Flash.
- Real Flex concurrency rollouts: batch 1, 2, 4, 8, and 16.
- Real Flex boundary rollouts: layers 0, 7, 15, 27, and 34.
- Real layer-15 decoder-key selection and three actor optimizer steps.
- Transformers/vLLM parity across one prefill logit and seven decode logits.
- Exact three-step training parity between Flex and compact-Flash rollouts.

Machine-readable results are stored in
`artifacts/flex-attention-h20/`.

## Recommended use

Use `layerwise_flex` while developing and debugging selectors that need a
chosen decoder layer's state.  It keeps algorithm code independent of vLLM's
cache internals and supports fast iteration across arbitrary layers and keep
ratios.  Before large training runs, replay the exact selected indices with
`layerwise_compact_flash` if rollout throughput matters.

Do not expect masked FlexAttention alone to deliver 5%/10% physical speedups.
Achieving that requires structured block masks that skip whole Flex blocks or
physical cache compaction; both reintroduce performance-specific engineering.
The branch intentionally separates that optimization problem from algorithm
experimentation.

Initial supported scope is Qwen2.5-VL, TP=1, one image/no video, no prefix
caching, and no chunked prefill.  Those constraints should remain explicit
until dedicated lifecycle tests are added for each feature.
