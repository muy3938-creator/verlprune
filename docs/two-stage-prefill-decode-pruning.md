# Two-stage prefill and decode visual-token pruning

## Semantics

This experiment combines two different reductions instead of reusing one
static subset for both stages:

```text
64 original visual tokens
  -> prefill embedding selector physically retains 32
  -> every decoder layer writes KV for those 32 only
  -> each decode query selects 16 of the cached 32 at layer 15
  -> layers 16+ read those 16 visual KV entries
```

With `prefill_keep_ratio=0.5` and `keep_ratio=0.5`, the second ratio is
relative to the first-stage subset. The nominal final visual fraction is
therefore `0.5 * 0.5 = 0.25`, not 0.5.

The first stage uses `embedding_norm` before decoder layer 0. It physically
shortens the prompt and paged KV cache. The second stage uses the existing
VisionPulse-style query/key score at the configured anchor layer and installs
a per-query FlexAttention mask for subsequent layers. Layers through the
anchor can use FlashAttention via `pre_pruning_backend=flash`.

## Configuration

The dedicated experiment preset now defaults to the requested 50%-then-50%
mode:

```yaml
vision_token_pruning:
  enabled: true

  # Stage 1: 64 -> 32 before decoder layer 0.
  prefill_keep_ratio: 0.5
  prefill_selector: embedding_norm
  prefill_selector_kwargs: {}

  # Stage 2: each query sees 16 of the cached 32 after layer 15.
  keep_ratio: 0.5
  selector: vision_pulse
  selector_input: decode_query
  selector_kwargs:
    budget_mode: fixed
    temperature: 0.1
    capture_capacity: 64

  prune_after_layer: 15
  layerwise_backend: flex
  pre_pruning_backend: flash
```

`VisionTokenPruningConfig` keeps `prefill_keep_ratio=null` as the compatibility
default, so existing physical-only and dynamic-only configurations remain
valid. The `vision_pruning_experiment.yaml` preset and launcher enable the
two-stage mode by default.

## Exact rollout-to-actor replay

The versioned `TwoStageVisionTokenSelection` contains two records:

- `prefill`: original token count and exact original indices physically kept;
- `decode`: one selection per query, indexed relative to the retained prefill
  subset.

On the actor, replay first removes the prefill-dropped positions and compacts
the visual embeddings. It then maps each decode-relative index onto the
remaining image positions and packs the resulting query-by-key mask with the
same compact attention mask. The unpruned OPD teacher receives neither stage.

This distinction is important. Treating second-stage indices as original
indices silently routes the wrong KV entries even when both stages retain the
same count.

## Real H20 validation

Environment: one NVIDIA H20, PyTorch 2.10.0+cu129, Transformers 4.57.6,
vLLM 0.18.0, BF16, eager FlexAttention after layer 15.

### Qwen2.5-VL-3B

- batch 1 rollout: `64 -> 32 -> 16` for all eight decode queries;
- batch 2 rollout: both requests independently produced `64 -> 32 -> 16`;
- three actor backward/update steps passed;
- peak actor allocation: 7471 MiB;
- parameter delta norm: 0.02594.

The real `DataParallelPPOActor._forward_micro_batch` parity run replayed the
same eight sampled tokens and both selection stages:

```text
sampled log-prob absolute difference mean = 0.01608
sampled log-prob absolute difference max  = 0.04906
probability Pearson correlation           = 0.999531
importance ratios outside [0.9, 1.1]      = 0 / 8
tokens with log-prob error > 0.05          = 0 / 8
```

This is within, and below many cases in, the existing Qwen2.5
vLLM-versus-actor backend-drift matrix (typically about 0.02-0.03 mean absolute
error). Those older cases use different batch/response shapes, so this is a
relative sanity check rather than a matched performance claim.

### Dense Qwen3-VL-4B

The same fused configuration also passed real rollout and three actor update
steps on the dense, non-MoE Qwen3 model:

```text
rollout visual tokens = 64 -> 32 -> 16 for 8/8 queries
training losses       = [2.8295e-4, 1.1341e-4, 2.7008e-5]
gradient norms        = [0.02222, 0.01257, 0.00531]
parameter delta norm  = 0.04307
peak actor allocation = 8813 MiB
```

A separate Qwen3 parity rerun was blocked by unrelated GPU occupancy after
the successful rollout/training run; Qwen2.5 provides the complete numerical
parity evidence for the combined protocol.

## Performance result

The combined method reduces the mathematical visual working set, but the
current eager Flex prototype is not a wall-clock acceleration. On the same
shared H20 workspace, batch 1, 32 fixed output tokens, one warmup and two
measured runs:

| Mode | Visual set | Median latency | Throughput |
|---|---:|---:|---:|
| physical prefill only | `64 -> 32` | 1.099 s | 29.13 tok/s |
| dynamic decode only | `64 -> 32/query` | 3.641 s | 8.79 tok/s |
| two-stage fused | `64 -> 32 -> 16/query` | 5.078 s | 6.30 tok/s |

The per-query selection, Flex score modifier, eager execution, and loss of
compiled/prefix-cached paths currently cost more than the smaller visual
attention matrix saves. This implementation is therefore useful as a simple,
correct algorithm experiment and training path. It should not yet be marketed
as an inference-speed optimization. Actual acceleration requires a compiled
sparse/paged kernel or physical gather path for the second stage.

The H20 was shared and external occupancy changed between runs, so the exact
latencies are indicative rather than publication-grade. The multi-fold gap to
the pure Flash path is large enough that this caveat does not change the
maintenance decision.

Raw compact summaries are in `artifacts/two-stage-prefill-decode-h20/`.
