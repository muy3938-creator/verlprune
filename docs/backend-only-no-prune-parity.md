# Backend-only numerical parity with no visual-token pruning

## Purpose

This experiment separates attention-backend numerical differences from
visual-token pruning differences. It compares three rollout paths against the
same verl actor replay:

1. native Qwen vLLM FlashAttention with no Vision-OPD architecture override;
2. the Vision-OPD layerwise plugin using FlexAttention in every decoder layer;
3. the plugin using FlashAttention through the configured boundary and
   FlexAttention afterwards.

For the two plugin paths, the selector protocol still executes, but it is
configured so the rounded keep count is exactly the complete 64-token visual
set. Every request asserts `len(kept_visual_indices) == 64` before it is
accepted. Therefore all three paths retain exactly 64/64 visual tokens.

The production configuration continues to reject `keep_ratio=1`; this
diagnostic uses `0.999999999`, whose protocol keep count is exactly 64, without
weakening that production guard.

## Matrix

Environment: Qwen2.5-VL-3B-Instruct, BF16, NVIDIA H20, PyTorch
2.10.0+cu129, vLLM 0.18.0, Transformers 4.57.6.

The matrix contains 23 cases and 2,528 sampled tokens:

- native unpruned: 5 cases, 672 tokens;
- all-Flex no-prune plugin path: 9 cases, 928 tokens;
- hybrid no-prune plugin path: 9 cases, 928 tokens;
- boundaries 7, 15, and 23;
- batch sizes 1, 4, and 8;
- response lengths 16, 32, and 64;
- sampling seeds 1234 and 2025.

As in the larger pruning study, vLLM samples with temperature 1.0 and no
top-p/top-k truncation. The actor loads a fresh checkpoint and calls the real
`DataParallelPPOActor._forward_micro_batch` remove-padding path.

## Rollout versus actor results

| Rollout path | Tokens | Mean abs log-prob diff | P95 | P99 | Max | K3 | Ratio outside `[0.8,1.2]` | Probability correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native unpruned | 672 | 0.03042 | 0.12270 | 0.17988 | 0.35276 | 0.001357 | 0.74% | 0.999650 |
| All Flex, 64/64 kept | 928 | 0.02966 | 0.12360 | 0.19439 | 0.30759 | 0.001403 | 0.75% | 0.999604 |
| Hybrid, 64/64 kept | 928 | 0.03081 | 0.12446 | 0.19340 | 0.37406 | 0.001567 | 0.97% | 0.999625 |

The differences are small:

- all Flex mean error is 2.49% below native;
- hybrid mean error is 1.31% above native;
- hybrid mean error is 3.90% above all Flex;
- all three probability correlations exceed 0.9996;
- fewer than 1% of tokens fall outside PPO's `[0.8,1.2]` ratio interval.

The small relative percentages should not be over-interpreted: their absolute
K3 difference is around `1e-4`, and all three P99 values sit in the same
`0.18-0.19` band. No backend introduces an order-of-magnitude change.

## Boundary controls

The following rows pool two seeds for batch 4 and response length 16. Full Flex
is identical across the nominal boundary, as expected, because the boundary has
no effect when all layers use Flex and all tokens are retained.

| Boundary | Prefix backend | Tokens | Mean abs diff | P99 | Max | K3 |
|---:|---|---:|---:|---:|---:|---:|
| 7 | Flex | 128 | 0.02985 | 0.21377 | 0.30759 | 0.001732 |
| 7 | Flash | 128 | 0.03694 | 0.24971 | 0.37406 | 0.002448 |
| 15 | Flex | 128 | 0.02985 | 0.21377 | 0.30759 | 0.001732 |
| 15 | Flash | 128 | 0.02993 | 0.24997 | 0.34808 | 0.001847 |
| 23 | Flex | 128 | 0.02985 | 0.21377 | 0.30759 | 0.001732 |
| 23 | Flash | 128 | 0.03131 | 0.24998 | 0.32866 | 0.001728 |

There is no monotonic increase with the number of Flash layers. Boundary 15 is
almost identical to all Flex in mean error; boundary 23 has nearly identical K3;
boundary 7 is the noisiest of the two sampled seeds but remains in the native
baseline range.

## Direct Flex-versus-hybrid comparison

Nine configurations have matched Flex and hybrid runs. Across 37 sampled
sequences:

- 24 sequences (64.86%) are token-for-token identical;
- 756/928 sampled token positions (81.47%) match;
- 747 tokens share an identical autoregressive prefix and sampled token.

On the 747 same-prefix tokens, the two vLLM rollout backends differ by:

- mean absolute log-probability: `0.02586`;
- P95: `0.11298`;
- P99: `0.18744`;
- maximum: `0.24995`.

On 528 tokens belonging to fully identical sequences, the direct rollout
difference is mean `0.02564`, P99 `0.18744`, maximum `0.24995`.

The corresponding actor outputs for those same 528 tokens are exactly equal:

```text
actor Flex-plan replay vs actor Hybrid-plan replay
mean = 0.0, P99 = 0.0, max = 0.0
```

This is the cleanest evidence that the observed difference is produced by the
vLLM attention backend/kernel execution. The selection and actor replay are not
changing the computation when all tokens are retained.

## Relation to real 5%/10% pruning

The earlier expanded hybrid pruning matrix had mean absolute difference
`0.02884` and P99 `0.19179`. The new 64/64 no-prune hybrid matrix has mean
`0.03081` and P99 `0.19340`.

Thus real 5%/10% pruning is 6.42% lower in mean error and only 0.83% lower at
P99 than the no-prune backend-only control. There is no evidence that aggressive
token pruning creates the main rollout/actor numerical gap.

## Conclusion

The rollout/actor mismatch is primarily an inherent backend-computation effect:
vLLM performs paged, incremental attention, whereas the actor replays the full
packed sequence through Transformers attention. Switching between all Flex and
hybrid Flash/Flex changes individual BF16 log-probabilities, but the aggregate
error remains essentially the same as native unpruned vLLM.

This supports retaining the hybrid implementation. Numerical monitoring should
continue to treat approximately `0.03` mean and `0.18-0.20` P99 sampled-token
log-probability differences as the current backend baseline, while investigating
rare outliers separately.

Raw data and the direct paired-backend summary are stored under
`artifacts/backend-only-no-prune-h20/`.
