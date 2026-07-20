# Hybrid FlashAttention/FlexAttention layerwise pruning

## Conclusion

The experiment is feasible without patching vLLM. The global vLLM backend remains
FlexAttention so the existing arbitrary-layer and dynamic visual-KV masks keep
working. Layers from zero through the configured pruning boundary can optionally
delegate their attention computation to vLLM's native FlashAttention implementation;
layers after the boundary continue on FlexAttention.

Both implementations use the same paged KV-cache layout and cache-write operation in
vLLM 0.18.0, so the switch does not copy KV tensors or rewrite block tables.

## Configuration

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.10
  prune_after_layer: 15
  layerwise_backend: flex
  pre_pruning_backend: flash
  selector: uniform
  selector_input: vision_embedding
```

`pre_pruning_backend` accepts:

- `flex` (default): preserve the original all-Flex behavior.
- `flash`: use FlashAttention for layers `0..prune_after_layer`, including the
  boundary layer after its selector has observed Q/K/V. Use FlexAttention for all
  later layers.

The Flash option requires `prune_after_layer >= 0` and
`layerwise_backend: flex`. It works with embedding selectors, boundary Q/K/V
selectors such as DART, and per-decode-query VisionPulse selection.

## Compatibility guard

The adapter converts Flex metadata to Flash metadata only for semantics that the
paged causal FlashAttention kernel can reproduce. It falls back to FlexAttention
for a forward call when any of these are present:

- non-causal attention;
- cascade attention;
- a multimodal prefix range;
- an additional Flex score modifier.

This is intentionally fail-safe: enabling the experiment cannot silently discard an
arbitrary Flex mask. Unsupported configuration values fail during config validation.

## H20 validation

Environment: NVIDIA H20 96 GB, PyTorch 2.10.0+cu129, vLLM 0.18.0,
Transformers 4.57.6, Qwen2.5-VL-3B-Instruct, BF16, eager execution. The synthetic
image contains 64 visual tokens. Performance rows use two warmups and five measured
runs; model loading is excluded.

### Numerical correctness

At layer 15 with 10% visual-token retention:

- Hybrid and all-Flex vLLM generated the same four greedy tokens.
- Hybrid and the Transformers PyTorch-mask reference matched every greedy token over
  one prefill and three decode steps.
- Maximum sampled-token log-probability difference against the reference was
  `0.03684196`, within the established BF16 tolerance of `0.08`.
- A second layer-23 check also matched all four reference greedy tokens; its maximum
  sampled-token log-probability difference was `0.01568234`.
- A real DART rollout retained 6/64 visual tokens and completed generation.
- A real VisionPulse rollout selected 6/64 visual KV entries independently for each
  of 12 decode queries and completed generation.
- The DART result fed a three-step training smoke test: all losses and gradients were
  finite, loss moved from `0.02892619` to `0.02870888`, and the trainable parameter
  delta norm was `0.02607900`.

### Performance

Median end-to-end rollout latency in seconds:

| Boundary | Keep | Batch | All Flex | Hybrid | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 15 | 5% | 1 | 0.4248 | 0.3841 | 9.59% |
| 15 | 5% | 4 | 0.4626 | 0.4104 | 11.28% |
| 15 | 5% | 8 | 0.4728 | 0.4411 | 6.70% |
| 15 | 10% | 1 | 0.3822 | 0.3475 | 9.07% |
| 15 | 10% | 4 | 0.4237 | 0.3891 | 8.16% |
| 15 | 10% | 8 | 0.4519 | 0.4116 | 8.92% |
| 7 | 10% | 4 | 0.4391 | 0.4281 | 2.50% |
| 23 | 10% | 4 | 0.5647 | 0.4821 | 14.63% |

The trend is consistent with the design: an early boundary exposes fewer layers to
FlashAttention and yields a small gain; a later boundary exposes more layers and
yields a larger gain. The 5% and 10% results are similar because this option speeds
up the layers before pruning. Post-boundary Flex masking remains a logical mask and
does not claim physical sparse-KV bandwidth savings.

The layer-23 performance row forces exactly 20 output tokens for both backends;
other rows naturally terminated but produced identical token sequences within each
pair. Flash and Flex are not bitwise identical in BF16. One separate free-running
layer-23 trial crossed an argmax boundary after the first token even though the
isolated four-step PyTorch-reference check passed. Consequently, acceptance should
continue to use teacher-forced/log-probability tolerances rather than require
long-horizon free-running text to be identical in every run.

## Maintenance impact

The implementation is an out-of-tree adapter in the existing Vision-OPD vLLM plugin.
It adds one config field, one metadata converter, and a lazy native FlashAttention
delegate. There is no vLLM source fork, custom CUDA kernel, KV-cache conversion, or
per-model attention rewrite. This makes it appropriate as an experimental path for
quick algorithm iteration; `pre_pruning_backend: flex` remains the stable default.

Raw result JSON is stored under `artifacts/hybrid-flash-flex-h20/`.
