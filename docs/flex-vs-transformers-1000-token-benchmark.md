# vLLM Flex versus Transformers Flash/SDPA at 1,024 visual tokens

## Scope and status

This report compares the current arbitrary-layer research implementations on
Qwen2.5-VL-3B-Instruct.  The planned controlled H20 matrix was stopped at the
user's request after repeated workspace allocations all landed on shared GPUs
held at 99--100% utilization.  Pilot runs validated the benchmark path and
exact token counts, but their latency is excluded from formal performance
claims.

## What is actually being compared

The short backend labels need one important qualification:

| Label | Layers through boundary `L` | Layers after `L` | KV behavior |
|---|---|---|---|
| `vllm_flex` | vLLM FlashAttention | vLLM FlexAttention `score_mod` | full paged KV retained; logical mask |
| `transformers_flash` | Hugging Face FlashAttention 2 | PyTorch SDPA + boolean mask | full DynamicCache retained; logical mask |

The Transformers reference is therefore not “all FlashAttention” after
pruning.  FlashAttention 2 cannot consume the arbitrary per-key mask used by
this experiment, so the existing cache-aware adapter changes to SDPA after the
boundary.  Likewise, current Flex `score_mod` masks logical keys but does not
physically compact vLLM's paged cache or guarantee structured sparse block
skipping.  A lower keep ratio can change outputs without reducing physical KV
traffic proportionally.

## Controlled protocol

- Hardware: one shared NVIDIA H20 96 GB; a case starts only after three
  consecutive samples at no more than 5% GPU utilization.
- Model/dtype: Qwen2.5-VL-3B-Instruct, BF16.
- Input: one synthetic 896x896 image and a fixed short instruction.
- Measured processor output: 1,059 prompt tokens per request, including
  exactly 1,024 merged visual tokens.
- Selection: deterministic uniform selection.  This removes selector compute
  and selection-quality variance from the backend comparison.
- Keep ratios: 100%, 50%, 25%, 10%, and 5%.
- Boundaries: after decoder layers 0, 7, 15, and 27.  Layers `0..L` use the
  early backend; masking starts at layer `L+1`.
- Decode: greedy, exactly 32 new tokens, KV cache enabled.
- Timing: one warmup and three measured runs per process-isolated case; report
  medians.
- Prefill metric: time to first token (TTFT).
- Decode metric: end-to-end latency minus TTFT, plus milliseconds per token
  over the remaining 31 decode transitions.

The 100% point uses `keep_ratio=1-1e-9`, which rounds to all 1,024 visual
tokens.  This deliberately keeps the same plugin, anchor selection, and
backend-switch path instead of giving the control an implementation shortcut.

## Batch-1 controlled matrix

Not completed.  No contaminated pilot timing is promoted into this table.

## Batch-4 confirmation matrix

Not completed.

## Existing controlled evidence

### Unpruned 1,024-visual-token backend baseline

An earlier uncontended H20 experiment used the same Qwen2.5-VL-3B model,
1,024 visual tokens, 32 generated tokens, and KV caching.  It compared native
unpruned vLLM generation with Transformers FlashAttention 2; it did **not**
exercise the post-boundary Flex mask.

| Batch | Transformers output tok/s | vLLM output tok/s | vLLM speedup | Transformers peak | vLLM peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.37 | 103.47 | 3.41x | 8,478 MiB | 9,290 MiB |
| 4 | 88.52 | 279.88 | 3.16x | 9,628 MiB | 9,290 MiB |

This is strong evidence for vLLM's scheduler and decode-engine advantage at
this context length, but it is not a direct Flex-versus-SDPA pruning result.

### Existing masked-Flex behavior at 64 visual tokens

Earlier controlled hybrid-vLLM measurements at boundary layer 15 showed that
reducing retention from 10% to 5% did not create a proportional speedup:

| Keep | Batch 1 | Batch 4 | Batch 8 |
|---:|---:|---:|---:|
| 10% | 0.3475 s | 0.3891 s | 0.4116 s |
| 5% | 0.3841 s | 0.4104 s | 0.4411 s |

The 5% case was slightly slower in those runs.  This supports the
implementation analysis: a `score_mod` mask changes which logits contribute
to softmax, but it does not compact paged KV or make physical work scale with
the number of retained visual tokens.

Replacing all-Flex early layers with native FlashAttention reduced latency by
2.50% at layer 7, 8.16% at layer 15, and 14.63% at layer 23 in the historical
batch-4 experiment.  The direction is consistent: a later boundary lets more
layers use Flash.  The rows did not all have the same natural output length,
so these percentages should not be used as a precise cross-layer scaling
curve.

At layer 15, a separate vLLM comparison between logical masked Flex and the
physically compacted Flash reference was stable at roughly 3x:

| Keep | Masked Flex | Compact Flash | Compact speedup |
|---:|---:|---:|---:|
| 50% | 2.662 s | 0.893 s | 2.98x |
| 10% | 2.055 s | 0.664 s | 3.10x |
| 5% | 2.176 s | 0.716 s | 3.04x |

This is not a Transformers comparison, but it isolates the most important
optimization fact: physical compaction matters much more than changing a
logical keep ratio from 10% to 5%.

## Pilot validation (excluded from performance conclusions)

The layer-0, 10% pilot proved that both adapters saw 1,024 visual tokens and
retained exactly 102.  Both generated 32 tokens with cache-aware decoding.
During these runs, however, an invisible sibling process used roughly
34--82 GiB and held the H20 at 99--100% utilization.  The observed latencies
varied too much to represent either backend, so no pilot timing is promoted to
the controlled result table.  For transparency only, the contaminated medians
were:

| Backend | TTFT | Decode | End to end | Output tok/s |
|---|---:|---:|---:|---:|
| vLLM Flash→Flex | 0.300 s | 5.029 s | 5.155 s | 6.21 |
| Transformers Flash→SDPA | 0.540 s | 6.309 s | 6.830 s | 4.69 |

These numbers suggest vLLM remained faster even under contention, but the two
processes did not receive identical competing load.  They must not be quoted
as a clean speedup measurement.

## Current best conclusion

1. At roughly 1,000 visual tokens, native unpruned vLLM was previously
   measured at 3.2--3.4x the end-to-end output throughput of Transformers.
2. The current arbitrary-layer vLLM Flex and Transformers SDPA adapters are
   both logical-mask implementations.  Neither should be expected to scale
   with a 10% or 5% keep ratio as if only that fraction of KV were read.
3. Moving the boundary later should help both implementations because more
   layers stay on FlashAttention.  Historical hybrid-vLLM data confirms the
   direction, but the requested clean 1,024-token layer curve was not obtained.
4. If speed at 5--10% retention becomes important, use physical token/KV
   compaction (the compact-Flash path) or a genuinely block-sparse Flex mask.
   The current Flex path remains valuable primarily because it is simple to
   modify and debug.

## Reproduction

One process-isolated case:

```bash
python scripts/benchmark_flex_vs_transformers_pruning.py vllm_flex \
  --model /path/to/Qwen2.5-VL-3B-Instruct \
  --output /tmp/vllm-r10-l15-b1.json \
  --keep-ratio 0.10 --prune-after-layer 15 \
  --batch-size 1 --resolution 896 --decode-tokens 32 \
  --warmup-runs 1 --measure-runs 3
```

Full matrix with an idle-GPU guard:

```bash
python scripts/run_flex_vs_transformers_matrix.py \
  --model /path/to/Qwen2.5-VL-3B-Instruct \
  --output-dir /tmp/flex-vs-transformers-b1 \
  --keep-ratios 1.0 0.5 0.25 0.1 0.05 \
  --layers 0 7 15 27 --batch-sizes 1 \
  --resolution 896 --decode-tokens 32
```
