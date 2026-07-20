# vLLM Hybrid versus Transformers all-Flash varlen pruning

## Scope and status

This report compares the current arbitrary-layer research implementations on
Qwen2.5-VL-3B-Instruct.  A clean H20 run completed 24 batch-1 matrix records
(12 at 1,024 visual tokens and 12 at 2,025 visual tokens), plus four batch-4
confirmation records.  Earlier SDPA pilot runs are retained only as historical
context and are excluded from the conclusions.

## What is actually being compared

The short backend labels need one important qualification:

| Label | Layers through boundary `L` | Layers after `L` | KV behavior |
|---|---|---|---|
| `vllm_flex` | vLLM FlashAttention | vLLM FlexAttention `score_mod` | full paged KV retained; logical mask |
| `transformers_flash` | Hugging Face FlashAttention 2 | packed varlen FlashAttention 2 | full DynamicCache retained; selected Q/K/V gathered per call |

After the boundary, the generation adapter gathers retained Q/K/V, builds
separate query/KV `cu_seqlens`, calls `flash_attn_varlen_func`, and scatters
the query output back.  The cache allocation remains full, but the attention
kernel receives only retained rows.  This is the same mathematical pattern as
the training-side compact FlashAttention path.

Current vLLM Flex `score_mod` still masks logical keys without physically
compacting paged KV or guaranteeing structured sparse block skipping.  Thus the
comparison is now Hybrid vLLM Flash→Flex versus Transformers Flash→packed
varlen Flash.

## Controlled protocol

- Hardware: one shared NVIDIA H20 96 GB; a case starts only after three
  consecutive samples at no more than 5% GPU utilization.
- Model/dtype: Qwen2.5-VL-3B-Instruct, BF16.
- Input: one synthetic 896x896 image and a fixed short instruction.
- Measured processor output: 1,059 prompt tokens for the 1,024-visual-token
  case; the 2,025-visual-token case used a 1,260px image and a 2,060-token
  prompt.
- Selection: deterministic uniform selection.  This removes selector compute
  and selection-quality variance from the backend comparison.
- Matrix keep ratios: 50%, 10%, and 5%.
- Matrix boundaries: after decoder layers 0 and 15.  Layers `0..L` use the
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

## Clean batch-1 matrix

Each cell is `vLLM / Transformers` median seconds, with the parenthesized
number being `Transformers ÷ vLLM`.

### 1,024 visual tokens

| Boundary | Keep | Prefill TTFT | Decode | End to end |
|---:|---:|---:|---:|---:|
| 0 | 50% | 0.1145 / 0.1538 (1.34×) | 0.7803 / 1.2341 (1.58×) | 0.8948 / 1.3877 (1.55×) |
| 0 | 10% | 0.1142 / 0.1546 (1.35×) | 0.7824 / 1.2344 (1.58×) | 0.8974 / 1.3897 (1.55×) |
| 0 | 5% | 0.1142 / 0.1541 (1.35×) | 0.7936 / 1.2249 (1.54×) | 0.9078 / 1.3790 (1.52×) |
| 15 | 50% | 0.0978 / 0.1488 (1.52×) | 0.6860 / 1.0712 (1.56×) | 0.7839 / 1.2200 (1.56×) |
| 15 | 10% | 0.0977 / 0.1490 (1.53×) | 0.6861 / 1.0603 (1.55×) | 0.7842 / 1.2089 (1.54×) |
| 15 | 5% | 0.0982 / 0.1491 (1.52×) | 0.6895 / 1.0654 (1.55×) | 0.7877 / 1.2153 (1.54×) |

### 2,025 visual tokens

| Boundary | Keep | Prefill TTFT | Decode | End to end |
|---:|---:|---:|---:|---:|
| 0 | 50% | 0.2196 / 0.2783 (1.27×) | 0.8223 / 1.2639 (1.54×) | 1.0419 / 1.5422 (1.48×) |
| 0 | 10% | 0.2179 / 0.2761 (1.27×) | 0.8144 / 1.2394 (1.52×) | 1.0324 / 1.5152 (1.47×) |
| 0 | 5% | 0.2193 / 0.2761 (1.26×) | 0.8504 / 1.2209 (1.44×) | 1.0693 / 1.4989 (1.40×) |
| 15 | 50% | 0.1767 / 0.2752 (1.56×) | 0.6983 / 1.0585 (1.52×) | 0.8789 / 1.3358 (1.52×) |
| 15 | 10% | 0.1764 / 0.2738 (1.55×) | 0.6857 / 1.0546 (1.54×) | 0.8629 / 1.3286 (1.54×) |
| 15 | 5% | 0.1773 / 0.2740 (1.55×) | 0.6910 / 1.0598 (1.53×) | 0.8695 / 1.3338 (1.53×) |

### Batch-4 confirmation at layer 15, 10%

| Visual tokens | vLLM TTFT / Decode / Total | Transformers TTFT / Decode / Total | Total ratio |
|---:|---:|---:|---:|
| 1,024 | 0.295 / 0.707 / 1.003s | 0.504 / 1.126 / 1.631s | 1.63× |
| 2,025 | 0.649 / 0.878 / 1.527s | 0.996 / 1.132 / 2.127s | 1.39× |

The matrix JSON records report zero host utilization before startup.  The
small nonzero utilization visible after warmup is the benchmark process itself,
not an external workload.

## Existing controlled evidence

### Numerical validation of all-Flash generation

- Real H20 varlen FlashAttention matched compacted SDPA for both multi-request
  prefill and one-query decode, including GQA heads: `2 passed`.
- The complete sampler tests passed locally and on CUDA.
- On real Qwen2.5-VL-3B with a shared eight-token forced history, varlen Flash
  versus SDPA sampled-log-probability absolute difference was `0.02269` mean
  and `0.07551` maximum; 7/8 raw argmax tokens matched.
- One free-running token diverged at the near-tied second step, as expected
  when BF16 reduction order crosses an argmax boundary.  The cache shapes,
  retained counts, and subsequent forced-history logits remained finite and
  aligned within the declared kernel tolerance.

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

## Current conclusion

1. The cache-aware Transformers path can use FlashAttention at every decoder
   layer.  It no longer needs SDPA after pruning.
2. Hybrid vLLM remains faster in every clean case: batch-1 end-to-end speedup
   is 1.40--1.56×, and the two batch-4 confirmation points are 1.39× and 1.63×.
3. Moving the boundary from layer 0 to 15 reduces total latency by roughly
   12--18%, because more early layers stay on their native dense Flash path and
   fewer layers pay packing/Flex mask overhead.
4. Doubling visual tokens mostly increases TTFT.  At layer 15/10%, vLLM total
   latency rose from 0.784 to 0.863s, while Transformers rose from 1.209 to
   1.329s; decode time barely moved.
5. 50%→10%→5% retention changes total latency by less than about 2% in this
   batch-1 workload.  Varlen Flash reduces attention rows, but QKV projections,
   MLPs, vision encoding, cache gathering, and launch overhead still operate.
   Extreme pruning is therefore more useful for algorithm/quality experiments
   than as a guaranteed wall-clock acceleration at this model size.

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
