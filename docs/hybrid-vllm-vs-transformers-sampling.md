# Hybrid vLLM versus Transformers sampling

Validated branch: `experiment/hybrid-flash-flex-pruning`.

## Decision

Keep Hybrid vLLM as the default training rollout backend.  Use the new
cache-aware Transformers adapter for algorithm debugging and numerical
reference runs, not as the throughput default.

The reason is measured rather than architectural: on Qwen2.5-VL-3B BF16 and
one H20, controlled unpruned generation is 2.7--4.9x faster end to end in vLLM.
Decode itself is 5.7--8.4x faster.  Peak memory differs by less than about
0.9 GiB in the tested configurations, so switching to Transformers does not
buy enough memory to offset that rollout loss.

## Controlled backend baseline

Each request had about 1,900 input tokens and generated exactly 32 tokens.
Both implementations used KV caching.  Model loading and one-time vLLM
compilation were excluded.

| Visual tokens | Batch | Transformers tok/s | vLLM tok/s | vLLM speedup | Transformers peak | vLLM peak |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 31.85 | 157.18 | 4.94x | 8,244 MiB | 8,928 MiB |
| 64 | 4 | 101.84 | 276.71 | 2.72x | 8,920 MiB | 9,168 MiB |
| 256 | 1 | 30.89 | 130.38 | 4.22x | 8,266 MiB | 9,170 MiB |
| 256 | 4 | 102.47 | 278.80 | 2.72x | 8,942 MiB | 9,290 MiB |
| 1,024 | 1 | 30.37 | 103.47 | 3.41x | 8,478 MiB | 9,290 MiB |
| 1,024 | 4 | 88.52 | 279.88 | 3.16x | 9,628 MiB | 9,290 MiB |

A repeat during this work produced the same conclusion: for 1,024 visual
tokens, batch 1 was 28.52 versus 104.08 tok/s and batch 4 was 82.85 versus
277.39 tok/s.  The shared H20 had 54 GiB of unrelated resident memory but 0%
utilization; per-process deltas and latency remained stable.

## Hybrid pruning data already validated

At layer 15 with 64 visual tokens, Hybrid uses FlashAttention through the
anchor and FlexAttention masking afterwards.

| Keep | Batch | Hybrid median rollout latency |
|---:|---:|---:|
| 5% | 1 | 0.3841 s |
| 5% | 4 | 0.4104 s |
| 5% | 8 | 0.4411 s |
| 10% | 1 | 0.3475 s |
| 10% | 4 | 0.3891 s |
| 10% | 8 | 0.4116 s |

The similar 5% and 10% values are expected: the current Flex path logically
masks KV entries but does not physically compact the paged KV cache.

## Direct Transformers pruning adapter

`transformers_sampler.py` adds an instance-local, cache-aware implementation:

- ordinary Hugging Face Qwen2.5-VL and its native `DynamicCache`;
- FlashAttention through the configured anchor layer;
- PyTorch SDPA with a visual-key mask after the anchor;
- static DART, DivPrune, and GreedyPrune selection at the anchor;
- query-dependent VisionPulse selection during decode;
- no vLLM import, source patch, custom CUDA kernel, or global Transformers
  monkey patch.

Real H20 smoke runs completed for uniform, DART, and VisionPulse.  At 10%
retention they selected 26/256, 6/64, and 6/64 visual tokens respectively and
generated with the native HF cache.  VisionPulse changed the generated second
token relative to static DART, confirming that per-query routing was active.

These smoke latencies are deliberately not reported as performance results:
the assigned shared H20 moved between 66--97 GiB resident memory and sustained
100% utilization from an invisible sibling container.  A later nominally idle
window disappeared while the 896-pixel matrix was starting.  Treating those
numbers as a backend comparison would be misleading.

## Training-stage interpretation

Once a rollout selection is fixed, actor replay is backend-neutral.  The
training forward, teacher forward, loss, backward, and optimizer step are the
same whether the selection record came from vLLM or Transformers.  Therefore:

- fixed-selection actor step time should be reported once, not attributed to
  either sampler;
- different methods may change training cost only through their mask shape or
  retained count;
- end-to-end time is `rollout time + common actor/ref/teacher update time`.

Existing real H20 training smokes already prove finite loss, gradients, and
parameter updates for DART, DivPrune, GreedyPrune, and VisionPulse.  The GPU
smoke tool now additionally records per-step latency, total teacher-plus-train
time, and CUDA peak allocated/reserved memory.  Those timing fields must be
filled on an uncontended H20 before making a numerical end-to-end percentage
claim.

## Reproduction

Transformers cache-aware pruning:

```bash
python scripts/benchmark_transformers_pruned_sampler.py \
  --model /path/to/Qwen2.5-VL-3B-Instruct \
  --output /tmp/hf-dart-r05-b4.json \
  --selector dart --keep-ratio 0.05 \
  --prune-after-layer 15 --batch-size 4 \
  --resolution 896 --decode-tokens 32 \
  --warmup-runs 1 --measure-runs 3
```

Hybrid rollout and the common actor step:

```bash
python scripts/gpu_e2e_smoke.py rollout \
  --model /path/to/Qwen2.5-VL-3B-Instruct \
  --output-dir /tmp/hybrid-dart-r05-b4 \
  --selector dart --keep-ratio 0.05 \
  --selector-input decoder_key --prune-after-layer 15 \
  --pre-pruning-backend flash --layerwise-backend flex \
  --batch-size 4 --resolution 896 --decode-tokens 32 \
  --warmup-runs 1 --measure-runs 3 --fixed-output-tokens

python scripts/gpu_e2e_smoke.py train \
  --output-dir /tmp/hybrid-dart-r05-b4 --steps 3
```

Run each backend in a separate process and reject a run if the pre-run GPU
utilization is nonzero or unrelated resident memory changes during the matrix.

