# ChartQA learned-budget H20 artifacts

This directory contains the retained small artifacts from the July 21, 2026
Qwen2.5-VL-3B ChartQA experiment. Large FSDP checkpoints remain outside the
repository; the exported 58 MiB LoRA adapter is stored in the dedicated local
research directory documented in `docs/chartvqa-learned-budget-h20.md`.

Files:

- `baseline16_512_stop.json`: dense 16-sample reference.
- `random10_16_512_stop.json`: random 10% visual-token reference.
- `top_p95_16_512_stop.json`: untrained dynamic top-p reference.
- `trained_top_p95_16_512_stop.json`: dynamic top-p after three LoRA steps.
- `baseline64_512_stop.json`: broader dense-only sanity check.
- `1.jsonl`, `2.jsonl`, `3.jsonl`: the three real training rollouts.
- `training-summary.json`: machine-readable training, adapter, and comparison
  metrics.

Every benchmark record includes the complete generated answer, latency,
generated-token count, maximum-length status, and pruning counts where
applicable.
