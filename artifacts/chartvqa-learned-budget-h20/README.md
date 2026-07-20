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
- `full128/`: the complete 128-row W&B benchmark JSON files and the ten
  low-learning-rate pure-dynamic rollouts.

Every benchmark record includes the complete generated answer, latency,
generated-token count, maximum-length status, and pruning counts where
applicable.

`full128/default1024-smoke.json` is a one-row H20 verification of the new
1024-token default and the expanded five-step reasoning prompt. It generated
181 tokens, stopped at `</answer>`, and did not hit the cap. The 128-row matrix
and training summaries remain historical 512-token measurements.

`full128/*cot1024-16.json` and `full128/cot1024-training/` contain the final
expanded-prompt 16-row matrix, ten training rollouts, and raw logs. The
expanded adapter is retained outside Git at
`/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-cot1024-top080-10step`.

The latest pure-dynamic adapter is outside the repository at
`/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-pure-lr1e6-10step`.
