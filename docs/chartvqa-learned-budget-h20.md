# ChartQA learned-budget curriculum: H20 validation

## Scope

This experiment validates the research-oriented dynamic-budget path on
Qwen2.5-VL-3B-Instruct and ChartQA. It is deliberately small: the goal is to
prove that a top-p visual-attention-mass budget can participate in the same
Vision-OPD rollout and LoRA update loop, not to claim a converged benchmark.

Environment: NVIDIA H20, PyTorch 2.10.0+cu129, vLLM 0.18.0,
Transformers 4.57.6, BF16, TP=1, rank-8 LoRA.

## Response-length change

The ChartQA launcher now defaults to a 512-token response budget instead of
256. `MAX_PROMPT_LENGTH` and `RESPONSE_LENGTH` are independently overridable,
and the actor/vLLM total length is computed from their sum. The benchmark uses
the same 512-token default and stops early after `</answer>` when the model
emits the requested answer tag.

The benchmark records generated-token counts and maximum-length hits. In every
completed run below the 512-token hit rate was zero. The longest observed
answer was 153 tokens, so the larger cap did not force longer responses; it
only removes truncation as a confound for harder future samples.

## Three-step training run

The successful run used physical random 10% prefill pruning plus a layer-0
VisionPulse selector with top-p 0.95 and curriculum
`[0.99, 0.97, 0.95, 0.90]`. The dynamic budget was clamped to 5%-50%.

| Step | Response tokens | Gradient norm | Max-length hit |
|---:|---:|---:|---:|
| 1 | 100 | 0.139531 | no |
| 2 | 58 | 0.158355 | no |
| 3 | 15 | 0.214126 | no |

All three real rollout, backward, optimizer, checkpoint, and exact-selection
replay steps completed. Peak allocated GPU memory was about 36.73 GiB. The
offline W&B teardown printed a closed-transport warning after completion; all
training outputs and checkpoints had already been written.

## Same-sample comparison

All rows below use the first 16 ChartQA validation samples and a 512-token cap.
The sample is too small for a quality claim, but it is sufficient for an
end-to-end feasibility check.

| Mode | Accuracy | Mean visual tokens | Mean response tokens | Cap hit | Mean latency |
|---|---:|---:|---:|---:|---:|
| Dense | 50.00% | all | 34.75 | 0% | 1.030 s |
| Random 10% | 18.75% | 20.75 | 77.06 | 0% | 3.202 s |
| Top-p 0.95 | 43.75% | 160.61 | 35.25 | 0% | 1.533 s |
| Top-p 0.95, 3-step LoRA | 50.00% | 162.24 | 31.81 | 0% | 2.087 s |

A separate dense-only 64-sample run produced 45.31% accuracy, 45.83 mean
response tokens, a maximum of 136 tokens, zero cap hits, and 1.720 s mean
latency.

## Interpretation

- Raising the response cap to 512 is safe for this training objective. Normal
  samples still stop naturally, and no completed sample approached the cap.
- Random 10% pruning damages both answer quality and answer concision.
- Dynamic top-p preserves much more visual context than a fixed 10% budget and
  stays close to dense quality in this smoke set.
- Three LoRA steps are enough to verify gradient flow and slightly change the
  selected budget/output behavior, but not enough to establish learned-budget
  convergence. The apparent 43.75%-to-50% improvement is one sample and must
  not be treated as a final result.
- FlexAttention remains slower than dense Transformers here. That is acceptable
  for the research path; a later physical compaction backend should provide the
  actual speedup once the learned policy is stable.

## Retained adapter

The exported adapter is intentionally outside the Git worktree:

```text
/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-512
```

`adapter_model.safetensors` SHA-256:

```text
095495b399cf3c69ef26f08c79f4733f82a6813751d961062ca428aea262a1d6
```

Raw small results are in `artifacts/chartvqa-learned-budget-h20/`.
