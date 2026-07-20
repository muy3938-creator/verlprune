# ChartQA learned-budget curriculum: H20 validation

## Scope and environment

This document records the complete rapid benchmark and training attempts for
the research-oriented dynamic visual-budget path on Qwen2.5-VL-3B-Instruct and
ChartQA. The validation split has 1,920 rows; the standard matrix below uses
the first 128 rows so every configuration can be compared under the same
inputs. It is a feasibility result, not a converged model claim.

Environment: NVIDIA H20, PyTorch 2.10.0+cu129, vLLM 0.18.0,
Transformers 4.57.6, BF16, TP=1, rank-8 LoRA.

## Response length and prompting

The ChartQA launcher now defaults to a 1024-token response budget instead of
the 512-token budget used for the completed runs below.
`MAX_PROMPT_LENGTH` and `RESPONSE_LENGTH` are independently overridable, and
the actor/vLLM total length is computed from their sum. Future runs request up
to five concise, evidence-grounded reasoning steps and an
`<answer>...</answer>` tag; unlike the completed runs, they no longer impose a
60-word reasoning limit. The benchmark stops at that tag when present and
records the configured token cap, generated-token counts, and maximum-length
hits.

Across every completed 128-row run, which used the earlier 512-token cap, the
hit rate was zero and the largest observed answer was 194 tokens. Raising the
default to 1024 gives future rollout sampling more room without changing these
historical results or forcing ordinary samples to become longer.

A fresh one-row H20 smoke with the new default and expanded prompt generated
181 tokens, answered correctly, stopped at `</answer>`, and still had a 0% cap
hit. A subsequent 16-row matrix reached 643 tokens, proving that the old 512
cap could truncate the expanded prompt even though the original prompt never
hit it. Raw smoke data is in
[`default1024-smoke.json`](../artifacts/chartvqa-learned-budget-h20/full128/default1024-smoke.json).

## 128-row benchmark

All rows use layer 0, dynamic top-p visual attention, a 5%-50% clamp, and the
same 512-token cap. `Mean retention` is relative to each sample's original
visual-token count.

| Mode | Accuracy | Mean retention | Mean response tokens | Cap hit | Mean latency |
|---|---:|---:|---:|---:|---:|
| Dense base | 57.03% | 100.00% | 39.98 | 0% | 1.134 s |
| Random 10% | 18.75% | 10.01% | 60.31 | 0% | 2.484 s |
| Top-p 0.80 | 52.34% | 37.77% | 40.85 | 0% | 1.727 s |
| Top-p 0.90 | 51.56% | 49.60% | 40.52 | 0% | 1.702 s |
| Top-p 0.95 | 53.13% | 49.96% | 40.69 | 0% | 1.719 s |

The dynamic selector is substantially better than random pruning. The quality
curve is not monotonic in retained tokens: 0.80 is the best untrained point
in this matrix, while 0.90 and 0.95 are effectively saturated by the 50%
clamp. This is evidence for learning or tuning a budget policy rather than
choosing a fixed percentage by hand.

## Training experiments

### 20-step two-stage diagnostic

The first 20-step run used 256 training rows, rank-8 LoRA, online W&B, and a
`0.95 -> 0.90 -> 0.85 -> 0.80` curriculum, but also physically prefilled with
random 10% visual tokens. Its post-training results were:

| Evaluation | Accuracy |
|---|---:|
| Dense with adapter | 54.69% |
| Top-p 0.80 with adapter | 52.34% |
| Top-p 0.95 with adapter | 50.00% |

This run is retained as a negative diagnostic. The Transformers benchmark
does not implement physical prefill compaction, so this configuration was not
semantically identical between rollout and benchmark.

### 10-step pure dynamic, low-learning-rate run

The corrected run explicitly set `PREFILL_KEEP_RATIO=none`, fixed top-p at 0.80,
and used `actor_rollout_ref.actor.optim.lr=1e-6` for 10 steps. This matches the Transformers
benchmark's pure dynamic semantics:

| Evaluation | Accuracy | Mean retention | Mean response tokens |
|---|---:|---:|---:|
| Dense base | 57.03% | 100.00% | 39.98 |
| Untrained top-p 0.80 | 52.34% | 37.77% | 40.85 |
| Pure dynamic LoRA, 10 steps | 53.13% | 37.71% | 40.95 |

The learned route improves the pruned score by 0.78 percentage points in this
single 128-row matrix, while remaining 3.91 points below dense. This is a
promising feasibility signal, not evidence of convergence. All ten rollouts,
backward passes, and the final checkpoint completed; no response hit 512.

The earlier pure-dynamic 20-step run at `1e-5` was stopped after its SSH
transport detached at step 8. Its rollout files are retained as a stability
diagnostic and should not be used as a trained result.

### 10-step expanded-CoT, 1024-token run

A final semantically aligned run regenerated all 256 training rows with the
five-step prompt, kept pure dynamic top-p 0.80 routing, and trained rank-8 LoRA
for 10 steps at `1e-6`. All ten checkpoints and rollouts completed. Gradient
norms ranged from 0.0251 to 0.9176, peak allocation was 36.75 GiB, and the
largest training rollout was 162 tokens with a zero clipping ratio.

The post-training diagnostic used the first 16 validation rows:

| Mode | Accuracy | Mean retention | Mean response | Max response | Cap hit |
|---|---:|---:|---:|---:|---:|
| Dense, expanded prompt | 50.00% | 100.00% | 274.31 | 588 | 0% |
| Top-p 0.80, untrained | 25.00% | 40.21% | 254.88 | 643 | 0% |
| Top-p 0.80, 10-step LoRA | 25.00% | 40.80% | 247.94 | 475 | 0% |

This is a negative quality result: the expanded CoT prompt makes the larger
token cap useful, but ten low-LR updates do not recover the pruning gap on this
small sample. It is not directly comparable with the 128-row table because the
prompt and sample count changed. Prompt verbosity should therefore remain an
explicit experimental variable rather than being treated as a guaranteed
quality improvement.

## W&B records

The online runs are in project `vision-opd-chartvqa`:

- dense 128: `qp0ctea6`
- random 10% 128: `37w5t86y`
- top-p 0.80/0.90/0.95: `5rfl4h4s`, `x8hqrhoc`, `nda66um5`
- two-stage 20-step: `gsywqzj0`
- two-stage adapter benchmarks: `7u32mpoc`, `f8w7hbll`, `ro42j30l`
- pure dynamic low-LR 10-step benchmark: `qaurj46g`
- expanded-CoT 1024-token 10-step training: `xxtxeu1i`

The training W&B service prints a closed-transport warning during Python
atexit cleanup after metrics have already been uploaded. The run pages and
summary metrics are present; this is a teardown issue, not a failed update.

## Single-GPU reproduction

With `WANDB_API_KEY` already exported, the complete dense/random/top-p
benchmark plus 10-step LoRA run is:

```bash
MODEL_PATH=/workspace/models/Qwen2.5-VL-3B-Instruct \
CHARTVQA_SOURCE=/workspace/data/chartvqa \
EXPERIMENT_DIR=/workspace/outputs/chartvqa-single-gpu \
bash scripts/run_chartvqa_single_gpu.sh
```

The defaults are 256 training rows, 128 validation/test rows, fixed top-p
0.80, 5%-50% retention clamps, no physical prefill pruning, learning rate
`1e-6`, response cap 1024, and total model length 3072. Set
`BUDGET_SCHEDULE='[0.95,0.90,0.85,0.80]'` only when explicitly testing the
request-order curriculum.

## Engineering conclusions

- Use the 1024-token default for future ChartQA training and evaluation. The
  `</answer>` stop criterion still ends normal answers early, while the larger
  ceiling avoids treating long but valid exploration trajectories as
  truncated samples; the expanded prompt produced a real 643-token response.
- Keep random 10% as a baseline only; its 18.75% score shows that token count
  alone is not a useful selection policy.
- Use top-p 0.80 as the current fast research setting. It retains about 38%
  of visual tokens and stays within 4.7 points of dense before training.
- Keep `PREFILL_KEEP_RATIO=none` for the Transformers/LoRA research loop unless
  the benchmark is switched to a vLLM-specific physical-prefill evaluator.
  The vLLM adapter still supports physical and delayed prefill for deployment
  experiments.
- The low-LR learned run is worth extending, but its current gain is small.
  A larger ChartQA training slice, reward tied to exact ChartQA correctness,
  or a dedicated trainable budget head is needed before claiming a learned
  router.
- Do not assume more verbose CoT improves ChartQA. The 16-row expanded-prompt
  diagnostic increased response length substantially without improving the
  trained top-p score.

## Retained artifacts

Small raw JSON and rollout files are in:

```text
/Users/test/Desktop/formalresearch/opd_mllm/Vision-OPD-flex-implementation/artifacts/chartvqa-learned-budget-h20
```

The latest pure-dynamic adapter is intentionally outside the Git worktree:

```text
/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-pure-lr1e6-10step
```

`adapter_model.safetensors` SHA-256:

```text
6d84f16cf3acefd36073eeb705db7e297b81e5c8c3d39902f6d4b1c47a695b84
```

The expanded-CoT adapter is also outside the worktree:

```text
/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-cot1024-top080-10step
```

Its `adapter_model.safetensors` SHA-256 is:

```text
af963f41220241e24f67eac42c5ccb39151bbf86a9aefbc06a2d4cfc653ecaa3
```
