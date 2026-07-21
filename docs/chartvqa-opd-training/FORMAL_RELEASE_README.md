# Vision-OPD token-pruning formal release

This release is a research extension of the original Vision-OPD training
platform. The original Vision-OPD implementation remains available as the
`original/vision-opd` Git branch; this branch adds a ChartQA data path and a
FlexAttention-based student-side visual-token pruning path without changing
the OPD reward or distillation objective.

## What this release implements

The training flow has four explicit pieces:

1. `scripts/prepare_chartvqa_opd.py` reads ChartQA (`image`, `query`, and
   `label`) and writes local complete chart images plus train/validation/test
   parquet files.
2. `images` and `teacher_images` point to the same complete chart. `prompt` and
   `teacher_prompt` contain the same user instruction, so student and teacher
   receive identical input semantics.
3. The student uses the configured visual-token pruning policy. The default
   research baseline selects random visual tokens at decoder layer 0 and keeps
   10% of them (`selector=random`, `selector_input=vision_embedding`,
   `prune_after_layer=0`, `keep_ratio=0.10`).
4. The teacher is an independent fixed checkpoint. It receives the complete
   image, has no optimizer, is marked `requires_grad=False`, and is never
   updated by EMA or progressive synchronization.

The FlexAttention path is intentionally a logical-mask research backend. It
keeps the implementation easy to change and debug; it does not claim to
release KV-cache memory unless a separate physical-compaction backend is used.

## Two training configurations

The release provides two separate presets rather than overloading one file:

| File | Student | Learning rate | Intended use |
|---|---|---:|---|
| `verl/trainer/config/chartvqa_opd_lora.yaml` | `use_lora: true`, rank 8 | `1e-5` | Daily algorithm experiments and quick checks |
| `verl/trainer/config/chartvqa_opd_full_training.yaml` | `use_lora: false`, full parameters | `1e-6` | Formal training and final measurements |

`use_lora` is an explicit switch added in this release. When it is `null`, the
legacy framework behavior is retained; these two presets set it explicitly so
that `lora_rank` alone cannot be misread as the experiment policy.

## Reproduce the data and training flow

Prepare ChartQA:

```bash
python3 scripts/prepare_chartvqa_opd.py \
  --dataset /root/data/chartvqa \
  --output-dir /root/experiments/chartvqa-opd/data \
  --train-limit 256 \
  --val-limit 128 \
  --test-limit 128
```

Run the daily LoRA path:

```bash
MODEL_PATH=/root/models/Qwen2.5-VL-3B-Instruct \
DATA_DIR=/root/experiments/chartvqa-opd/data \
TRAINING_CONFIG=chartvqa_opd_lora \
OUTPUT_DIR=/root/experiments/chartvqa-opd/training-lora \
bash scripts/run_chartvqa_opd_training.sh
```

Run the formal full-parameter path:

```bash
MODEL_PATH=/root/models/Qwen2.5-VL-3B-Instruct \
DATA_DIR=/root/experiments/chartvqa-opd/data \
TRAINING_CONFIG=chartvqa_opd_full_training \
OUTPUT_DIR=/root/experiments/chartvqa-opd/training-full \
bash scripts/run_chartvqa_opd_training.sh
```

The launcher is training-only. It checks that `train.parquet` exists, selects
the requested config, sets the fixed-teacher image column, and passes prompt
and sequence limits consistently. Benchmarking remains a separate operation.
For an offline smoke without W&B credentials, add `USE_WANDB=false`; the
launcher then uses only the console logger.
The launcher defaults vLLM to `gpu_memory_utilization=0.10` because the fixed
teacher and full-parameter student coexist during training. Increase
`ROLLOUT_GPU_MEMORY_UTILIZATION` only when the available GPU memory permits it.
On exit, the launcher stops its Ray cluster and cleans its Ray/vLLM child
processes. The cleanup deliberately does not touch the separate `./gg`
keepalive process.

## Changing the pruning algorithm

For a new selector, keep the data and teacher contract unchanged and override
only the pruning settings, for example:

```bash
SELECTOR=vision_pulse \
SELECTOR_INPUT=decode_query \
KEEP_RATIO=0.10 \
PRUNE_AFTER_LAYER=15 \
TRAINING_CONFIG=chartvqa_opd_lora \
bash scripts/run_chartvqa_opd_training.sh
```

Random selection is compatible with `vision_embedding`; query-dependent
selectors such as VisionPulse use `decode_query`. The selector should not
modify the teacher input or the ChartQA parquet schema.

## Validation record

The contract suite checks:

- complete-image preservation and identical student/teacher parquet fields;
- validation split naming (`validation.parquet`);
- explicit fixed-teacher launcher settings;
- both LoRA and full-parameter configuration contracts;
- fixed-teacher parameter immutability;
- Python and shell syntax.

On the H20 environment, the broader dependency-backed regression previously
passed (`49 passed`), and the focused ChartQA release contract currently passes
`8 passed, 2 warnings`. The focused checks include the complete-image/same-
prompt contract, fixed-teacher immutability, both LoRA/full-parameter presets,
cleanup behavior, Transformers 5.x Qwen2.5 helpers, and the FLOPs-counter
compatibility path for configs whose language fields live under `text_config`.

The one-step full-parameter smoke reached model construction and actor setup.
Its first post-update failure was a Transformers 5.x Qwen2.5-VL FLOPs-counter
field mismatch; that compatibility fix is now included and covered by the
focused test. A rerun on the shared H20 was then blocked before model loading
because `nvtop` reported `94.712/95.577 GiB` in use by an invisible external
workload. The launcher exited through its cleanup trap: no Ray, vLLM, or
`main_ppo` processes remained afterward. This is an infrastructure/resource
interruption, not a correctness or implementation failure, and no throughput
claim is made from that run.
