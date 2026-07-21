# ChartQA OPD training

For the formal release description and original Vision-OPD branch semantics,
see [`FORMAL_RELEASE_README.md`](FORMAL_RELEASE_README.md).

This directory documents the reproducible ChartQA training path for the
Vision-OPD FlexAttention research branch.

## Data processing

The converter reads the original ChartQA rows (`image`, `query`, and `label`)
from a local datasets directory or a Hugging Face dataset name. It writes the
complete chart image to a local cache and creates one parquet file per split:

```bash
python3 scripts/prepare_chartvqa_opd.py \
  --dataset /root/data/chartvqa \
  --output-dir /root/experiments/chartvqa-opd/data \
  --train-limit 256 \
  --val-limit 128 \
  --test-limit 128
```

The output files are `train.parquet`, `validation.parquet`, and
`test.parquet`. Each row contains:

| Field | Meaning |
|---|---|
| `prompt` | Student chat instruction containing one `<image>` placeholder. |
| `teacher_prompt` | A byte-for-byte equivalent copy of the same instruction. |
| `images` | The complete chart image used by the student. |
| `teacher_images` | The same complete chart image path used by the fixed teacher. |
| `reward_model.ground_truth` | The first ChartQA label, used for evaluation. |
| `extra_info.labels` | All labels supplied by ChartQA. |

`teacher_images` is intentionally not a bounding-box or cropped image. The
converter writes the same path into `images` and `teacher_images`, so teacher
and student see the same pixels. The student-side pruning happens inside the
model after the configured layer; it is not performed by the data converter.

## Teacher/student semantics

The training launcher uses `teacher_always_on=true` and an independent
`teacher_model_source=fixed` module. By default the fixed teacher is loaded
from the same initial checkpoint as the student; set `TEACHER_MODEL_PATH` to a
different frozen checkpoint when desired.

For every sample:

1. Student and teacher receive the same complete image and the same instruction.
2. The student rollout may apply the configured visual-token pruning policy.
3. The teacher is run on the full image/input and is never updated.
4. The teacher has no optimizer, its parameters have `requires_grad=False`, and
   the actor update path does not perform EMA or progressive synchronization.

The launcher also sets `max_reprompt_len` to the student prompt limit so the
teacher does not silently receive a shorter prompt.

## Training-only launcher

After conversion, start the daily LoRA research run with:

```bash
MODEL_PATH=/root/models/Qwen2.5-VL-3B-Instruct \
DATA_DIR=/root/experiments/chartvqa-opd/data \
OUTPUT_DIR=/root/experiments/chartvqa-opd/training-lora \
TRAINING_CONFIG=chartvqa_opd_lora \
TOTAL_TRAINING_STEPS=10 \
bash scripts/run_chartvqa_opd_training.sh
```

For the formal full-parameter run, use the second configuration:

```bash
TRAINING_CONFIG=chartvqa_opd_full_training \
OUTPUT_DIR=/root/experiments/chartvqa-opd/training-full \
bash scripts/run_chartvqa_opd_training.sh
```

The two configuration files are deliberately separate:

- `chartvqa_opd_lora.yaml`: `use_lora: true`, rank 8, learning rate `1e-5`.
- `chartvqa_opd_full_training.yaml`: `use_lora: false`, `lora_rank: 0`,
  `lora_adapter_path: null`, learning rate `1e-6`.

`use_lora` is an explicit switch added to this branch. It avoids relying on
the meaning of a rank value alone; when it is `null`, the framework retains
its legacy inference from rank/adapter path.

Useful overrides include:

```bash
KEEP_RATIO=0.10                 # retain 10% (random is the default)
SELECTOR=random                 # default layer-0 random baseline
PREFILL_KEEP_RATIO=none         # keep prefill full for pure dynamic research
PRUNE_AFTER_LAYER=0             # layer at which decode-side mask is installed
MAX_PROMPT_LENGTH=2048
RESPONSE_LENGTH=1024
ACTOR_LR=1e-5                  # optional command-line override
```

The default ChartQA preset trains the student without LoRA, installs the
random selector at decoder layer 0, and retains 10% of visual tokens. Set
`SELECTOR` and `KEEP_RATIO` explicitly when evaluating another research
policy; the fixed teacher settings remain unchanged.

The script expects `train.parquet` to exist and does not regenerate data or run
benchmarks. Benchmarking is kept separate in
`scripts/benchmark_chartvqa.py`; the older all-in-one convenience wrapper is
`scripts/run_chartvqa_single_gpu.sh`.

W&B credentials are read from the environment by the normal trainer. Do not
put API keys in this README, the parquet files, or shell scripts.
For an offline smoke test without W&B credentials, set `USE_WANDB=false`.

## Reproducibility checklist

Before a run, verify that:

```bash
python3 -m py_compile scripts/prepare_chartvqa_opd.py
bash -n scripts/run_chartvqa_opd_training.sh
```

The contract tests also check that the two image paths and two prompt columns
remain identical, and that the fixed teacher path is explicit in the launcher.
