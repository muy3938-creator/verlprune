#!/usr/bin/env bash

set -euo pipefail

# One-card ChartQA experiment. The command intentionally runs small, resumable
# slices first; increase the *_LIMIT values after the first successful pass.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-VL-3B-Instruct}"
CHARTVQA_SOURCE="${CHARTVQA_SOURCE:-/root/data/chartvqa}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/root/experiments/chartvqa-learned-budget}"
TRAIN_LIMIT="${TRAIN_LIMIT:-256}"
VAL_LIMIT="${VAL_LIMIT:-128}"
TEST_LIMIT="${TEST_LIMIT:-128}"
TRAINING_STEPS="${TRAINING_STEPS:-10}"
KEEP_RATIO="${KEEP_RATIO:-0.10}"
TOP_P="${TOP_P:-0.95}"
BUDGET_SCHEDULE="${BUDGET_SCHEDULE:-[0.99,0.97,0.95,0.90]}"
MIN_KEEP_RATIO="${MIN_KEEP_RATIO:-0.05}"
MAX_KEEP_RATIO="${MAX_KEEP_RATIO:-0.50}"
PRUNE_LAYER="${PRUNE_LAYER:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-vision-opd-chartvqa}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
MAX_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + RESPONSE_LENGTH))

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_PLUGINS=vision_opd_token_pruning
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${EXPERIMENT_DIR}"

python3 "${PROJECT_ROOT}/scripts/prepare_chartvqa_opd.py" \
  --dataset "${CHARTVQA_SOURCE}" \
  --output-dir "${EXPERIMENT_DIR}/data" \
  --train-limit "${TRAIN_LIMIT}" \
  --val-limit "${VAL_LIMIT}" \
  --test-limit "${TEST_LIMIT}"

COMMON=(
  --model "${MODEL_PATH}"
  --dataset "${EXPERIMENT_DIR}/data"
  --split validation
  --limit "${VAL_LIMIT}"
  --max-new-tokens "${RESPONSE_LENGTH}"
  --wandb-project "${WANDB_PROJECT}"
)

python3 "${PROJECT_ROOT}/scripts/benchmark_chartvqa.py" "${COMMON[@]}" \
  --mode baseline \
  --output "${EXPERIMENT_DIR}/benchmark/dense-baseline.json" \
  --wandb-run dense-baseline

python3 "${PROJECT_ROOT}/scripts/benchmark_chartvqa.py" "${COMMON[@]}" \
  --mode random \
  --keep-ratio "${KEEP_RATIO}" \
  --layer "${PRUNE_LAYER}" \
  --output "${EXPERIMENT_DIR}/benchmark/random-${KEEP_RATIO}.json" \
  --wandb-run random-${KEEP_RATIO}

python3 "${PROJECT_ROOT}/scripts/benchmark_chartvqa.py" "${COMMON[@]}" \
  --mode top_p \
  --keep-ratio "${KEEP_RATIO}" \
  --top-p "${TOP_P}" \
  --layer "${PRUNE_LAYER}" \
  --output "${EXPERIMENT_DIR}/benchmark/top-p-${TOP_P}.json" \
  --wandb-run top-p-${TOP_P}

# The training run uses rank-8 LoRA and the same top-p selector. Override any
# argument after this script when running on a larger slice or resuming.
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${EXPERIMENT_DIR}/data/train.parquet" \
OUTPUT_DIR="${EXPERIMENT_DIR}/training" \
KEEP_RATIO="${KEEP_RATIO}" \
SELECTOR="vision_pulse" \
SELECTOR_KWARGS="{budget_mode:top_p,top_p:${TOP_P},budget_schedule:${BUDGET_SCHEDULE},temperature:1.0,min_keep_ratio:${MIN_KEEP_RATIO},max_keep_ratio:${MAX_KEEP_RATIO},capture_capacity:256}" \
PREFILL_KEEP_RATIO="${KEEP_RATIO}" \
PREFILL_SELECTOR="random" \
PRUNE_AFTER_LAYER="${PRUNE_LAYER}" \
TOTAL_TRAINING_STEPS="${TRAINING_STEPS}" \
"${PROJECT_ROOT}/scripts/run_vision_pruning_experiment.sh" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_SEQUENCE_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_SEQUENCE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_SEQUENCE_LENGTH}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.group_name=chartvqa-learned-budget

echo "Training complete. Set TRAINED_ADAPTER to the saved LoRA adapter and rerun benchmark_chartvqa.py in top_p mode for the post-training score."
