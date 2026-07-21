#!/usr/bin/env bash

set -euo pipefail

# Train the ChartQA OPD experiment after prepare_chartvqa_opd.py has created
# the parquet files.  This is intentionally a training-only launcher: data
# conversion and benchmark generation are separate, auditable steps.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-VL-3B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:-/root/experiments/chartvqa-opd/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/experiments/chartvqa-opd/training}"
TRAINING_CONFIG="${TRAINING_CONFIG:-chartvqa_opd_lora}"
KEEP_RATIO="${KEEP_RATIO:-0.10}"
SELECTOR="${SELECTOR:-random}"
SELECTOR_KWARGS="${SELECTOR_KWARGS:-{}}"
SELECTOR_INPUT="${SELECTOR_INPUT:-vision_embedding}"
PREFILL_KEEP_RATIO="${PREFILL_KEEP_RATIO:-none}"
PREFILL_SELECTOR="${PREFILL_SELECTOR:-embedding_norm}"
PREFILL_PRUNE_AFTER_LAYER="${PREFILL_PRUNE_AFTER_LAYER:--1}"
PRUNE_AFTER_LAYER="${PRUNE_AFTER_LAYER:-0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
ACTOR_LR="${ACTOR_LR:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
MAX_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + RESPONSE_LENGTH))
WANDB_PROJECT="${WANDB_PROJECT:-vision-opd-chartvqa}"
USE_WANDB="${USE_WANDB:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.10}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Missing ChartQA training parquet: ${TRAIN_FILE}" >&2
    echo "Run scripts/prepare_chartvqa_opd.py first." >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing student model directory: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
    echo "Missing fixed teacher model directory: ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_PLUGINS=vision_opd_token_pruning
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0

cleanup_runtime() {
    set +e
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/tmp/vision-opd-ray-stop.log 2>&1 || true
    fi
    for pattern in raylet gcs_server vllm main_ppo; do
        pkill -TERM -f "${pattern}" 2>/dev/null || true
    done
    sleep 2
    for pattern in raylet gcs_server vllm main_ppo; do
        pkill -KILL -f "${pattern}" 2>/dev/null || true
    done
}

# This launcher owns the Ray/vLLM processes in its CNB workspace.  Always
# clean them after success, failure, or Ctrl-C; do not touch the independent
# ./gg keepalive process.
trap cleanup_runtime EXIT INT TERM

if command -v ray >/dev/null 2>&1; then
    ray stop --force >/tmp/vision-opd-ray-stop-before-run.log 2>&1 || true
fi

DATA_ARGS=()
if [[ -n "${VAL_FILE}" ]]; then
    DATA_ARGS+=("data.val_files=[\"${VAL_FILE}\"]")
fi

OPTIM_ARGS=()
if [[ -n "${ACTOR_LR}" ]]; then
    OPTIM_ARGS+=("actor_rollout_ref.actor.optim.lr=${ACTOR_LR}")
fi

LOGGER_ARGS=("trainer.logger=['console','wandb']")
if [[ "${USE_WANDB}" == "false" ]]; then
    LOGGER_ARGS=("trainer.logger=['console']")
fi

env \
    CONFIG_NAME="${TRAINING_CONFIG}" \
    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_FILE="${TRAIN_FILE}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    KEEP_RATIO="${KEEP_RATIO}" \
    SELECTOR="${SELECTOR}" \
    SELECTOR_KWARGS="${SELECTOR_KWARGS}" \
    SELECTOR_INPUT="${SELECTOR_INPUT}" \
    PREFILL_KEEP_RATIO="${PREFILL_KEEP_RATIO}" \
    PREFILL_SELECTOR="${PREFILL_SELECTOR}" \
    PREFILL_PRUNE_AFTER_LAYER="${PREFILL_PRUNE_AFTER_LAYER}" \
    PRUNE_AFTER_LAYER="${PRUNE_AFTER_LAYER}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    PYTHONPATH="${PYTHONPATH}" \
    "${PROJECT_ROOT}/scripts/run_vision_pruning_experiment.sh" \
    "${DATA_ARGS[@]}" \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed \
    actor_rollout_ref.actor.self_distillation.teacher_model_path="${TEACHER_MODEL_PATH}" \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images \
    actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_SEQUENCE_LENGTH}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=false \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_SEQUENCE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_SEQUENCE_LENGTH}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.group_name=chartvqa-opd-fixed-teacher \
    "${LOGGER_ARGS[@]}" \
    "${OPTIM_ARGS[@]}"
