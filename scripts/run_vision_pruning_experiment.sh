#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/tmp/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-/tmp/opd_data/train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/vision_pruning_experiment}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
SELECTOR="${SELECTOR:-vision_pulse}"
SELECTOR_KWARGS="${SELECTOR_KWARGS:-}"
PREFILL_KEEP_RATIO="${PREFILL_KEEP_RATIO:-0.5}"
PREFILL_SELECTOR="${PREFILL_SELECTOR:-embedding_norm}"
PREFILL_SELECTOR_KWARGS="${PREFILL_SELECTOR_KWARGS:-}"
PREFILL_PRUNE_AFTER_LAYER="${PREFILL_PRUNE_AFTER_LAYER:--1}"
SELECTOR_INPUT="${SELECTOR_INPUT:-decode_query}"
PRE_PRUNING_BACKEND="${PRE_PRUNING_BACKEND:-flash}"
PRUNE_AFTER_LAYER="${PRUNE_AFTER_LAYER:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
SAVE_FREQ="${SAVE_FREQ:-1}"
RESUME_MODE="${RESUME_MODE:-auto}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_PLUGINS=vision_opd_token_pruning
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0

STRATEGY_ARGS=()
if [[ -n "${SELECTOR_KWARGS}" ]]; then
    STRATEGY_ARGS+=("actor_rollout_ref.model.vision_token_pruning.selector_kwargs=${SELECTOR_KWARGS}")
fi
if [[ -n "${PREFILL_SELECTOR_KWARGS}" ]]; then
    STRATEGY_ARGS+=("actor_rollout_ref.model.vision_token_pruning.prefill_selector_kwargs=${PREFILL_SELECTOR_KWARGS}")
fi

python3 -m verl.trainer.main_ppo --config-name vision_pruning_experiment \
    "data.train_files=[\"${TRAIN_FILE}\"]" \
    "data.val_files=[]" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.vision_token_pruning.keep_ratio=${KEEP_RATIO}" \
    "actor_rollout_ref.model.vision_token_pruning.selector=${SELECTOR}" \
    "actor_rollout_ref.model.vision_token_pruning.prefill_keep_ratio=${PREFILL_KEEP_RATIO}" \
    "actor_rollout_ref.model.vision_token_pruning.prefill_selector=${PREFILL_SELECTOR}" \
    "actor_rollout_ref.model.vision_token_pruning.prefill_prune_after_layer=${PREFILL_PRUNE_AFTER_LAYER}" \
    "actor_rollout_ref.model.vision_token_pruning.selector_input=${SELECTOR_INPUT}" \
    "actor_rollout_ref.model.vision_token_pruning.pre_pruning_backend=${PRE_PRUNING_BACKEND}" \
    "actor_rollout_ref.model.vision_token_pruning.prune_after_layer=${PRUNE_AFTER_LAYER}" \
    "${STRATEGY_ARGS[@]}" \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
    "trainer.rollout_data_dir=${OUTPUT_DIR}/rollouts" \
    "$@"
