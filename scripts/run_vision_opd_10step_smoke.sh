#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/tmp/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-/tmp/opd_data/train.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/opd_10step}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
SELECTOR="${SELECTOR:-random}"
PRUNE_AFTER_LAYER="${PRUNE_AFTER_LAYER:--1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
SAVE_FREQ="${SAVE_FREQ:-1}"
RESUME_MODE="${RESUME_MODE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.1}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_PLUGINS=vision_opd_token_pruning
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=0

DATA_ARGS=(
    "data.train_files=[\"${TRAIN_FILE}\"]"
    "data.val_files=[]"
    "data.train_max_samples=16"
    "data.train_batch_size=1"
    "data.max_prompt_length=256"
    "data.max_response_length=32"
    "data.filter_overlong_prompts=False"
    "data.shuffle=False"
    "data.return_multi_modal_inputs=True"
    "data.image_key=images"
    "data.dataloader_num_workers=0"
)

MODEL_ARGS=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_fused_kernels=False"
    "actor_rollout_ref.model.lora_rank=8"
    "actor_rollout_ref.model.lora_alpha=16"
    "actor_rollout_ref.model.target_modules=\".*language_model.layers.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\""
    "actor_rollout_ref.model.vision_token_pruning.enabled=True"
    "actor_rollout_ref.model.vision_token_pruning.keep_ratio=${KEEP_RATIO}"
    "actor_rollout_ref.model.vision_token_pruning.selector=${SELECTOR}"
    "actor_rollout_ref.model.vision_token_pruning.prune_after_layer=${PRUNE_AFTER_LAYER}"
)

ACTOR_ARGS=(
    "actor_rollout_ref.actor.ppo_mini_batch_size=1"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=False"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.calculate_entropy=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    "actor_rollout_ref.actor.optim.lr=1e-5"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vopd"
)

DISTILLATION_ARGS=(
    "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
    "actor_rollout_ref.actor.self_distillation.distillation_topk=32"
    "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
    "actor_rollout_ref.actor.self_distillation.max_reprompt_len=256"
    "actor_rollout_ref.actor.self_distillation.is_clip=2.0"
    "actor_rollout_ref.actor.self_distillation.teacher_always_on=True"
    "actor_rollout_ref.actor.self_distillation.teacher_model_source=current"
    "actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images"
    "actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher=False"
    "actor_rollout_ref.actor.self_distillation.alpha=0.5"
    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
)

ROLLOUT_ARGS=(
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.load_format=safetensors"
    "actor_rollout_ref.rollout.layered_summon=True"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.enable_prefix_caching=False"
    "actor_rollout_ref.rollout.max_num_batched_tokens=512"
    "actor_rollout_ref.rollout.max_model_len=512"
    "actor_rollout_ref.rollout.max_num_seqs=4"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.response_length=32"
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "actor_rollout_ref.rollout.agent.num_workers=1"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=134217728"
)

TRAINER_ARGS=(
    "critic.enable=False"
    "reward_model.enable=False"
    "reward_model.use_reward_loop=False"
    "custom_reward_function.path=null"
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=False"
    "algorithm.use_kl_in_reward=False"
    "algorithm.rollout_correction.rollout_is=token"
    "algorithm.rollout_correction.rollout_is_threshold=2.0"
    "trainer.balance_batch=False"
    "trainer.n_gpus_per_node=1"
    "trainer.nnodes=1"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.total_epochs=1"
    "trainer.logger=[\"console\"]"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=-1"
    "trainer.val_before_train=False"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints"
    "trainer.rollout_data_dir=${OUTPUT_DIR}/rollouts"
)

python3 -m verl.trainer.main_ppo --config-name vopd \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${DISTILLATION_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=True" \
    "${TRAINER_ARGS[@]}" \
    "$@"
