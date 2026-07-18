# Vision token pruning implementation before refactoring

## Comparison baseline

This document describes the first end-to-end implementation relative to the
upstream Vision-OPD commit `c8a8fdd` (`update README`).  The implementation was
validated before refactoring and is intentionally documented as a separate
checkpoint so that later structural changes can be reviewed independently from
the feature itself.

## Goal

The implementation adds a layer-0 visual-token pruning baseline to Vision-OPD:

- vLLM physically keeps a random subset of image tokens during rollout.
- The rollout returns the exact retained token indices.
- The actor replays that selection by masking the same image-token positions
  before its remove-padding/FlashAttention forward.
- The OPD teacher reuses the current actor weights but receives the complete
  image and explicitly bypasses pruning.
- The final visual token is always retained as the MRoPE anchor.

The first version deliberately supports one image per request, no video, and
Qwen2.5-VL/Qwen3-VL at decoder layer 0.

## Main changes

### Configuration and protocol

- Added `VisionTokenPruningConfig` with `enabled` and `keep_ratio` validation.
- Added a versioned `VisionTokenSelection` wire format containing the original
  token count and the exact sorted retained indices.
- Added deterministic random selection and mask-construction helpers.
- Added model and actor configuration fields plus Hydra YAML and launch-script
  overrides.

### vLLM out-of-tree integration

- Added the `vision_opd_token_pruning` vLLM plugin entry point.
- Registered pruned Qwen2.5-VL and Qwen3-VL model implementations without
  modifying the installed vLLM package.
- Pruned image embeddings before decoder execution and shortened image prompt
  replacements consistently.
- Reused vLLM's routed-expert return channel to transport compact, positive,
  one-based retained indices back to VerL.
- Added validation for model type, image count, keep ratio, prompt expansion,
  MRoPE anchor retention, and metadata shape.
- Fixed the async server so an explicit `max_model_len` is not overwritten by
  the checkpoint's advertised context length.

### VerL rollout-to-training replay

- Extended `TokenOutput` with `vision_token_selection`.
- Propagated selection metadata through the single-turn agent loop and attached
  it to each sample's non-tensor multimodal inputs.
- Changed the internal multimodal schema to allow structured rollout metadata
  alongside processor tensors.
- Verified that actor-side masks exactly match the rollout selection and the
  configured keep ratio before changing the attention mask.

### Actor and teacher behavior

- Added pruning-aware forwarding to the data-parallel actor.
- Required remove-padding mode so masked image tokens are physically absent
  from the varlen FlashAttention sequence.
- Removed the matching visual embeddings before image embeddings are scattered
  into Qwen2.5-VL/Qwen3-VL input embeddings.
- Explicitly disabled pruning for the OPD teacher forward.
- Avoided building an unused second reference model when
  `teacher_model_source=current`; the teacher and student share weights but use
  different visual inputs.

### LoRA and runtime compatibility

- Pinned the tested pair `vllm==0.18.0` and `transformers==4.57.6`.
- Limited LoRA wrapper renaming to modules actually selected by the PEFT
  configuration, preventing visual and text projection names from being mixed
  during vLLM weight synchronization.
- Added a reproducible 10-step smoke script using text-decoder LoRA,
  safetensors base loading, layered LoRA synchronization, CPU offload, bounded
  KV cache, and automatic checkpoint recovery.

## Files added

- `verl/models/vision_token_pruning/`: configuration, protocol, selection, and
  actor runtime helpers.
- `verl/vllm_plugins/`: plugin registration and pruned Qwen-VL models.
- `tests/vision_token_pruning/`: protocol, runtime, source-contract, integration,
  vLLM-contract, and LoRA synchronization tests.
- `scripts/gpu_e2e_smoke.py`: direct GPU integration smoke test.
- `scripts/run_vision_opd_10step_smoke.sh`: native VerL/vLLM 10-step test.

## Validation evidence

The native Vision-OPD loop completed 10 logical training steps on an NVIDIA H20
with Qwen2.5-VL-3B-Instruct, 224×224 images, rank-8 LoRA, and a 0.5 visual-token
keep ratio.  The run produced non-zero OPD/JSD losses, non-zero gradients,
teacher image-swap fraction 1.0, policy fallback fraction 0.0, saved rollout
generations, and a final `global_step_10` LoRA adapter.  The focused GPU test
suite reported `21 passed, 1 skipped`.

## Known structural issues in this checkpoint

This checkpoint is functionally complete but not yet optimally organized:

- Pruning imports and preprocessing decisions are embedded inside large actor
  and agent-loop methods.
- vLLM server setup, request validation, and response decoding are interleaved
  with generic server logic.
- Selection transport uses a low-level replay channel and contains encoding
  details inside the model plugin.
- Qwen2.5-VL and Qwen3-VL contain similar pruning call sites.
- The smoke script has many inline overrides and no grouped explanation of why
  each memory setting is required.

These issues are the target of the subsequent refactoring commit and its
separate document.
