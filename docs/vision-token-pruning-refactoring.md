# Vision token pruning refactoring

## Refactoring baseline

This document describes the structural cleanup performed after commit
`f055b44` (`feat: add rollout-replayed vision token pruning`). The feature and
its pre-refactoring design are documented separately in
`docs/vision-token-pruning-implementation.md`.

The refactoring preserves the validated layer-0 pruning behavior. It does not
change the selection algorithm, keep-ratio semantics, teacher inputs, or the
physical sequence seen by FlashAttention.

## Resulting execution flow

1. The vLLM out-of-tree model physically selects and gathers image embeddings.
2. vLLM returns the exact retained image-token indices with the rollout.
3. The agent loop stores those indices in the explicit
   `vision_token_selection` field and attaches them to the sample's multimodal
   inputs.
4. The actor validates and replays the same selection before remove-padding.
5. Qwen removes the matching visual features, so token positions and visual
   embeddings stay aligned.
6. The OPD teacher strips all pruning metadata and executes with its complete
   teacher image.

Consequently, the student's removed tokens are absent from the packed varlen
sequence and are not included in subsequent FlashAttention computation. This
is physical layer-0 pruning, not zeroing an embedding while retaining its
attention position.

## Main refactoring changes

### One configuration boundary

`coerce_vision_token_pruning_config` now normalizes Hydra mappings, dataclass
instances, and missing configuration in one place. Actor and rollout code no
longer duplicate type checks or construct configuration objects independently.

### Actor preparation is an explicit operation

`prepare_actor_pruning_inputs` now owns the complete actor/teacher preparation
contract:

- the student path validates and replays rollout selection;
- the teacher path keeps the original attention mask;
- both paths remove internal protocol fields before calling the model;
- invalid states, such as pruning without `image_token_id`, fail at the
  integration boundary.

This removes the pruning-specific import and branching block from the middle of
`DataParallelPPOActor._forward_micro_batch`. The actor now consumes a small,
typed `PreparedActorPruningInputs` result before its existing packing logic.

The old generic function name was also replaced with
`replay_rollout_selection_on_attention_mask`, which states that training is
replaying the exact rollout choice rather than independently sampling a mask.

### vLLM policy is separated from server transport

The new `VisionTokenPruningRollout` class contains the pruning-specific rules
that were previously interleaved with HTTP server construction:

- select the Qwen2.5-VL or Qwen3-VL OOT architecture;
- construct vLLM launch overrides;
- reject routing-replay channel conflicts;
- validate one-image/no-video requests;
- count original expanded image tokens;
- decode returned retained indices.

`vLLMHttpServer` is left responsible for generic process launch and request
transport. The independently testable `resolve_rollout_max_model_len` helper
also makes the checkpoint-limit rule explicit.

### Selection transport has a named schema field

`AgentLoopOutput.vision_token_selection` is now the sole agent-loop transport
field for pruning selection. It replaces use of the unrestricted
`extra_fields` dictionary for this protocol, while ordinary tool and reward
metadata can continue to use `extra_fields`.

This makes the rollout-to-training contract visible in model validation,
constructor calls, and type hints.

### Shared model and plugin helpers

- Qwen3-VL now uses `prune_visual_embedding_outputs` to prune the main visual
  output and every aligned DeepStack output together.
- Qwen2.5-VL and Qwen3-VL import pruning helpers at module scope instead of
  during every forward.
- vLLM metadata encoding/decoding is centralized in named helpers.
- The Qwen3-specific fourth MRoPE axis is represented by a class capability
  flag, eliminating a boolean argument threaded through shared code.
- Keep-ratio extraction and validation is performed once by `_get_keep_ratio`.

### Smaller supporting cleanup

- OPD worker selection now uses the direct names `uses_opd_teacher` and
  `needs_separate_opd_teacher`.
- LoRA base-layer name conversion shares one weight/bias code path.
- The direct GPU smoke test uses the renamed replay API.
- The 10-step launcher groups data, model, actor, distillation, rollout, and
  trainer settings into arrays. Important smoke parameters can now be changed
  with `KEEP_RATIO`, `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `RESUME_MODE`, and
  `GPU_MEMORY_UTILIZATION` without editing the script.

## Tests added or strengthened

- Configuration normalization is tested independently.
- Actor preparation verifies that teacher inputs cannot leak pruning metadata.
- Rollout launch, request validation, selection decoding, model support, and
  routing-channel conflict behavior are covered without starting vLLM.
- Existing protocol, physical pruning, backward-pass, OOT plugin, LoRA sync,
  source-contract, and real-vLLM helper tests remain in the focused suite.

## Validation after refactoring

### Static and focused tests

- Python compilation passed for all changed runtime modules and tests.
- `git diff --check` and shell syntax validation passed.
- Ruff passed on the new pruning package, pruning tests, OOT plugin, Qwen call
  sites, rollout server, single-turn loop, and direct GPU smoke test.
- On the CNB H20 environment, the complete focused suite reported
  `26 passed, 1 skipped`.

### Native 10-step GPU run

A fresh run, separate from the pre-refactoring checkpoint, completed with exit
code 0 using:

- NVIDIA H20;
- Qwen2.5-VL-3B-Instruct;
- vLLM 0.18.0 and transformers 4.57.6;
- the `VerlRandomPrunedQwen2_5VLForConditionalGeneration` OOT model;
- a 0.5 visual-token keep ratio;
- rank-8 text-decoder LoRA;
- remove-padding and the Qwen2.5-VL FlashAttention patch;
- current-policy OPD teacher with bounding-box teacher images.

All ten rollout files were produced. At every step, teacher activation and
teacher image-swap fractions were 1.0, while policy fallback and empty-target
fractions were 0.0. The run produced finite non-zero VOPD losses and gradients;
step 10 reported a VOPD loss of `0.0099025` and gradient norm `0.312231`. Peak
allocated GPU memory was approximately 36.12 GiB. The final
`global_step_10` checkpoint contains the LoRA adapter and training state.

## Remaining scope and future extension

This cleanup intentionally retains the simple, validated baseline:

- pruning occurs before decoder layer 0;
- selection is random with mandatory final-anchor retention;
- each pruned request contains exactly one image and no video;
- pruning metadata currently uses vLLM's routed-expert return channel, so it
  cannot be enabled simultaneously with routing replay;
- arbitrary decoder-layer KV removal remains a future advanced feature.

These constraints are checked explicitly rather than silently approximated.
The new actor and rollout boundaries provide clearer extension points for a
future selector, transport channel, or arbitrary-layer implementation.
