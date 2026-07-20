from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import compute_mrope_for_media
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptUpdate

from verl.models.vision_token_pruning.protocol import compute_keep_count
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine
from verl.models.vision_token_pruning.transport import (
    decode_embedding_selection_metadata,
    encode_embedding_selection_metadata,
)
from verl.vllm_plugins.vision_token_pruning_common import pruning_config_from_hf


class _PrunedImagePromptMixin:
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        config = pruning_config_from_hf(self.info.get_hf_config())
        if config.uses_layerwise_backend and not config.uses_physical_prefill_pruning:
            return updates
        keep_ratio = config.prefill_keep_ratio if config.uses_two_stage_pruning else config.keep_ratio
        assert keep_ratio is not None
        for update in updates:
            if update.modality != "image":
                continue
            original_replacement = update.replacement

            def pruned_replacement(item_idx: int, replacement=original_replacement):
                tokens = replacement(item_idx) if callable(replacement) else replacement
                return tokens[: compute_keep_count(len(tokens), keep_ratio)]

            update.replacement = pruned_replacement
        return updates


class VerlPrunedQwen2_5VLMultiModalProcessor(
    _PrunedImagePromptMixin,
    Qwen2_5_VLMultiModalProcessor,
):
    pass


class VerlPrunedQwen3VLMultiModalProcessor(
    _PrunedImagePromptMixin,
    Qwen3VLMultiModalProcessor,
):
    pass


class _PruningMixin:
    supports_multimodal_pruning = True
    _append_zero_position_axis = False
    _binary_selection_capture = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._pruning_config = pruning_config_from_hf(self.config)
        self._selection_engine = VisionTokenSelectionEngine(
            self._pruning_config,
            seed=int(vllm_config.model_config.seed),
        )
        expected_pruning_rate = 1.0 - self._pruning_config.keep_ratio
        if self.video_pruning_rate is None or abs(float(self.video_pruning_rate) - expected_pruning_rate) > 1e-8:
            raise ValueError(
                "vLLM video_pruning_rate must equal 1 - vision_token_pruning.keep_ratio "
                "to activate consistent image pruning"
            )
        self._pending_selection_indices: list[torch.Tensor] = []
        self._pending_original_counts: list[torch.Tensor] = []
        self._pending_capture_values: tuple[torch.Tensor, torch.Tensor] | None = None

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ):
        for mm in multimodal_embeddings:
            if not len(mm):
                continue
            if self._binary_selection_capture:
                self._pending_selection_indices.append(
                    decode_embedding_selection_metadata(mm[:, -4:-2])
                )
                self._pending_original_counts.append(
                    decode_embedding_selection_metadata(mm[:, -2:])
                )
            else:
                self._pending_selection_indices.append(decode_embedding_selection_metadata(mm))
                self._pending_original_counts.append(mm.new_zeros(len(mm), dtype=torch.long))
        metadata_columns = 4 if self._binary_selection_capture else 2
        embeddings_without_metadata = tuple(
            mm[:, :-metadata_columns] for mm in multimodal_embeddings
        )
        return super().recompute_mrope_positions(
            input_ids,
            embeddings_without_metadata,
            mrope_positions,
            num_computed_tokens,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        encoded = (
            torch.cat(self._pending_selection_indices)
            if self._pending_selection_indices
            else input_ids.new_empty(0)
        )
        original_counts = (
            torch.cat(self._pending_original_counts)
            if self._pending_selection_indices
            else input_ids.new_empty(0)
        )
        self._pending_selection_indices.clear()
        self._pending_original_counts.clear()

        capture_values = torch.zeros((input_ids.numel(), 1), dtype=torch.int32, device=input_ids.device)
        capture_counts = torch.zeros(input_ids.numel(), dtype=torch.int32, device=input_ids.device)
        if encoded.numel():
            if is_multimodal is None or int(is_multimodal.sum()) != encoded.numel():
                raise ValueError("pruned visual embedding count does not match vLLM placeholders")
            capture_values[is_multimodal, 0] = encoded.to(device=input_ids.device, dtype=torch.int32)
            capture_counts[is_multimodal] = original_counts.to(
                device=input_ids.device,
                dtype=torch.int32,
            )
        if self._pending_capture_values is not None:
            raise RuntimeError("stale visual-token selection metadata was not consumed by vLLM forward")
        self._pending_capture_values = (capture_values, capture_counts)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        pending_capture = self._pending_capture_values
        self._pending_capture_values = None
        if pending_capture is not None:
            capture_values, capture_counts = pending_capture
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                if self._binary_selection_capture:
                    encoded = capture_values[:, 0].long()
                    layer_count = int(capturer._device_buffer.shape[1])
                    if layer_count < 33:
                        raise ValueError("Qwen3 selection capture requires at least 33 decoder layers")
                    for bit in range(16):
                        capturer.capture(bit, encoded.bitwise_right_shift(bit).bitwise_and(1)[:, None])
                        capturer.capture(
                            16 + bit,
                            capture_counts.long().bitwise_right_shift(bit).bitwise_and(1)[:, None],
                        )
                    # The final layer distinguishes the binary Qwen3 layout
                    # from legacy layer-0 integer values.  The middle bits
                    # carry the original visual-token count, so replay does
                    # not depend on model-specific processor grid formulas.
                    capturer.capture(layer_count - 1, encoded.gt(0).to(torch.int32)[:, None])
                else:
                    capturer.capture(0, capture_values)
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def _prune_and_annotate_images(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        merge_size = self.visual.spatial_merge_size
        output = []
        for embeddings, grid_thw in zip(image_embeds_split, image_grid_thw.tolist(), strict=True):
            kept_indices = self._selection_engine.select(
                embeddings,
                grid_thw=grid_thw,
            )
            positions = compute_mrope_for_media(grid_thw, merge_size).to(embeddings.device)
            if self._append_zero_position_axis:
                positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=1)

            metadata = encode_embedding_selection_metadata(kept_indices, dtype=embeddings.dtype)
            if self._binary_selection_capture:
                original_count_metadata = encode_embedding_selection_metadata(
                    kept_indices.new_full((len(kept_indices),), len(embeddings) - 1),
                    dtype=embeddings.dtype,
                )
                metadata = torch.cat((metadata, original_count_metadata), dim=1)
            output.append(torch.cat([embeddings[kept_indices], positions[kept_indices], metadata], dim=1))
        return tuple(output)


@MULTIMODAL_REGISTRY.register_processor(
    VerlPrunedQwen2_5VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class VerlPrunedQwen2_5VLForConditionalGeneration(
    _PruningMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._prune_and_annotate_images(
            image_embeds_split,
            image_input["image_grid_thw"],
        )


@MULTIMODAL_REGISTRY.register_processor(
    VerlPrunedQwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class VerlPrunedQwen3VLForConditionalGeneration(
    _PruningMixin,
    Qwen3VLForConditionalGeneration,
):
    _append_zero_position_axis = True
    _binary_selection_capture = True

    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._prune_and_annotate_images(
            image_embeds_split,
            image_input["image_grid_thw"],
        )


# Compatibility aliases for checkpoints and launch commands from the random-only prototype.
_RandomPruningMixin = _PruningMixin
VerlRandomPrunedQwen2_5VLMultiModalProcessor = VerlPrunedQwen2_5VLMultiModalProcessor
VerlRandomPrunedQwen3VLMultiModalProcessor = VerlPrunedQwen3VLMultiModalProcessor
VerlRandomPrunedQwen2_5VLForConditionalGeneration = VerlPrunedQwen2_5VLForConditionalGeneration
VerlRandomPrunedQwen3VLForConditionalGeneration = VerlPrunedQwen3VLForConditionalGeneration
