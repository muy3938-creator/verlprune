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
from verl.models.vision_token_pruning.selectors import select_random_visual_tokens

_METADATA_RADIX = 256


def _pruning_config(hf_config: Any) -> dict[str, Any]:
    config = getattr(hf_config, "vision_token_pruning", None)
    if not isinstance(config, dict):
        raise ValueError("pruned vLLM model requires hf_config.vision_token_pruning")
    keep_ratio = float(config.get("keep_ratio", 0.0))
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError("pruned vLLM model requires keep_ratio in (0, 1)")
    return config


class _PrunedImagePromptMixin:
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        keep_ratio = float(_pruning_config(self.info.get_hf_config())["keep_ratio"])
        for update in updates:
            if update.modality != "image":
                continue
            original_replacement = update.replacement

            def pruned_replacement(item_idx: int, replacement=original_replacement):
                tokens = replacement(item_idx) if callable(replacement) else replacement
                return tokens[: compute_keep_count(len(tokens), keep_ratio)]

            update.replacement = pruned_replacement
        return updates


class VerlRandomPrunedQwen2_5VLMultiModalProcessor(
    _PrunedImagePromptMixin,
    Qwen2_5_VLMultiModalProcessor,
):
    pass


class VerlRandomPrunedQwen3VLMultiModalProcessor(
    _PrunedImagePromptMixin,
    Qwen3VLMultiModalProcessor,
):
    pass


class _RandomPruningMixin:
    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._keep_ratio = float(_pruning_config(self.config)["keep_ratio"])
        expected_pruning_rate = 1.0 - self._keep_ratio
        if self.video_pruning_rate is None or abs(float(self.video_pruning_rate) - expected_pruning_rate) > 1e-8:
            raise ValueError(
                "vLLM video_pruning_rate must equal 1 - vision_token_pruning.keep_ratio "
                "to activate consistent image pruning"
            )
        self._pending_selection_indices: list[torch.Tensor] = []
        self._pending_capture_values: torch.Tensor | None = None
        self._selection_seed = int(vllm_config.model_config.seed)
        self._selection_counter = 0

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ):
        self._pending_selection_indices.extend(
            mm[:, -2].long() + mm[:, -1].long() * _METADATA_RADIX
            for mm in multimodal_embeddings
            if len(mm)
        )
        embeddings_without_metadata = tuple(mm[:, :-2] for mm in multimodal_embeddings)
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
        self._pending_selection_indices.clear()

        capture_values = torch.zeros((input_ids.numel(), 1), dtype=torch.int32, device=input_ids.device)
        if encoded.numel():
            if is_multimodal is None or int(is_multimodal.sum()) != encoded.numel():
                raise ValueError("pruned visual embedding count does not match vLLM placeholders")
            capture_values[is_multimodal, 0] = encoded.to(device=input_ids.device, dtype=torch.int32)
        if self._pending_capture_values is not None:
            raise RuntimeError("stale visual-token selection metadata was not consumed by vLLM forward")
        self._pending_capture_values = capture_values
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        capture_values = self._pending_capture_values
        self._pending_capture_values = None
        if capture_values is not None:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
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
        *,
        qwen3: bool,
    ) -> tuple[torch.Tensor, ...]:
        merge_size = self.visual.spatial_merge_size
        output = []
        for embeddings, grid_thw in zip(image_embeds_split, image_grid_thw.tolist(), strict=True):
            if len(embeddings) >= _METADATA_RADIX**2:
                raise ValueError("visual-token selection metadata supports fewer than 65536 image tokens")

            keep_count = compute_keep_count(len(embeddings), self._keep_ratio)
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed(self._selection_seed + self._selection_counter)
            self._selection_counter += 1
            kept_indices = select_random_visual_tokens(
                len(embeddings),
                keep_count,
                device=embeddings.device,
                generator=generator,
            )
            positions = compute_mrope_for_media(grid_thw, merge_size).to(embeddings.device)
            if qwen3:
                positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=1)

            encoded = kept_indices + 1
            metadata = torch.stack(
                [encoded.remainder(_METADATA_RADIX), encoded.div(_METADATA_RADIX, rounding_mode="floor")],
                dim=1,
            ).to(dtype=embeddings.dtype)
            output.append(torch.cat([embeddings[kept_indices], positions[kept_indices], metadata], dim=1))
        return tuple(output)


@MULTIMODAL_REGISTRY.register_processor(
    VerlRandomPrunedQwen2_5VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class VerlRandomPrunedQwen2_5VLForConditionalGeneration(
    _RandomPruningMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._prune_and_annotate_images(
            image_embeds_split,
            image_input["image_grid_thw"],
            qwen3=False,
        )


@MULTIMODAL_REGISTRY.register_processor(
    VerlRandomPrunedQwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class VerlRandomPrunedQwen3VLForConditionalGeneration(
    _RandomPruningMixin,
    Qwen3VLForConditionalGeneration,
):
    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._prune_and_annotate_images(
            image_embeds_split,
            image_input["image_grid_thw"],
            qwen3=True,
        )
