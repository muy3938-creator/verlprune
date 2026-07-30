"""Out-of-tree vLLM plugin for Pre-LLM visual token pruning.

Integrates Pre-LLM visual token selection into vLLM's prefill and generation pipeline:
1. Truncates prompt placeholders to K visual tokens.
2. Selects visual embeddings and 3D mROPE position IDs via select_vision_tokens.
3. Registers selection indices into thread-safe side-channel registry (_PRUNING_REQUEST_REGISTRY).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

try:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLDummyInputsBuilder,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLMultiModalProcessor,
        Qwen2_5_VLProcessingInfo,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.evs import compute_mrope_for_media
    from vllm.multimodal.inputs import MultiModalKwargsItems
    from vllm.multimodal.parse import MultiModalDataItems
    from vllm.multimodal.processing import PromptUpdate

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from verl.models.vision_token_pruning.pre_llm_pruner import select_vision_tokens
from verl.models.vision_token_pruning.vllm_transport import register_request_selection

if HAS_VLLM:

    class _PrunedImagePromptMixin:
        """Prompt processor mixin: truncates expanded image token placeholders in the prompt."""

        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
            hf_config = self.info.get_hf_config()
            pruning_config = getattr(hf_config, "vision_token_pruning", {})
            keep_ratio = pruning_config.get("keep_ratio", 0.5)

            for update in updates:
                if update.modality != "image":
                    continue
                original_replacement = update.replacement

                def pruned_replacement(item_idx: int, replacement=original_replacement):
                    tokens = replacement(item_idx) if callable(replacement) else replacement
                    n_keep = max(1, int(len(tokens) * keep_ratio))
                    return tokens[:n_keep]

                update.replacement = pruned_replacement
            return updates

    class VerlPrunedQwen2_5VLMultiModalProcessor(
        _PrunedImagePromptMixin,
        Qwen2_5_VLMultiModalProcessor,
    ):
        pass

    class _VLLMPreLLMPruningMixin:
        """Model mixin for vLLM visual embedding selection & 3D mROPE position alignment."""

        supports_multimodal_pruning = True

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            pruning_config = getattr(self.config, "vision_token_pruning", {})
            self._keep_ratio = pruning_config.get("keep_ratio", 0.5)
            self._method = pruning_config.get("method", "random")

            # Force vLLM to enable multimodal pruning flow for images when keep_ratio < 1.0
            if self._keep_ratio < 1.0 and getattr(self, "model_config", None) is not None:
                if self.model_config.multimodal_config is not None:
                    if getattr(self.model_config.multimodal_config, "video_pruning_rate", None) is None:
                        self.model_config.multimodal_config.video_pruning_rate = 1.0 - self._keep_ratio

        def _prune_and_annotate_images(
            self,
            image_embeds_split: tuple[torch.Tensor, ...],
            image_grid_thw: torch.Tensor,
            request_id: str | None = None,
        ) -> tuple[torch.Tensor, ...]:
            merge_size = self.visual.spatial_merge_size
            output = []
            for embeddings, grid_thw in zip(image_embeds_split, image_grid_thw.tolist(), strict=True):
                kept_indices = select_vision_tokens(
                    embeddings,
                    keep_ratio=self._keep_ratio,
                    method=self._method,
                    image_grid_thw=torch.tensor(grid_thw, device=embeddings.device),
                )
                if request_id is not None:
                    register_request_selection(request_id, kept_indices)

                positions = compute_mrope_for_media(grid_thw, merge_size).to(
                    embeddings.device, non_blocking=True
                )
                # Concatenate only visual embeddings + 3D mROPE position IDs (no metadata columns added)
                output.append(torch.cat([embeddings[kept_indices], positions[kept_indices]], dim=1))
            return tuple(output)

    @MULTIMODAL_REGISTRY.register_processor(
        VerlPrunedQwen2_5VLMultiModalProcessor,
        info=Qwen2_5_VLProcessingInfo,
        dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
    )
    class VerlPrunedQwen2_5VLForConditionalGeneration(
        _VLLMPreLLMPruningMixin,
        Qwen2_5_VLForConditionalGeneration,
    ):
        def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
            return self._prune_and_annotate_images(
                image_embeds_split,
                image_input["image_grid_thw"],
            )
