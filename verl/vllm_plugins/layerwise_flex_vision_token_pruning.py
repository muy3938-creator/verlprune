"""Layerwise visual-token masking on vLLM's native FlexAttention backend.

The integration deliberately leaves vLLM's paged KV layout untouched.  A
boolean sidecar is indexed by the physical slots that vLLM already assigns;
FlexAttention consults it from a score modifier after the configured layer.
Selection can happen either in the vision tower or once from decoder Q/K/V at
the layer boundary.  Later prefill layers and every decode step reuse the same
per-layer sidecar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import vllm.forward_context
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import compute_mrope_for_media
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
    FlexAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine
from verl.models.vision_token_pruning.transport import (
    decode_embedding_selection_metadata,
    encode_embedding_selection_metadata,
)
from verl.vllm_plugins.vision_token_pruning_common import pruning_config_from_hf

_FORWARD_CONTEXT_KEY = "verl_layerwise_flex_vision_pruning"
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class LayerwiseFlexPruningPlan:
    """Algorithm state for one scheduled vLLM forward call."""

    prune_after_layer: int
    selector_input: str
    candidate_ids: torch.Tensor
    query_keep_mask: torch.Tensor | None
    selection_engine: VisionTokenSelectionEngine
    capture_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.prune_after_layer < 0:
            raise ValueError("layerwise Flex pruning requires prune_after_layer >= 0")
        if self.candidate_ids.ndim != 1 or self.candidate_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be a rank-1 integer tensor")
        if self.query_keep_mask is not None and (
            self.query_keep_mask.ndim != 1 or self.query_keep_mask.dtype != torch.bool
        ):
            raise ValueError("query_keep_mask must be a rank-1 bool tensor")


def layer_index_from_name(layer_name: str) -> int:
    match = _LAYER_INDEX_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"cannot determine decoder layer index from {layer_name!r}")
    return int(match.group(1))


def _active_plan(layer: torch.nn.Module) -> tuple[LayerwiseFlexPruningPlan, int] | None:
    context = vllm.forward_context.get_forward_context()
    plan = context.additional_kwargs.get(_FORWARD_CONTEXT_KEY)
    if plan is None:
        return None
    if not isinstance(plan, LayerwiseFlexPruningPlan):
        raise TypeError("invalid layerwise Flex pruning plan in vLLM forward context")
    return plan, layer_index_from_name(layer.layer_name)


def _query_lengths(metadata: FlexAttentionMetadata) -> torch.Tensor:
    return metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]


def _select_from_decoder_states(
    plan: LayerwiseFlexPruningPlan,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: FlexAttentionMetadata,
    *,
    layer_index: int,
) -> None:
    """Run the pure-PyTorch selector once and retain its packed-query mask."""

    if plan.query_keep_mask is not None or not bool((plan.candidate_ids > 0).any()):
        return
    actual_tokens = metadata.num_actual_tokens
    candidate_ids = plan.candidate_ids[:actual_tokens].to(key.device)
    keep = torch.ones(actual_tokens, dtype=torch.bool, device=key.device)
    capture = torch.zeros((actual_tokens, 1), dtype=torch.int32, device=key.device)

    for request_index in range(len(metadata.query_start_loc) - 1):
        start = int(metadata.query_start_loc[request_index].item())
        end = int(metadata.query_start_loc[request_index + 1].item())
        image_positions = (candidate_ids[start:end] > 0).nonzero(as_tuple=False).flatten() + start
        if not len(image_positions):
            continue
        selected_relative = plan.selection_engine.select_decoder_states(
            query_states=query[:actual_tokens].index_select(0, image_positions),
            key_states=key[:actual_tokens].index_select(0, image_positions),
            value_states=value[:actual_tokens].index_select(0, image_positions),
            layer_index=layer_index,
        )
        selected_positions = image_positions.index_select(0, selected_relative)
        keep[image_positions] = False
        keep[selected_positions] = True
        capture[selected_positions, 0] = candidate_ids[selected_positions].to(torch.int32)

    plan.query_keep_mask = keep
    plan.capture_values = capture
    capturer = RoutedExpertsCapturer.get_instance()
    if capturer is not None:
        # The compatibility transport reads capture channel zero.  The value
        # itself records which boundary layer produced the selection.
        capturer.capture(0, capture)


class LayerwisePrunedFlexAttentionImpl(FlexAttentionImpl):
    """Native paged FlexAttention with a persistent physical-slot keep mask."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._physical_slot_keep: torch.Tensor | None = None

    def _slot_sidecar(self, metadata: FlexAttentionMetadata) -> torch.Tensor:
        sidecar = self._physical_slot_keep
        if (
            sidecar is None
            or sidecar.device != metadata.slot_mapping.device
            or sidecar.numel() != metadata.total_cache_tokens
        ):
            sidecar = torch.ones(
                metadata.total_cache_tokens,
                dtype=torch.bool,
                device=metadata.slot_mapping.device,
            )
            self._physical_slot_keep = sidecar
        return sidecar

    @staticmethod
    def _scheduled_slots(metadata: FlexAttentionMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        slots = metadata.slot_mapping[: metadata.num_actual_tokens].long()
        valid = (slots >= 0) & (slots < metadata.total_cache_tokens)
        return slots, valid

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = _active_plan(layer)
        if active is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        plan, layer_index = active
        if plan.selector_input == "decoder_key" and layer_index == plan.prune_after_layer:
            _select_from_decoder_states(
                plan,
                query,
                key,
                value,
                attn_metadata,
                layer_index=layer_index,
            )

        if layer_index <= plan.prune_after_layer:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        sidecar = self._slot_sidecar(attn_metadata)
        slots, valid_slots = self._scheduled_slots(attn_metadata)
        sidecar[slots[valid_slots]] = True

        keep = plan.query_keep_mask
        if keep is not None:
            keep = keep[: attn_metadata.num_actual_tokens].to(sidecar.device)
            if keep.numel() != attn_metadata.num_actual_tokens:
                raise ValueError("Flex pruning mask does not match scheduled token count")
            sidecar[slots[valid_slots]] = keep[valid_slots]

        base_score_mod = attn_metadata.get_transformed_score_mod()

        def physical_slot_score_mod(score, batch, head, query_index, physical_kv_index):
            if base_score_mod is not None:
                score = base_score_mod(
                    score,
                    batch,
                    head,
                    query_index,
                    physical_kv_index,
                )
            return torch.where(sidecar[physical_kv_index], score, -float("inf"))

        attn_metadata.transformed_score_mod = physical_slot_score_mod
        result = super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if keep is not None:
            result[: attn_metadata.num_actual_tokens].masked_fill_(~keep[:, None, None], 0)
        return result


@register_backend(AttentionBackendEnum.FLEX_ATTENTION)
class LayerwisePrunedFlexAttentionBackend(FlexAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[LayerwisePrunedFlexAttentionImpl]:
        return LayerwisePrunedFlexAttentionImpl


class _LayerwiseFlexPruningMixin:
    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._pruning_config = pruning_config_from_hf(self.config)
        if not self._pruning_config.uses_layerwise_backend:
            raise ValueError("layerwise Flex model requires prune_after_layer >= 0")
        if self._pruning_config.layerwise_backend != "flex":
            raise ValueError("layerwise Flex model requires layerwise_backend='flex'")
        self._selection_engine = VisionTokenSelectionEngine(
            self._pruning_config,
            seed=int(vllm_config.model_config.seed),
        )
        self._pending_selection_metadata: list[torch.Tensor] = []
        self._pending_query_keep_mask: torch.Tensor | None = None
        self._pending_candidate_ids: torch.Tensor | None = None
        self._pending_capture_values: torch.Tensor | None = None

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ):
        self._pending_selection_metadata.extend(
            decode_embedding_selection_metadata(embeddings)
            for embeddings in multimodal_embeddings
            if len(embeddings)
        )
        embeddings_without_metadata = tuple(embeddings[:, :-2] for embeddings in multimodal_embeddings)
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
            torch.cat(self._pending_selection_metadata)
            if self._pending_selection_metadata
            else input_ids.new_empty(0)
        )
        self._pending_selection_metadata.clear()
        if encoded.numel():
            if is_multimodal is None or int(is_multimodal.sum()) != encoded.numel():
                raise ValueError("Flex selection metadata does not match image placeholders")
            candidate_ids = torch.zeros(input_ids.numel(), dtype=torch.int64, device=input_ids.device)
            candidate_ids[is_multimodal] = encoded.to(input_ids.device)
            self._pending_candidate_ids = candidate_ids

            if self._pruning_config.selector_input == "vision_embedding":
                keep = torch.ones(input_ids.numel(), dtype=torch.bool, device=input_ids.device)
                keep[is_multimodal] = encoded.to(input_ids.device) > 0
                self._pending_query_keep_mask = keep
                capture = torch.zeros((input_ids.numel(), 1), dtype=torch.int32, device=input_ids.device)
                capture[is_multimodal, 0] = encoded.to(device=input_ids.device, dtype=torch.int32)
                self._pending_capture_values = capture
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        context = vllm.forward_context.get_forward_context()
        metadata_by_layer = context.attn_metadata
        if isinstance(metadata_by_layer, dict) and metadata_by_layer:
            metadata = next(iter(metadata_by_layer.values()))
            if not isinstance(metadata, FlexAttentionMetadata):
                raise TypeError("layerwise Flex pruning requires FlexAttention metadata")
            actual_tokens = metadata.num_actual_tokens
            candidate_ids = self._pending_candidate_ids
            self._pending_candidate_ids = None
            if candidate_ids is None:
                candidate_ids = torch.zeros(actual_tokens, dtype=torch.int64, device=metadata.seq_lens.device)
            else:
                candidate_ids = candidate_ids[:actual_tokens].to(metadata.seq_lens.device)
            keep = self._pending_query_keep_mask
            self._pending_query_keep_mask = None
            if keep is not None:
                keep = keep[:actual_tokens].to(metadata.seq_lens.device)
            context.additional_kwargs[_FORWARD_CONTEXT_KEY] = LayerwiseFlexPruningPlan(
                prune_after_layer=self._pruning_config.prune_after_layer,
                selector_input=self._pruning_config.selector_input,
                candidate_ids=candidate_ids,
                query_keep_mask=keep,
                selection_engine=self._selection_engine,
            )

        capture = self._pending_capture_values
        self._pending_capture_values = None
        if capture is not None:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.capture(0, capture)
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def _annotate_image_selection(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        merge_size = self.visual.spatial_merge_size
        output = []
        for embeddings, grid_thw in zip(image_embeds_split, image_grid_thw.tolist(), strict=True):
            if self._pruning_config.selector_input == "decoder_key":
                indices = torch.arange(len(embeddings), device=embeddings.device)
            else:
                indices = self._selection_engine.select(embeddings, grid_thw=grid_thw)
            positions = compute_mrope_for_media(grid_thw, merge_size).to(embeddings.device)
            metadata = torch.zeros((len(embeddings), 2), dtype=embeddings.dtype, device=embeddings.device)
            metadata[indices] = encode_embedding_selection_metadata(indices, dtype=embeddings.dtype)
            output.append(torch.cat([embeddings, positions, metadata], dim=1))
        return tuple(output)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration(
    _LayerwiseFlexPruningMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._annotate_image_selection(
            image_embeds_split,
            image_input["image_grid_thw"],
        )
