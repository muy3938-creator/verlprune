"""Experimental layerwise visual-token pruning for vLLM 0.18.

This module keeps the model's hidden-state shape stable while compacting the
per-request Q/K/V sequences and logical KV-cache layout after a selected layer.
It supports a continuously changing vLLM batch by associating each request's
pruned length with the first physical block in that request's block table.
"""

from __future__ import annotations

import dataclasses
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
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from verl.models.vision_token_pruning.protocol import compute_keep_count
from verl.models.vision_token_pruning.selectors import select_visual_tokens
from verl.vllm_plugins.vision_token_pruning import (
    _decode_selection_metadata,
    _encode_selection_metadata,
    _get_keep_ratio,
    _get_selector_name,
)

_FORWARD_CONTEXT_KEY = "verl_layerwise_vision_pruning"
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class LayerwisePruningPlan:
    """Current-batch layerwise pruning state shared with the attention backend."""

    prune_after_layer: int
    query_keep_mask: torch.Tensor
    pruned_token_counts: torch.Tensor

    def __post_init__(self) -> None:
        if self.prune_after_layer < 0:
            raise ValueError("layerwise pruning requires prune_after_layer >= 0")
        if self.query_keep_mask.ndim != 1 or self.query_keep_mask.dtype != torch.bool:
            raise ValueError("query_keep_mask must be a rank-1 bool tensor")
        if self.pruned_token_counts.ndim != 1 or self.pruned_token_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("pruned_token_counts must be a rank-1 integer tensor")
        if bool((self.pruned_token_counts < 0).any()):
            raise ValueError("pruned_token_counts cannot contain negative values")


def layer_index_from_name(layer_name: str) -> int:
    match = _LAYER_INDEX_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"cannot determine decoder layer index from {layer_name!r}")
    return int(match.group(1))


def _active_plan(layer: torch.nn.Module) -> tuple[LayerwisePruningPlan, int] | None:
    context = vllm.forward_context.get_forward_context()
    plan = context.additional_kwargs.get(_FORWARD_CONTEXT_KEY)
    if plan is None:
        return None
    if not isinstance(plan, LayerwisePruningPlan):
        raise TypeError("invalid layerwise pruning plan in vLLM forward context")
    layer_index = layer_index_from_name(layer.layer_name)
    if layer_index <= plan.prune_after_layer:
        return None
    return plan, layer_index


def _layer_attention_metadata(layer: torch.nn.Module) -> FlashAttentionMetadata:
    metadata_by_layer = vllm.forward_context.get_forward_context().attn_metadata
    if not isinstance(metadata_by_layer, dict):
        raise ValueError("experimental layerwise pruning does not support dual-batch overlap")
    metadata = metadata_by_layer[layer.layer_name]
    if not isinstance(metadata, FlashAttentionMetadata):
        raise TypeError(f"layerwise pruning requires FlashAttentionMetadata, got {type(metadata).__name__}")
    if metadata.use_cascade or metadata.common_prefix_len:
        raise ValueError("experimental layerwise pruning does not support prefix/cascade attention")
    return metadata


def _logical_slot_mapping(
    metadata: FlashAttentionMetadata,
    sequence_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    block_indices = torch.div(logical_positions, block_size, rounding_mode="floor")
    block_offsets = logical_positions.remainder(block_size)
    physical_blocks = metadata.block_table[sequence_indices.long(), block_indices.long()]
    return (physical_blocks * block_size + block_offsets).to(dtype=torch.int64)


def _compact_metadata(
    metadata: FlashAttentionMetadata,
    *,
    compact_query_lengths: torch.Tensor,
    compact_sequence_lengths: torch.Tensor,
) -> FlashAttentionMetadata:
    device = metadata.seq_lens.device
    compact_query_lengths = compact_query_lengths.to(device=device, dtype=torch.int32)
    compact_sequence_lengths = compact_sequence_lengths.to(device=device, dtype=torch.int32)
    query_start_loc = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            # torch.cumsum promotes int32 inputs to int64 unless dtype is
            # explicit. FlashAttention requires cu_seqlens_q to stay int32.
            compact_query_lengths.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    return dataclasses.replace(
        metadata,
        num_actual_tokens=int(query_start_loc[-1].item()),
        max_query_len=int(compact_query_lengths.max().item()),
        query_start_loc=query_start_loc,
        max_seq_len=int(compact_sequence_lengths.max().item()),
        seq_lens=compact_sequence_lengths,
        use_cascade=False,
        common_prefix_len=0,
        scheduler_metadata=None,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        prefix_scheduler_metadata=None,
    )


def _query_lengths(metadata: FlashAttentionMetadata) -> torch.Tensor:
    return metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]


def _compact_batch_layout(
    metadata: FlashAttentionMetadata,
    plan: LayerwisePruningPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return retained query indices, lengths, sequence ids, and KV positions."""

    query_lengths = _query_lengths(metadata)
    if plan.query_keep_mask.numel() != metadata.num_actual_tokens:
        raise ValueError(
            "query keep mask length does not match vLLM num_actual_tokens: "
            f"{plan.query_keep_mask.numel()} != {metadata.num_actual_tokens}"
        )
    if plan.pruned_token_counts.numel() != query_lengths.numel():
        raise ValueError("pruned-token counts do not match the number of active requests")

    keep_mask = plan.query_keep_mask.to(metadata.seq_lens.device)
    retained_indices = keep_mask.nonzero(as_tuple=False).flatten()
    sequence_indices = torch.repeat_interleave(
        torch.arange(query_lengths.numel(), device=query_lengths.device),
        query_lengths.to(dtype=torch.long),
    )
    retained_sequence_indices = sequence_indices[retained_indices]
    compact_query_lengths = torch.zeros_like(query_lengths)
    compact_query_lengths.scatter_add_(
        0,
        retained_sequence_indices,
        torch.ones_like(retained_sequence_indices, dtype=compact_query_lengths.dtype),
    )

    logical_positions: list[torch.Tensor] = []
    for sequence_index in range(query_lengths.numel()):
        query_start = int(metadata.query_start_loc[sequence_index].item())
        query_end = int(metadata.query_start_loc[sequence_index + 1].item())
        original_query_length = query_end - query_start
        original_sequence_length = int(metadata.seq_lens[sequence_index].item())
        context_length = original_sequence_length - original_query_length
        pruned_before_query = (
            int(plan.pruned_token_counts[sequence_index].item()) if context_length > 0 else 0
        )
        local_keep = keep_mask[query_start:query_end]
        local_positions = local_keep.to(torch.int64).cumsum(dim=0) - 1
        logical_positions.append(
            local_positions[local_keep] + context_length - pruned_before_query
        )

    return (
        retained_indices,
        compact_query_lengths,
        retained_sequence_indices,
        torch.cat(logical_positions) if logical_positions else metadata.seq_lens.new_empty(0),
    )


class LayerwisePrunedFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention with a compact logical KV sequence after a chosen layer."""

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        active = _active_plan(layer)
        if active is None:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        plan, _ = active
        metadata = _layer_attention_metadata(layer)
        block_size = kv_cache.shape[2]
        retained_indices, _, sequence_indices, logical_positions = _compact_batch_layout(metadata, plan)
        retained_indices = retained_indices.to(key.device)
        sequence_indices = sequence_indices.to(key.device)
        logical_positions = logical_positions.to(key.device)
        key_to_cache = key[: metadata.num_actual_tokens].index_select(0, retained_indices)
        value_to_cache = value[: metadata.num_actual_tokens].index_select(0, retained_indices)

        compact_slot_mapping = _logical_slot_mapping(
            metadata,
            sequence_indices,
            logical_positions,
            block_size=block_size,
        )
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key_to_cache,
            value_to_cache,
            key_cache,
            value_cache,
            compact_slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
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
        if output is None:
            raise ValueError("layerwise pruning requires the vLLM output buffer")

        plan, layer_index = active
        actual_tokens = attn_metadata.num_actual_tokens
        retained_indices, compact_query_lengths, _, _ = _compact_batch_layout(attn_metadata, plan)
        retained_indices = retained_indices.to(query.device)
        compact_query = query[:actual_tokens].index_select(0, retained_indices)
        compact_key = key[:actual_tokens].index_select(0, retained_indices)
        compact_value = value[:actual_tokens].index_select(0, retained_indices)
        compact_output = torch.empty(
            (retained_indices.numel(), output.shape[1], output.shape[2]),
            dtype=output.dtype,
            device=output.device,
        )
        compact_sequence_lengths = (
            attn_metadata.seq_lens
            - plan.pruned_token_counts.to(attn_metadata.seq_lens.device, attn_metadata.seq_lens.dtype)
        )
        if bool((compact_sequence_lengths <= 0).any()):
            raise ValueError("layerwise pruning produced an empty request sequence")
        compact_metadata = _compact_metadata(
            attn_metadata,
            compact_query_lengths=compact_query_lengths,
            compact_sequence_lengths=compact_sequence_lengths,
        )
        super().forward(
            layer,
            compact_query,
            compact_key,
            compact_value,
            kv_cache,
            compact_metadata,
            compact_output,
            output_scale,
            output_block_scale,
        )
        output[:actual_tokens].zero_()
        output[:actual_tokens].index_copy_(0, retained_indices, compact_output)

        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is not None:
            trace = torch.full(
                (actual_tokens, 1),
                -(layer_index + 1),
                dtype=torch.int32,
                device=query.device,
            )
            capturer.capture(layer_index, trace)
        return output


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class LayerwisePrunedFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[LayerwisePrunedFlashAttentionImpl]:
        return LayerwisePrunedFlashAttentionImpl


class _LayerwisePruningMixin:
    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._keep_ratio = _get_keep_ratio(self.config)
        self._selector = _get_selector_name(self.config)
        pruning_config = self.config.vision_token_pruning
        self._prune_after_layer = int(pruning_config.get("prune_after_layer", -1))
        if self._prune_after_layer < 0:
            raise ValueError("layerwise vLLM model requires prune_after_layer >= 0")
        self._selection_seed = int(vllm_config.model_config.seed)
        self._selection_counter = 0
        self._pending_selection_metadata: list[torch.Tensor] = []
        self._pending_query_keep_mask: torch.Tensor | None = None
        self._pending_capture_values: torch.Tensor | None = None
        self._pruned_count_by_first_block: dict[tuple[int, int], int] = {}

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ):
        self._pending_selection_metadata.extend(
            _decode_selection_metadata(embeddings)
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
                raise ValueError("layerwise selection metadata does not match image placeholders")
            query_keep_mask = torch.ones(input_ids.numel(), dtype=torch.bool, device=input_ids.device)
            query_keep_mask[is_multimodal] = encoded.to(input_ids.device) > 0
            self._pending_query_keep_mask = query_keep_mask

            capture_values = torch.zeros((input_ids.numel(), 1), dtype=torch.int32, device=input_ids.device)
            capture_values[is_multimodal, 0] = encoded.to(device=input_ids.device, dtype=torch.int32)
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
        context = vllm.forward_context.get_forward_context()
        metadata_by_layer = context.attn_metadata
        if isinstance(metadata_by_layer, dict) and metadata_by_layer:
            metadata = next(iter(metadata_by_layer.values()))
            if not isinstance(metadata, FlashAttentionMetadata):
                raise TypeError("layerwise pruning requires FlashAttention metadata")
            query_lengths = _query_lengths(metadata)
            query_keep_mask = self._pending_query_keep_mask
            self._pending_query_keep_mask = None
            if query_keep_mask is None:
                query_keep_mask = torch.ones(
                    metadata.num_actual_tokens,
                    dtype=torch.bool,
                    device=metadata.seq_lens.device,
                )
            else:
                query_keep_mask = query_keep_mask[: metadata.num_actual_tokens].to(metadata.seq_lens.device)
            if query_keep_mask.numel() != metadata.num_actual_tokens:
                raise ValueError("model-side query mask does not match the scheduled vLLM token count")

            pruned_counts: list[int] = []
            virtual_engine = int(context.virtual_engine)
            for sequence_index, query_length_tensor in enumerate(query_lengths):
                query_start = int(metadata.query_start_loc[sequence_index].item())
                query_length = int(query_length_tensor.item())
                sequence_length = int(metadata.seq_lens[sequence_index].item())
                context_length = sequence_length - query_length
                first_block = int(metadata.block_table[sequence_index, 0].item())
                request_key = (virtual_engine, first_block)
                if context_length == 0:
                    local_keep = query_keep_mask[query_start : query_start + query_length]
                    if not bool(local_keep[-1]):
                        raise ValueError("layerwise pruning must retain each prompt's final token")
                    pruned_count = query_length - int(local_keep.sum().item())
                    self._pruned_count_by_first_block[request_key] = pruned_count
                else:
                    if not bool(
                        query_keep_mask[query_start : query_start + query_length].all()
                    ):
                        raise ValueError("layerwise pruning may only drop tokens during initial prefill")
                    pruned_count = self._pruned_count_by_first_block.get(request_key, 0)
                pruned_counts.append(pruned_count)

            context.additional_kwargs[_FORWARD_CONTEXT_KEY] = LayerwisePruningPlan(
                prune_after_layer=self._prune_after_layer,
                query_keep_mask=query_keep_mask,
                pruned_token_counts=torch.tensor(
                    pruned_counts,
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                ),
            )

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

    def _annotate_image_selection(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        merge_size = self.visual.spatial_merge_size
        output = []
        for embeddings, grid_thw in zip(image_embeds_split, image_grid_thw.tolist(), strict=True):
            keep_count = compute_keep_count(len(embeddings), self._keep_ratio)
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed(self._selection_seed + self._selection_counter)
            self._selection_counter += 1
            kept_indices = select_visual_tokens(
                self._selector,
                len(embeddings),
                keep_count,
                device=embeddings.device,
                generator=generator,
                features=embeddings,
                grid_thw=grid_thw,
            )
            positions = compute_mrope_for_media(grid_thw, merge_size).to(embeddings.device)
            metadata = torch.zeros((len(embeddings), 2), dtype=embeddings.dtype, device=embeddings.device)
            metadata[kept_indices] = _encode_selection_metadata(kept_indices, dtype=embeddings.dtype)
            output.append(torch.cat([embeddings, positions, metadata], dim=1))
        return tuple(output)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class VerlLayerwisePrunedQwen2_5VLForConditionalGeneration(
    _LayerwisePruningMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._annotate_image_selection(
            image_embeds_split,
            image_input["image_grid_thw"],
        )


# Compatibility aliases for the random-only prototype architecture.
_LayerwiseRandomPruningMixin = _LayerwisePruningMixin
VerlLayerwiseRandomPrunedQwen2_5VLForConditionalGeneration = (
    VerlLayerwisePrunedQwen2_5VLForConditionalGeneration
)
