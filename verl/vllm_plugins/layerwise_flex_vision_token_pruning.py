"""Layerwise visual-token masking on vLLM's native FlexAttention backend.

The integration deliberately leaves vLLM's paged KV layout untouched.  A
boolean sidecar is indexed by the physical slots that vLLM already assigns;
FlexAttention consults it from a score modifier after the configured layer.
Selection can happen in the vision tower, once from decoder Q/K/V at the layer
boundary, or independently for every decode query using VisionPulse-style QK
attention. Static modes reuse a persistent physical-slot sidecar. Dynamic
decode mode keeps the complete cache and installs a query-dependent mask only
for the current scheduled step.
"""

from __future__ import annotations

import os
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
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import compute_mrope_for_media
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
    FlexAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.dynamic import select_dynamic_visual_kv
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine
from verl.models.vision_token_pruning.transport import (
    decode_embedding_selection_metadata,
    encode_embedding_selection_metadata,
)
from verl.vllm_plugins.vision_token_pruning_common import pruning_config_from_hf
from verl.vllm_plugins.vision_token_pruning import (
    VerlPrunedQwen2_5VLMultiModalProcessor,
    VerlPrunedQwen3VLMultiModalProcessor,
)

_FORWARD_CONTEXT_KEY = "verl_layerwise_flex_vision_pruning"
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _debug(message: str) -> None:
    if os.getenv("VERL_VISION_PRUNING_DEBUG") == "1":
        print(f"[verl-pruning-debug] {message}", flush=True)


@dataclass
class LayerwiseFlexPruningPlan:
    """Algorithm state for one scheduled vLLM forward call."""

    prune_after_layer: int
    selector_input: str
    candidate_ids: torch.Tensor
    query_keep_mask: torch.Tensor | None
    selection_engine: VisionTokenSelectionEngine
    candidate_original_counts: torch.Tensor | None = None
    pre_pruning_backend: str = "flex"
    # Defaults to the decode boundary for legacy single-stage plans. -1 is
    # the physically compacted two-stage mode; non-negative values enable a
    # first-stage static mask before the later dynamic decode boundary.
    prefill_prune_after_layer: int | None = None
    capture_values: torch.Tensor | None = None
    dynamic_request_slot_keep: torch.Tensor | None = None
    dynamic_request_by_query: torch.Tensor | None = None
    dynamic_query_active: torch.Tensor | None = None
    budget_override: float | None = None

    def __post_init__(self) -> None:
        if self.prune_after_layer < 0:
            raise ValueError("layerwise Flex pruning requires prune_after_layer >= 0")
        if self.pre_pruning_backend not in {"flex", "flash"}:
            raise ValueError("pre_pruning_backend must be 'flex' or 'flash'")
        if self.candidate_ids.ndim != 1 or self.candidate_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be a rank-1 integer tensor")
        if self.candidate_original_counts is None:
            self.candidate_original_counts = torch.zeros_like(self.candidate_ids)
        if self.candidate_original_counts.shape != self.candidate_ids.shape:
            raise ValueError("candidate_original_counts must match candidate_ids")
        if self.query_keep_mask is not None and (
            self.query_keep_mask.ndim != 1 or self.query_keep_mask.dtype != torch.bool
        ):
            raise ValueError("query_keep_mask must be a rank-1 bool tensor")
        if self.prefill_prune_after_layer is None:
            self.prefill_prune_after_layer = self.prune_after_layer
        if self.prefill_prune_after_layer < -1:
            raise ValueError("prefill_prune_after_layer must be >= -1")
        if self.prefill_prune_after_layer > self.prune_after_layer:
            raise ValueError("prefill boundary cannot follow the decode boundary")

    @property
    def uses_dynamic_decode_selection(self) -> bool:
        return self.selector_input == "decode_query"

    @property
    def uses_delayed_prefill_pruning(self) -> bool:
        return bool(
            self.uses_dynamic_decode_selection
            and self.prefill_prune_after_layer is not None
            and self.prefill_prune_after_layer >= 0
        )


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
            context_query_states=query[start:end],
            context_key_states=key[start:end],
            context_value_states=value[start:end],
            visual_context_mask=candidate_ids[start:end] > 0,
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


def _request_physical_slots(
    metadata: FlexAttentionMetadata,
    request_index: int,
) -> torch.Tensor:
    """Return one request's live paged-cache slots in logical token order."""

    sequence_length = int(metadata.seq_lens[request_index].item())
    logical = torch.arange(sequence_length, device=metadata.block_table.device)
    logical_blocks = logical // metadata.block_size
    offsets = logical % metadata.block_size
    physical_blocks = metadata.block_table[request_index].long().index_select(0, logical_blocks)
    slots = physical_blocks * metadata.block_size + offsets
    valid = (physical_blocks >= 0) & (slots >= 0) & (slots < metadata.total_cache_tokens)
    if not bool(valid.all()):
        raise ValueError("dynamic visual KV selection found an invalid live cache block")
    return slots


class LayerwisePrunedFlexAttentionImpl(FlexAttentionImpl):
    """Native paged FlexAttention with a persistent physical-slot keep mask."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._physical_slot_keep: torch.Tensor | None = None
        self._anchor_key_sidecar: torch.Tensor | None = None
        self._anchor_visual_sidecar: torch.Tensor | None = None
        self._flash_impl: FlashAttentionImpl | None = None
        self._pre_boundary_flash_forwards = 0
        self._pre_boundary_flex_fallbacks = 0

    def _get_flash_impl(self) -> FlashAttentionImpl:
        if self._flash_impl is None:
            self._flash_impl = FlashAttentionImpl(
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                num_kv_heads=self.num_kv_heads,
                alibi_slopes=None,
                sliding_window=self.sliding_window,
                kv_cache_dtype=self.kv_cache_dtype,
                logits_soft_cap=self.logits_soft_cap,
                attn_type=self.attn_type,
                kv_sharing_target_layer_name=None,
                sinks=None,
            )
        return self._flash_impl

    @staticmethod
    def _flash_metadata_from_flex(
        metadata: FlexAttentionMetadata,
    ) -> FlashAttentionMetadata | None:
        """Adapt only cases whose Flex semantics FlashAttention can preserve."""

        if (
            not metadata.causal
            or metadata.use_cascade
            or bool(metadata.mm_prefix_range)
            or metadata.score_mod is not None
            or metadata.transformed_score_mod is not None
        ):
            return None
        return FlashAttentionMetadata(
            num_actual_tokens=metadata.num_actual_tokens,
            max_query_len=metadata.max_query_len,
            query_start_loc=metadata.query_start_loc,
            max_seq_len=metadata.max_seq_len,
            seq_lens=metadata.seq_lens,
            block_table=metadata.block_table,
            slot_mapping=metadata.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            max_dcp_context_kv_len=None,
            dcp_context_kv_lens=None,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            max_num_splits=0,
            causal=True,
        )

    def _anchor_sidecars(
        self,
        metadata: FlexAttentionMetadata,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_key_shape = (metadata.total_cache_tokens, *key.shape[1:])
        key_sidecar = self._anchor_key_sidecar
        visual_sidecar = self._anchor_visual_sidecar
        if (
            key_sidecar is None
            or key_sidecar.device != key.device
            or tuple(key_sidecar.shape) != expected_key_shape
            or key_sidecar.dtype != key.dtype
        ):
            key_sidecar = torch.zeros(expected_key_shape, dtype=key.dtype, device=key.device)
            visual_sidecar = torch.zeros(
                metadata.total_cache_tokens,
                dtype=torch.bool,
                device=key.device,
            )
            self._anchor_key_sidecar = key_sidecar
            self._anchor_visual_sidecar = visual_sidecar
        assert visual_sidecar is not None
        return key_sidecar, visual_sidecar

    def _select_dynamic_decode_queries(
        self,
        plan: LayerwiseFlexPruningPlan,
        query: torch.Tensor,
        key: torch.Tensor,
        metadata: FlexAttentionMetadata,
    ) -> None:
        """Build a different visual-KV mask for every current decode query."""

        actual_tokens = metadata.num_actual_tokens
        _debug(f"dynamic-select actual_tokens={actual_tokens}")
        slots, valid_slots = self._scheduled_slots(metadata)
        key_sidecar, visual_sidecar = self._anchor_sidecars(metadata, key)
        key_sidecar[slots[valid_slots]] = key[:actual_tokens][valid_slots]
        candidate_ids = plan.candidate_ids[:actual_tokens].to(visual_sidecar.device)
        visual_sidecar[slots[valid_slots]] = candidate_ids[valid_slots] > 0

        request_count = len(metadata.query_start_loc) - 1
        request_slot_keep = torch.ones(
            (request_count, metadata.total_cache_tokens),
            dtype=torch.bool,
            device=key.device,
        )
        request_by_query = torch.repeat_interleave(
            torch.arange(request_count, device=key.device),
            _query_lengths(metadata).to(key.device),
        )
        query_active = torch.zeros(actual_tokens, dtype=torch.bool, device=key.device)

        options = plan.selection_engine.config.selector_kwargs
        temperature = float(options.get("temperature", 0.1))
        budget_mode = str(options.get("budget_mode", "visual_mass"))
        top_p = float(
            plan.budget_override
            if plan.budget_override is not None
            else options.get("top_p", 0.95)
        )
        min_keep_ratio = float(options.get("min_keep_ratio", 0.0))
        max_keep_ratio = float(options.get("max_keep_ratio", 1.0))
        capture_capacity = int(options.get("capture_capacity", 64))
        capture = torch.zeros(
            (actual_tokens, capture_capacity),
            dtype=torch.int32,
            device=key.device,
        )
        if plan.selection_engine.config.uses_two_stage_pruning:
            if capture_capacity < 3:
                raise ValueError("two-stage selection capture_capacity must be at least 3")
            prefill_rows = candidate_ids > 0
            original_counts = plan.candidate_original_counts[:actual_tokens].to(key.device)
            if bool(prefill_rows.any()):
                capture[prefill_rows, 0] = candidate_ids[prefill_rows].to(torch.int32)
                capture[prefill_rows, 1] = original_counts[prefill_rows].to(torch.int32)
                capture[prefill_rows, -1] = 1

        for request_index in range(request_count):
            start = int(metadata.query_start_loc[request_index].item())
            end = int(metadata.query_start_loc[request_index + 1].item())
            if end <= start:
                continue
            # Normal decode has one scheduled query. During prefill only the
            # final prompt query predicts the first generated token.
            query_index = end - 1
            context_slots = _request_physical_slots(metadata, request_index)
            context_visual = visual_sidecar.index_select(0, context_slots)
            if not bool(context_visual.any()):
                continue
            query_heads = query[query_index].reshape(-1, self.head_size)
            context_key_heads = key_sidecar.index_select(0, context_slots).reshape(
                len(context_slots),
                -1,
                self.head_size,
            )
            result = select_dynamic_visual_kv(
                query_heads,
                context_key_heads,
                context_visual,
                softmax_scale=self.scale,
                temperature=temperature,
                budget_mode=budget_mode,
                fixed_keep_ratio=plan.selection_engine.config.keep_ratio,
                top_p=top_p,
                min_keep_ratio=min_keep_ratio,
                max_keep_ratio=max_keep_ratio,
            )
            usable_capacity = (
                capture_capacity - 1
                if plan.selection_engine.config.uses_two_stage_pruning
                else capture_capacity
            )
            if result.keep_count > usable_capacity:
                raise ValueError(
                    f"dynamic selection kept {result.keep_count} visual tokens, exceeding "
                    f"usable capture capacity={usable_capacity}"
                )
            if bool(capture[query_index, -1]):
                raise ValueError("two-stage prefill metadata collided with the dynamic query row")
            visual_slots = context_slots[context_visual]
            selected_slots = visual_slots.index_select(0, result.kept_visual_indices)
            request_slot_keep[request_index, visual_slots] = False
            request_slot_keep[request_index, selected_slots] = True
            query_active[query_index] = True
            capture[query_index, : result.keep_count] = (
                result.kept_visual_indices.to(torch.int32) + 1
            )

        plan.dynamic_request_slot_keep = request_slot_keep
        plan.dynamic_request_by_query = request_by_query
        plan.dynamic_query_active = query_active
        plan.capture_values = capture
        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is not None:
            capturer.capture(0, capture)
        _debug("dynamic-select complete")

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
        assert plan.prefill_prune_after_layer is not None
        if layer_index in {
            plan.prefill_prune_after_layer,
            plan.prefill_prune_after_layer + 1,
            plan.prune_after_layer,
            plan.prune_after_layer + 1,
        }:
            _debug(
                f"attention layer={layer_index} actual={attn_metadata.num_actual_tokens} "
                f"static_keep={plan.query_keep_mask is not None}"
            )
        if layer_index == plan.prune_after_layer:
            if plan.selector_input == "decoder_key":
                _select_from_decoder_states(
                    plan,
                    query,
                    key,
                    value,
                    attn_metadata,
                    layer_index=layer_index,
                )
            elif plan.uses_dynamic_decode_selection:
                self._select_dynamic_decode_queries(
                    plan,
                    query,
                    key,
                    attn_metadata,
                )

        flash_boundary = (
            plan.prefill_prune_after_layer
            if plan.uses_delayed_prefill_pruning
            else plan.prune_after_layer
        )
        if layer_index <= flash_boundary:
            if plan.pre_pruning_backend == "flash":
                flash_metadata = self._flash_metadata_from_flex(attn_metadata)
                if flash_metadata is not None:
                    result = self._get_flash_impl().forward(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        flash_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    )
                    self._pre_boundary_flash_forwards += 1
                    return result
                self._pre_boundary_flex_fallbacks += 1
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

        # A delayed first stage is already active at the decode anchor, but
        # the per-query selection only affects subsequent layers.
        if layer_index <= plan.prune_after_layer:
            base_score_mod = attn_metadata.get_transformed_score_mod()

            def stage_one_score_mod(score, batch, head, query_index, physical_kv_index):
                if base_score_mod is not None:
                    score = base_score_mod(
                        score,
                        batch,
                        head,
                        query_index,
                        physical_kv_index,
                    )
                return torch.where(sidecar[physical_kv_index], score, -float("inf"))

            attn_metadata.transformed_score_mod = stage_one_score_mod
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

        if plan.uses_dynamic_decode_selection:
            request_slot_keep = plan.dynamic_request_slot_keep
            request_by_query = plan.dynamic_request_by_query
            query_active = plan.dynamic_query_active
            if request_slot_keep is None or request_by_query is None or query_active is None:
                raise RuntimeError("anchor layer did not produce dynamic visual KV routing")
            base_score_mod = attn_metadata.get_transformed_score_mod()

            def dynamic_query_score_mod(score, batch, head, query_index, physical_kv_index):
                if base_score_mod is not None:
                    score = base_score_mod(
                        score,
                        batch,
                        head,
                        query_index,
                        physical_kv_index,
                    )
                request_index = request_by_query[query_index]
                dynamic_keep = request_slot_keep[request_index, physical_kv_index]
                keep_slot = sidecar[physical_kv_index] & (
                    ~query_active[query_index] | dynamic_keep
                )
                return torch.where(keep_slot, score, -float("inf"))

            attn_metadata.transformed_score_mod = dynamic_query_score_mod
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
    _append_zero_position_axis = False
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
        self._prefill_selection_engine = None
        if self._pruning_config.uses_two_stage_pruning:
            assert self._pruning_config.prefill_keep_ratio is not None
            self._prefill_selection_engine = VisionTokenSelectionEngine(
                VisionTokenPruningConfig(
                    enabled=True,
                    keep_ratio=self._pruning_config.prefill_keep_ratio,
                    selector=self._pruning_config.prefill_selector,
                    selector_kwargs=self._pruning_config.prefill_selector_kwargs,
                ),
                seed=int(vllm_config.model_config.seed),
            )
            expected_pruning_rate = (
                1.0 - self._pruning_config.prefill_keep_ratio
                if self._pruning_config.uses_physical_prefill_pruning
                else 0.0
            )
            if (
                self.video_pruning_rate is None
                or abs(float(self.video_pruning_rate) - expected_pruning_rate) > 1e-8
            ):
                raise ValueError(
                    "two-stage vLLM video_pruning_rate does not match the configured prefill mode"
                )
        self._pending_selection_metadata: list[torch.Tensor] = []
        self._pending_original_counts: list[torch.Tensor] = []
        self._pending_query_keep_mask: torch.Tensor | None = None
        self._pending_candidate_ids: torch.Tensor | None = None
        self._pending_candidate_original_counts: torch.Tensor | None = None
        self._pending_capture_values: torch.Tensor | None = None
        self._budget_schedule_index = 0

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ):
        _debug(f"recompute-mrope groups={len(multimodal_embeddings)}")
        for embeddings in multimodal_embeddings:
            if not len(embeddings):
                continue
            if self._pruning_config.uses_two_stage_pruning:
                self._pending_selection_metadata.append(
                    decode_embedding_selection_metadata(embeddings[:, -4:-2])
                )
                self._pending_original_counts.append(
                    decode_embedding_selection_metadata(embeddings[:, -2:])
                )
            else:
                self._pending_selection_metadata.append(
                    decode_embedding_selection_metadata(embeddings)
                )
                self._pending_original_counts.append(
                    embeddings.new_zeros(len(embeddings), dtype=torch.long)
                )
        metadata_columns = 4 if self._pruning_config.uses_two_stage_pruning else 2
        embeddings_without_metadata = tuple(
            embeddings[:, :-metadata_columns] for embeddings in multimodal_embeddings
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
        _debug(f"embed-input token_count={input_ids.numel()}")
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
        original_counts = (
            torch.cat(self._pending_original_counts)
            if self._pending_original_counts
            else input_ids.new_empty(0)
        )
        self._pending_selection_metadata.clear()
        self._pending_original_counts.clear()
        if encoded.numel():
            _debug(f"embed-input metadata_rows={encoded.numel()}")
            if is_multimodal is None or int(is_multimodal.sum()) != encoded.numel():
                raise ValueError("Flex selection metadata does not match image placeholders")
            candidate_ids = torch.zeros(input_ids.numel(), dtype=torch.int64, device=input_ids.device)
            candidate_ids[is_multimodal] = encoded.to(input_ids.device)
            self._pending_candidate_ids = candidate_ids
            candidate_original_counts = torch.zeros_like(candidate_ids)
            candidate_original_counts[is_multimodal] = original_counts.to(input_ids.device)
            self._pending_candidate_original_counts = candidate_original_counts

            if self._pruning_config.selector_input == "vision_embedding" or (
                self._pruning_config.uses_delayed_prefill_pruning
                and self._pruning_config.uses_two_stage_pruning
            ):
                keep = torch.ones(input_ids.numel(), dtype=torch.bool, device=input_ids.device)
                keep[is_multimodal] = encoded.to(input_ids.device) > 0
                self._pending_query_keep_mask = keep
            if self._pruning_config.selector_input == "vision_embedding":
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
        _debug("model-forward start")
        context = vllm.forward_context.get_forward_context()
        metadata_by_layer = context.attn_metadata
        if isinstance(metadata_by_layer, dict) and metadata_by_layer:
            metadata = next(iter(metadata_by_layer.values()))
            if not isinstance(metadata, FlexAttentionMetadata):
                raise TypeError("layerwise Flex pruning requires FlexAttention metadata")
            actual_tokens = metadata.num_actual_tokens
            candidate_ids = self._pending_candidate_ids
            self._pending_candidate_ids = None
            candidate_original_counts = self._pending_candidate_original_counts
            self._pending_candidate_original_counts = None
            if candidate_ids is None:
                candidate_ids = torch.zeros(actual_tokens, dtype=torch.int64, device=metadata.seq_lens.device)
            else:
                candidate_ids = candidate_ids[:actual_tokens].to(metadata.seq_lens.device)
            if candidate_original_counts is None:
                candidate_original_counts = torch.zeros_like(candidate_ids)
            else:
                candidate_original_counts = candidate_original_counts[:actual_tokens].to(
                    metadata.seq_lens.device
                )
            keep = self._pending_query_keep_mask
            self._pending_query_keep_mask = None
            if keep is not None:
                keep = keep[:actual_tokens].to(metadata.seq_lens.device)
            schedule = self._pruning_config.selector_kwargs.get("budget_schedule", ())
            budget_override = None
            if schedule:
                schedule = tuple(float(value) for value in schedule)
                budget_override = schedule[min(self._budget_schedule_index, len(schedule) - 1)]
                self._budget_schedule_index += 1
            context.additional_kwargs[_FORWARD_CONTEXT_KEY] = LayerwiseFlexPruningPlan(
                prune_after_layer=self._pruning_config.prune_after_layer,
                selector_input=self._pruning_config.selector_input,
                candidate_ids=candidate_ids,
                candidate_original_counts=candidate_original_counts,
                query_keep_mask=keep,
                selection_engine=self._selection_engine,
                pre_pruning_backend=self._pruning_config.pre_pruning_backend,
                prefill_prune_after_layer=(
                    self._pruning_config.prefill_prune_after_layer
                    if self._pruning_config.uses_two_stage_pruning
                    else None
                ),
                budget_override=budget_override,
            )
            _debug(
                f"plan actual={actual_tokens} candidates={int((candidate_ids > 0).sum())} "
                f"static_keep={keep is not None}"
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
            if self._pruning_config.uses_two_stage_pruning:
                assert self._prefill_selection_engine is not None
                indices = self._prefill_selection_engine.select(embeddings, grid_thw=grid_thw)
                _debug(f"annotate-image original={len(embeddings)} selected={len(indices)}")
            elif self._pruning_config.selector_input in {"decoder_key", "decode_query"}:
                indices = torch.arange(len(embeddings), device=embeddings.device)
            else:
                indices = self._selection_engine.select(embeddings, grid_thw=grid_thw)
            positions = compute_mrope_for_media(grid_thw, merge_size).to(embeddings.device)
            if self._append_zero_position_axis:
                positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=1)
            if self._pruning_config.uses_two_stage_pruning:
                index_metadata = encode_embedding_selection_metadata(indices, dtype=embeddings.dtype)
                count_metadata = encode_embedding_selection_metadata(
                    indices.new_full((len(indices),), len(embeddings) - 1),
                    dtype=embeddings.dtype,
                )
                if self._pruning_config.uses_physical_prefill_pruning:
                    metadata = torch.cat((index_metadata, count_metadata), dim=1)
                    output.append(
                        torch.cat(
                            [embeddings[indices], positions[indices], metadata],
                            dim=1,
                        )
                    )
                else:
                    metadata = embeddings.new_zeros((len(embeddings), 4))
                    metadata[indices] = torch.cat((index_metadata, count_metadata), dim=1)
                    output.append(torch.cat([embeddings, positions, metadata], dim=1))
            else:
                metadata = torch.zeros(
                    (len(embeddings), 2),
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
                metadata[indices] = encode_embedding_selection_metadata(indices, dtype=embeddings.dtype)
                output.append(torch.cat([embeddings, positions, metadata], dim=1))
        return tuple(output)


@MULTIMODAL_REGISTRY.register_processor(
    VerlPrunedQwen2_5VLMultiModalProcessor,
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


@MULTIMODAL_REGISTRY.register_processor(
    VerlPrunedQwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class VerlLayerwiseFlexPrunedQwen3VLForConditionalGeneration(
    _LayerwiseFlexPruningMixin,
    Qwen3VLForConditionalGeneration,
):
    _append_zero_position_axis = True

    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        return self._annotate_image_selection(
            image_embeds_split,
            image_input["image_grid_thw"],
        )
