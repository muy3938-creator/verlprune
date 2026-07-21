"""Forward-plan objects shared by Flex observe/apply paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import vllm.forward_context

if TYPE_CHECKING:
    from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine

FORWARD_CONTEXT_KEY = "verl_layerwise_flex_vision_pruning"
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


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
    prefill_prune_after_layer: int | None = None
    capture_values: torch.Tensor | None = None
    dynamic_request_slot_keep: torch.Tensor | None = None
    dynamic_request_by_query: torch.Tensor | None = None
    dynamic_query_active: torch.Tensor | None = None
    budget_override: float | None = None
    keep_ratio_override: float | None = None

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
        if self.keep_ratio_override is not None and not 0.0 < self.keep_ratio_override < 1.0:
            raise ValueError("keep_ratio_override must be in (0, 1)")

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


def active_plan(layer: torch.nn.Module) -> tuple[LayerwiseFlexPruningPlan, int] | None:
    context = vllm.forward_context.get_forward_context()
    plan = context.additional_kwargs.get(FORWARD_CONTEXT_KEY)
    if plan is None:
        return None
    if not isinstance(plan, LayerwiseFlexPruningPlan):
        raise TypeError("invalid layerwise Flex pruning plan in vLLM forward context")
    return plan, layer_index_from_name(layer.layer_name)
