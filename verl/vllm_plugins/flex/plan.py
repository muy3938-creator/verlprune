"""Forward-plan objects shared by Flex observe/apply paths.

Plan semantics come from PruningSpec stage kinds, not free-form strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import vllm.forward_context

from verl.models.vision_token_pruning.stages import InputKind, StageKind

if TYPE_CHECKING:
    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.stages import PruningSpec
    from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine

FORWARD_CONTEXT_KEY = "verl_layerwise_flex_vision_pruning"
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

_INPUT_KIND_TO_LEGACY = {
    InputKind.VISION_EMBEDDING: "vision_embedding",
    InputKind.BOUNDARY_QKV: "decoder_key",
    InputKind.DECODE_QK: "decode_query",
}

_LEGACY_TO_INPUT_KIND = {value: key for key, value in _INPUT_KIND_TO_LEGACY.items()}


@dataclass
class LayerwiseFlexPruningPlan:
    """Algorithm state for one scheduled vLLM forward call."""

    prune_after_layer: int
    stage_kind: StageKind
    input_kind: InputKind
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
        if not isinstance(self.stage_kind, StageKind):
            object.__setattr__(self, "stage_kind", StageKind(self.stage_kind))
        if not isinstance(self.input_kind, InputKind):
            object.__setattr__(self, "input_kind", InputKind(self.input_kind))
        if self.prune_after_layer < 0:
            raise ValueError("layerwise Flex pruning requires prune_after_layer >= 0")
        if self.stage_kind is StageKind.PHYSICAL_PRE_DECODER:
            raise ValueError("Flex plan cannot own physical_pre_decoder; use physical plugin")
        if self.stage_kind not in {StageKind.BOUNDARY_ONCE, StageKind.DECODE_QUERY}:
            raise ValueError(f"unsupported flex stage_kind={self.stage_kind!r}")
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
    def selector_input(self) -> str:
        """Legacy string form of input_kind (tests / debug only)."""

        return _INPUT_KIND_TO_LEGACY[self.input_kind]

    @property
    def uses_dynamic_decode_selection(self) -> bool:
        return self.stage_kind is StageKind.DECODE_QUERY

    @property
    def uses_boundary_qkv(self) -> bool:
        return self.input_kind is InputKind.BOUNDARY_QKV

    @property
    def uses_vision_embedding_input(self) -> bool:
        return self.input_kind is InputKind.VISION_EMBEDDING

    @property
    def uses_delayed_prefill_pruning(self) -> bool:
        return bool(
            self.uses_dynamic_decode_selection
            and self.prefill_prune_after_layer is not None
            and self.prefill_prune_after_layer >= 0
        )


def plan_kwargs_from_spec(spec: PruningSpec) -> dict:
    """Static plan fields derived from PruningSpec (fail if not a flex plan)."""

    if not spec.enabled or not spec.uses_layerwise_backend:
        raise ValueError("flex plan requires an enabled flex_mask PruningSpec")
    primary = spec.primary_stage
    if primary.kind is StageKind.PHYSICAL_PRE_DECODER:
        raise ValueError("flex plan primary stage cannot be physical_pre_decoder")
    if primary.observe_layer is None:
        raise ValueError(f"{primary.kind.value} stage missing observe_layer")
    prefill_layer: int | None = None
    if spec.uses_two_stage:
        prefill = spec.prefill_stage
        if prefill is None:
            raise ValueError("two-stage plan missing prefill stage")
        if prefill.kind is StageKind.PHYSICAL_PRE_DECODER:
            prefill_layer = -1
        elif prefill.kind is StageKind.BOUNDARY_ONCE:
            if prefill.observe_layer is None:
                raise ValueError("prefill boundary stage missing observe_layer")
            prefill_layer = prefill.observe_layer
        else:
            raise ValueError(f"unsupported prefill stage kind {prefill.kind.value}")
    return {
        "prune_after_layer": primary.observe_layer,
        "stage_kind": primary.kind,
        "input_kind": primary.input_kind,
        "pre_pruning_backend": spec.pre_pruning_backend,
        "prefill_prune_after_layer": prefill_layer,
    }


def plan_kwargs_from_config(config: VisionTokenPruningConfig) -> dict:
    return plan_kwargs_from_spec(config.spec)


def layerwise_plan(
    *,
    config: VisionTokenPruningConfig,
    candidate_ids: torch.Tensor,
    query_keep_mask: torch.Tensor | None,
    selection_engine: VisionTokenSelectionEngine,
    candidate_original_counts: torch.Tensor | None = None,
    budget_override: float | None = None,
    keep_ratio_override: float | None = None,
) -> LayerwiseFlexPruningPlan:
    """Build a forward plan from the frozen config.spec."""

    return LayerwiseFlexPruningPlan(
        **plan_kwargs_from_config(config),
        candidate_ids=candidate_ids,
        query_keep_mask=query_keep_mask,
        selection_engine=selection_engine,
        candidate_original_counts=candidate_original_counts,
        budget_override=budget_override,
        keep_ratio_override=keep_ratio_override,
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


def input_kind_from_legacy(selector_input: str) -> InputKind:
    try:
        return _LEGACY_TO_INPUT_KIND[selector_input]
    except KeyError as exc:
        raise ValueError(f"unsupported selector_input={selector_input!r}") from exc


def stage_kind_from_legacy(selector_input: str) -> StageKind:
    input_kind = input_kind_from_legacy(selector_input)
    if input_kind is InputKind.DECODE_QK:
        return StageKind.DECODE_QUERY
    return StageKind.BOUNDARY_ONCE
