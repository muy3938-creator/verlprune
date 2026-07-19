"""Pure-PyTorch decode-time visual KV selection algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicVisualSelectionResult:
    """One decode query's VisionPulse-style routing decision."""

    kept_visual_indices: torch.Tensor
    importance_scores: torch.Tensor
    per_head_visual_mass: torch.Tensor
    visual_mass_max: torch.Tensor

    @property
    def keep_count(self) -> int:
        return int(self.kept_visual_indices.numel())


def _ratio_bound(token_count: int, ratio: float, *, rounding: str) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"retention ratio must be in [0, 1], got {ratio}")
    value = token_count * ratio
    if rounding == "ceil":
        count = math.ceil(value)
    elif rounding == "floor":
        count = math.floor(value)
    elif rounding == "round":
        count = int(round(value))
    else:
        raise ValueError(f"unknown rounding mode {rounding!r}")
    return max(1, min(token_count, count))


def select_dynamic_visual_kv(
    query_states: torch.Tensor,
    context_key_states: torch.Tensor,
    visual_context_mask: torch.Tensor,
    *,
    softmax_scale: float,
    temperature: float,
    budget_mode: str,
    fixed_keep_ratio: float,
    min_keep_ratio: float = 0.0,
    max_keep_ratio: float = 1.0,
) -> DynamicVisualSelectionResult:
    """Select visual KV for one query using the VisionPulse equations.

    The softmax denominator covers the complete visible context.  Restricting
    it to visual keys would make visual attention mass identically one and
    destroy the dynamic-budget signal.

    Args:
        query_states: ``[num_query_heads, head_dim]`` boundary-layer query.
        context_key_states: ``[context_length, num_kv_heads, head_dim]`` keys.
        visual_context_mask: Boolean mask over the full context.
        softmax_scale: The model attention scale, normally ``head_dim**-0.5``.
        temperature: VisionPulse temperature ``tau``.
        budget_mode: ``visual_mass`` for Eq. (7), or ``fixed`` for a fixed
            keep ratio with query-dependent token identities.
        fixed_keep_ratio: Used only by ``fixed`` budgeting.
        min_keep_ratio: Optional lower clamp for the dynamic budget.
        max_keep_ratio: Optional upper clamp for the dynamic budget.
    """

    if query_states.ndim != 2:
        raise ValueError("query_states must have shape [num_heads, head_dim]")
    if context_key_states.ndim != 3:
        raise ValueError(
            "context_key_states must have shape [context_length, num_kv_heads, head_dim]"
        )
    if visual_context_mask.ndim != 1 or visual_context_mask.dtype != torch.bool:
        raise ValueError("visual_context_mask must be a rank-1 bool tensor")
    if len(context_key_states) != len(visual_context_mask):
        raise ValueError("visual_context_mask must match context_key_states")
    if query_states.shape[-1] != context_key_states.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if budget_mode not in {"visual_mass", "fixed"}:
        raise ValueError("budget_mode must be 'visual_mass' or 'fixed'")
    if not 0.0 <= min_keep_ratio <= max_keep_ratio <= 1.0:
        raise ValueError("dynamic keep-ratio clamps must satisfy 0 <= min <= max <= 1")

    visual_count = int(visual_context_mask.sum().item())
    if visual_count <= 0:
        raise ValueError("dynamic visual KV selection requires at least one visual key")

    num_query_heads = query_states.shape[0]
    num_kv_heads = context_key_states.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if num_query_heads != num_kv_heads:
        context_key_states = context_key_states.repeat_interleave(
            num_query_heads // num_kv_heads,
            dim=1,
        )

    logits = torch.einsum(
        "hd,lhd->hl",
        query_states.float(),
        context_key_states.float(),
    )
    probabilities = torch.softmax(logits * (float(softmax_scale) / float(temperature)), dim=-1)
    visual_probabilities = probabilities[:, visual_context_mask]
    per_head_visual_mass = visual_probabilities.sum(dim=-1)
    visual_mass_max = per_head_visual_mass.max()
    importance_scores = visual_probabilities.mean(dim=0)

    if budget_mode == "visual_mass":
        keep_count = math.ceil(float(visual_mass_max.item()) * visual_count)
    else:
        keep_count = _ratio_bound(visual_count, fixed_keep_ratio, rounding="round")

    minimum = _ratio_bound(visual_count, min_keep_ratio, rounding="ceil")
    maximum = _ratio_bound(visual_count, max_keep_ratio, rounding="floor")
    keep_count = max(minimum, min(maximum, keep_count))

    selected = importance_scores.topk(keep_count, sorted=False).indices.sort().values
    return DynamicVisualSelectionResult(
        kept_visual_indices=selected,
        importance_scores=importance_scores,
        per_head_visual_mass=per_head_visual_mass,
        visual_mass_max=visual_mass_max,
    )
