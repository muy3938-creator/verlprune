"""Training-step visual-token budget schedules.

The schedule is deliberately backend-neutral.  Rollout and actor code use the
same piecewise-linear resolver, while the actor still replays the exact
indices returned by rollout rather than selecting a second set of tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PRUNING_GLOBAL_STEP_KEY = "_verl_vision_token_pruning_global_step"


def _parse_milestones(schedule: Mapping[str, Any] | None) -> tuple[tuple[int, float], ...]:
    if not schedule:
        return ()
    raw = schedule.get("milestones", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("keep_ratio_schedule.milestones must be a non-empty list")

    milestones: list[tuple[int, float]] = []
    for item in raw:
        if isinstance(item, Mapping):
            if "step" not in item or "keep_ratio" not in item:
                raise ValueError("each keep-ratio milestone needs step and keep_ratio")
            step, ratio = item["step"], item["keep_ratio"]
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            step, ratio = item
        else:
            raise ValueError("keep-ratio milestones must be [step, ratio] pairs")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("keep-ratio milestone steps must be non-negative integers")
        step = int(step)
        ratio = float(ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("keep-ratio milestone values must be in (0, 1)")
        if milestones and step <= milestones[-1][0]:
            raise ValueError("keep-ratio milestone steps must be strictly increasing")
        milestones.append((step, ratio))
    return tuple(milestones)


def validate_keep_ratio_schedule(schedule: Mapping[str, Any] | None) -> None:
    """Validate a schedule without changing the caller's mapping."""

    if schedule in (None, {}, ()):
        return
    if not isinstance(schedule, Mapping):
        raise ValueError("keep_ratio_schedule must be a mapping")
    interpolation = str(schedule.get("interpolation", "linear"))
    if interpolation != "linear":
        raise ValueError("keep_ratio_schedule.interpolation must be 'linear'")
    _parse_milestones(schedule)


def resolve_keep_ratio(
    schedule: Mapping[str, Any] | None,
    global_step: int | None,
    *,
    fallback: float,
) -> float:
    """Resolve the retention ratio at one optimizer/rollout step.

    Before the first milestone and after the last milestone the endpoint value
    is held.  Between endpoints, linear interpolation makes the curriculum
    smooth; e.g. ``[[0, .50], [800, .10], [1000, .05]]`` gradually moves from
    50% to 10%, then finishes at 5%.
    """

    milestones = _parse_milestones(schedule)
    if not milestones or global_step is None:
        return float(fallback)
    step = max(0, int(global_step))
    if step <= milestones[0][0]:
        return milestones[0][1]
    for (left_step, left_ratio), (right_step, right_ratio) in zip(milestones, milestones[1:]):
        if step <= right_step:
            fraction = (step - left_step) / (right_step - left_step)
            return left_ratio + fraction * (right_ratio - left_ratio)
    return milestones[-1][1]
