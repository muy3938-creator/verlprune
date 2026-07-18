"""Compatibility facade for the milestone request-based strategy API."""

from __future__ import annotations

from typing import Any

import torch

from .strategy import (
    VisionTokenSelectionRequest,
    available_vision_token_strategies,
    register_vision_token_strategy,
    run_vision_token_strategy,
)


def select_visual_tokens(
    selector_name: str,
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    features: torch.Tensor | None = None,
    grid_thw: torch.Tensor | list[int] | None = None,
    selector_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run a strategy through the legacy flat-call integration surface."""

    return run_vision_token_strategy(
        selector_name,
        VisionTokenSelectionRequest(
            token_count=token_count,
            keep_count=keep_count,
            device=device,
            generator=generator,
            features=features,
            grid_thw=grid_thw,
            options=selector_kwargs or {},
        ),
    )


def select_random_visual_tokens(
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    **_: Any,
) -> torch.Tensor:
    return select_visual_tokens(
        "random",
        token_count,
        keep_count,
        device=device,
        generator=generator,
    )


def select_uniform_visual_tokens(
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    **_: Any,
) -> torch.Tensor:
    return select_visual_tokens(
        "uniform",
        token_count,
        keep_count,
        device=device,
        generator=generator,
    )


def register_vision_token_selector(name: str, selector, *, replace: bool = False) -> None:
    register_vision_token_strategy(name, selector, replace=replace)


def available_vision_token_selectors() -> tuple[str, ...]:
    return available_vision_token_strategies()
