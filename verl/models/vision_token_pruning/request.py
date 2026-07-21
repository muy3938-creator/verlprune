"""Typed request object passed to every selection policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class VisionTokenSelectionRequest:
    """All rollout-side inputs available to one selection strategy."""

    token_count: int
    keep_count: int
    device: torch.device
    generator: torch.Generator | None = None
    features: torch.Tensor | None = None
    grid_thw: torch.Tensor | list[int] | None = None
    query_states: torch.Tensor | None = None
    key_states: torch.Tensor | None = None
    value_states: torch.Tensor | None = None
    context_query_states: torch.Tensor | None = None
    context_key_states: torch.Tensor | None = None
    context_value_states: torch.Tensor | None = None
    visual_context_mask: torch.Tensor | None = None
    layer_index: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
