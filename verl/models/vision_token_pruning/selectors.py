from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch

VisionTokenSelector = Callable[..., torch.Tensor]
_SELECTORS: dict[str, VisionTokenSelector] = {}


def select_random_visual_tokens(
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    **_: Any,
) -> torch.Tensor:
    """Randomly retain tokens while preserving the final MRoPE anchor token."""

    if not 0 < keep_count <= token_count:
        raise ValueError(f"keep_count must be in [1, {token_count}], got {keep_count}")
    if keep_count == token_count:
        return torch.arange(token_count, device=device)
    if keep_count == 1:
        return torch.tensor([token_count - 1], device=device)

    random_indices = torch.randperm(token_count - 1, device=device, generator=generator)[: keep_count - 1]
    return torch.cat((random_indices, random_indices.new_tensor([token_count - 1]))).sort().values


def select_uniform_visual_tokens(
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    **_: Any,
) -> torch.Tensor:
    """Retain spatially uniform indices, including the final MRoPE anchor."""

    del generator
    if not 0 < keep_count <= token_count:
        raise ValueError(f"keep_count must be in [1, {token_count}], got {keep_count}")
    if keep_count == 1:
        return torch.tensor([token_count - 1], device=device)
    return torch.linspace(0, token_count - 1, steps=keep_count, device=device).round().long()


def register_vision_token_selector(
    name: str,
    selector: VisionTokenSelector,
    *,
    replace: bool = False,
) -> None:
    """Register one rollout-side selection algorithm.

    Selectors receive ``token_count``, ``keep_count``, ``device``,
    ``generator``, ``features``, and ``grid_thw``. Training never calls the
    selector; it replays the exact indices returned by rollout.
    """

    normalized = name.strip()
    if not normalized:
        raise ValueError("selector name must be non-empty")
    if normalized in _SELECTORS and not replace:
        raise ValueError(f"vision token selector {normalized!r} is already registered")
    _SELECTORS[normalized] = selector


def available_vision_token_selectors() -> tuple[str, ...]:
    return tuple(sorted(_SELECTORS))


def _resolve_selector(selector_name: str) -> VisionTokenSelector:
    selector = _SELECTORS.get(selector_name)
    if selector is not None:
        return selector
    if ":" in selector_name:
        module_name, attribute_name = selector_name.rsplit(":", 1)
        try:
            selector = getattr(importlib.import_module(module_name), attribute_name)
        except (AttributeError, ImportError) as exc:
            raise ValueError(f"cannot import vision token selector {selector_name!r}") from exc
        if not callable(selector):
            raise ValueError(f"vision token selector {selector_name!r} is not callable")
        return selector
    raise ValueError(
        f"unknown vision token selector {selector_name!r}; "
        f"available selectors: {available_vision_token_selectors()}"
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
) -> torch.Tensor:
    """Run and validate a registered selection algorithm."""

    selector = _resolve_selector(selector_name)
    indices = selector(
        token_count,
        keep_count,
        device=device,
        generator=generator,
        features=features,
        grid_thw=grid_thw,
    )
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"selector {selector_name!r} must return a rank-1 integer tensor")
    if len(indices) != keep_count:
        raise ValueError(f"selector {selector_name!r} returned {len(indices)} tokens; expected {keep_count}")
    if not torch.equal(indices, indices.sort().values) or len(indices.unique()) != keep_count:
        raise ValueError(f"selector {selector_name!r} must return sorted unique indices")
    if int(indices[0]) < 0 or int(indices[-1]) >= token_count:
        raise ValueError(f"selector {selector_name!r} returned an out-of-range index")
    if int(indices[-1]) != token_count - 1:
        raise ValueError(f"selector {selector_name!r} must retain the final MRoPE anchor")
    return indices


register_vision_token_selector("random", select_random_visual_tokens)
register_vision_token_selector("uniform", select_uniform_visual_tokens)
