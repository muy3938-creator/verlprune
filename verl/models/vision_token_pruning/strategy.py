"""Public strategy registry and rollout-side selection engine.

Built-in algorithms live under ``policies/``. This module only resolves names,
validates outputs, and owns deterministic seeding for rollout workers.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch

from .config import VisionTokenPruningConfig
from .protocol import compute_keep_count
from .request import VisionTokenSelectionRequest

VisionTokenStrategy = Callable[..., torch.Tensor]
_STRATEGIES: dict[str, VisionTokenStrategy] = {}

# Re-export for existing imports.
__all__ = [
    "VisionTokenSelectionEngine",
    "VisionTokenSelectionRequest",
    "VisionTokenStrategy",
    "available_vision_token_strategies",
    "register_vision_token_strategy",
    "run_vision_token_strategy",
    "validate_selected_indices",
]


def register_vision_token_strategy(
    name: str,
    strategy: VisionTokenStrategy,
    *,
    replace: bool = False,
) -> None:
    """Register a request-based strategy callable for rollout workers."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("strategy name must be non-empty")
    if not callable(strategy):
        raise TypeError("strategy must be callable")
    if normalized in _STRATEGIES and not replace:
        raise ValueError(f"vision token strategy {normalized!r} is already registered")
    _STRATEGIES[normalized] = strategy


def available_vision_token_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def _resolve_strategy(name: str) -> VisionTokenStrategy:
    strategy = _STRATEGIES.get(name)
    if strategy is not None:
        return strategy
    if ":" not in name:
        raise ValueError(
            f"unknown vision token strategy {name!r}; "
            f"available strategies: {available_vision_token_strategies()}"
        )
    module_name, attribute_name = name.rsplit(":", 1)
    try:
        strategy = getattr(importlib.import_module(module_name), attribute_name)
    except (AttributeError, ImportError) as exc:
        raise ValueError(f"cannot import vision token strategy {name!r}") from exc
    if not callable(strategy):
        raise ValueError(f"vision token strategy {name!r} is not callable")
    return strategy


def _uses_request_api(strategy: VisionTokenStrategy) -> bool:
    """Distinguish the request API from the legacy two-argument API."""

    try:
        parameters = list(inspect.signature(strategy).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) == 1


def _invoke_strategy(strategy: VisionTokenStrategy, request: VisionTokenSelectionRequest) -> torch.Tensor:
    if _uses_request_api(strategy):
        return strategy(request)
    return strategy(
        request.token_count,
        request.keep_count,
        device=request.device,
        generator=request.generator,
        features=request.features,
        grid_thw=request.grid_thw,
        **request.options,
    )


def validate_selected_indices(
    indices: torch.Tensor,
    request: VisionTokenSelectionRequest,
    *,
    strategy_name: str,
) -> torch.Tensor:
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"strategy {strategy_name!r} must return a rank-1 integer tensor")
    if indices.device != request.device:
        raise ValueError(
            f"strategy {strategy_name!r} returned indices on {indices.device}; "
            f"expected {request.device}"
        )
    if len(indices) != request.keep_count:
        raise ValueError(
            f"strategy {strategy_name!r} returned {len(indices)} tokens; expected {request.keep_count}"
        )
    if not torch.equal(indices, indices.sort().values) or len(indices.unique()) != request.keep_count:
        raise ValueError(f"strategy {strategy_name!r} must return sorted unique indices")
    if int(indices[0]) < 0 or int(indices[-1]) >= request.token_count:
        raise ValueError(f"strategy {strategy_name!r} returned an out-of-range index")
    return indices


def run_vision_token_strategy(
    strategy_name: str,
    request: VisionTokenSelectionRequest,
) -> torch.Tensor:
    if not 0 < request.keep_count <= request.token_count:
        raise ValueError(
            f"keep_count must be in [1, {request.token_count}], got {request.keep_count}"
        )
    strategy = _resolve_strategy(strategy_name)
    return validate_selected_indices(
        _invoke_strategy(strategy, request),
        request,
        strategy_name=strategy_name,
    )


def _stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a reproducible seed from stable parts (not a global counter alone)."""

    import hashlib

    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


class VisionTokenSelectionEngine:
    """Deterministic rollout-side owner of strategy invocation and seeding."""

    def __init__(self, config: VisionTokenPruningConfig, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self._selection_counter = 0

    @property
    def selection_count(self) -> int:
        return self._selection_counter

    def _next_generator(self, device: torch.device, *, stage: str, layer_index: int | None) -> torch.Generator:
        # Counter still advances so successive calls differ, but the seed also
        # mixes stage/layer so async reordering is less likely to collide.
        counter = self._selection_counter
        self._selection_counter += 1
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _stable_seed(self.seed, stage, layer_index if layer_index is not None else -1, counter)
        )
        return generator

    def select(
        self,
        features: torch.Tensor,
        *,
        grid_thw: torch.Tensor | list[int] | None,
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        token_count = len(features)
        effective_ratio = self.config.keep_ratio if keep_ratio is None else float(keep_ratio)
        keep_count = compute_keep_count(token_count, effective_ratio)
        generator = self._next_generator(features.device, stage="vision", layer_index=None)
        request = VisionTokenSelectionRequest(
            token_count=token_count,
            keep_count=keep_count,
            device=features.device,
            generator=generator,
            features=features,
            grid_thw=grid_thw,
            options=self.config.selector_kwargs,
        )
        return run_vision_token_strategy(self.config.selector, request)

    def select_decoder_states(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_index: int,
        context_query_states: torch.Tensor | None = None,
        context_key_states: torch.Tensor | None = None,
        context_value_states: torch.Tensor | None = None,
        visual_context_mask: torch.Tensor | None = None,
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        token_count = len(key_states)
        effective_ratio = self.config.keep_ratio if keep_ratio is None else float(keep_ratio)
        keep_count = compute_keep_count(token_count, effective_ratio)
        generator = self._next_generator(
            key_states.device, stage="boundary", layer_index=layer_index
        )
        request = VisionTokenSelectionRequest(
            token_count=token_count,
            keep_count=keep_count,
            device=key_states.device,
            generator=generator,
            features=key_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            context_query_states=context_query_states,
            context_key_states=context_key_states,
            context_value_states=context_value_states,
            visual_context_mask=visual_context_mask,
            layer_index=layer_index,
            options=self.config.selector_kwargs,
        )
        return run_vision_token_strategy(self.config.selector, request)


def _register_builtin_policies() -> None:
    from .policies import (
        dart_policy,
        divprune_policy,
        embedding_norm_policy,
        greedy_prune_policy,
        key_norm_policy,
        random_policy,
        uniform_policy,
    )

    register_vision_token_strategy("random", random_policy)
    register_vision_token_strategy("uniform", uniform_policy)
    register_vision_token_strategy("key_norm", key_norm_policy)
    register_vision_token_strategy("embedding_norm", embedding_norm_policy)
    register_vision_token_strategy("dart", dart_policy)
    register_vision_token_strategy("divprune", divprune_policy)
    register_vision_token_strategy("greedy_prune", greedy_prune_policy)


_register_builtin_policies()
