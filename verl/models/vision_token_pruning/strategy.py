"""Public strategy API for visual-token selection experiments."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import VisionTokenPruningConfig
from .protocol import compute_keep_count


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
    layer_index: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


VisionTokenStrategy = Callable[..., torch.Tensor]
_STRATEGIES: dict[str, VisionTokenStrategy] = {}


def _random_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    if request.keep_count == 1:
        return torch.tensor([request.token_count - 1], device=request.device)
    indices = torch.randperm(
        request.token_count - 1,
        device=request.device,
        generator=request.generator,
    )[: request.keep_count - 1]
    return torch.cat((indices, indices.new_tensor([request.token_count - 1]))).sort().values


def _uniform_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == 1:
        return torch.tensor([request.token_count - 1], device=request.device)
    return torch.linspace(
        0,
        request.token_count - 1,
        steps=request.keep_count,
        device=request.device,
    ).round().long()


def _key_norm_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Keep the largest decoder-key norms plus the final MRoPE anchor."""

    if request.key_states is None:
        raise ValueError("key_norm requires decoder key states")
    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    if request.keep_count == 1:
        return torch.tensor([request.token_count - 1], device=request.device)
    scores = request.key_states[:-1].float().square().sum(dim=tuple(range(1, request.key_states.ndim)))
    selected = scores.topk(request.keep_count - 1, sorted=False).indices
    return torch.cat((selected, selected.new_tensor([request.token_count - 1]))).sort().values


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
    """Distinguish the milestone request API from the legacy two-argument API."""

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
    if int(indices[-1]) != request.token_count - 1:
        raise ValueError(f"strategy {strategy_name!r} must retain the final MRoPE anchor")
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


class VisionTokenSelectionEngine:
    """Deterministic rollout-side owner of strategy invocation and seeding."""

    def __init__(self, config: VisionTokenPruningConfig, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self._selection_counter = 0

    @property
    def selection_count(self) -> int:
        return self._selection_counter

    def select(
        self,
        features: torch.Tensor,
        *,
        grid_thw: torch.Tensor | list[int] | None,
    ) -> torch.Tensor:
        token_count = len(features)
        keep_count = compute_keep_count(token_count, self.config.keep_ratio)
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self.seed + self._selection_counter)
        self._selection_counter += 1
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
    ) -> torch.Tensor:
        token_count = len(key_states)
        keep_count = compute_keep_count(token_count, self.config.keep_ratio)
        generator = torch.Generator(device=key_states.device)
        generator.manual_seed(self.seed + self._selection_counter)
        self._selection_counter += 1
        request = VisionTokenSelectionRequest(
            token_count=token_count,
            keep_count=keep_count,
            device=key_states.device,
            generator=generator,
            features=key_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            layer_index=layer_index,
            options=self.config.selector_kwargs,
        )
        return run_vision_token_strategy(self.config.selector, request)


register_vision_token_strategy("random", _random_strategy)
register_vision_token_strategy("uniform", _uniform_strategy)
register_vision_token_strategy("key_norm", _key_norm_strategy)
