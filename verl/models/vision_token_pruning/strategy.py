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
    context_query_states: torch.Tensor | None = None
    context_key_states: torch.Tensor | None = None
    context_value_states: torch.Tensor | None = None
    visual_context_mask: torch.Tensor | None = None
    layer_index: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


VisionTokenStrategy = Callable[..., torch.Tensor]
_STRATEGIES: dict[str, VisionTokenStrategy] = {}


def _random_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    indices = torch.randperm(
        request.token_count,
        device=request.device,
        generator=request.generator,
    )[: request.keep_count]
    return indices.sort().values


def _uniform_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    return torch.linspace(
        0,
        request.token_count - 1,
        steps=request.keep_count,
        device=request.device,
    ).round().long()


def _key_norm_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Keep the visual tokens with the largest decoder-key norms."""

    if request.key_states is None:
        raise ValueError("key_norm requires decoder key states")
    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    scores = request.key_states.float().square().sum(dim=tuple(range(1, request.key_states.ndim)))
    return scores.topk(request.keep_count, sorted=False).indices.sort().values


def _embedding_norm_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Keep visual encoder embeddings with the largest L2 norms."""

    features = _flat_float(request.features, name="embedding_norm")
    scores = features.square().sum(dim=-1)
    return scores.topk(request.keep_count, sorted=False).indices.sort().values


def _flat_float(features: torch.Tensor | None, *, name: str) -> torch.Tensor:
    if features is None:
        raise ValueError(f"{name} requires decoder states")
    if features.ndim < 2:
        raise ValueError(f"{name} decoder states must have a token dimension")
    return features.float().reshape(len(features), -1)


def _decoder_text_features(
    request: VisionTokenSelectionRequest,
    *,
    source: str,
    strategy_name: str,
) -> torch.Tensor:
    context = getattr(request, f"context_{source}_states")
    visual_mask = request.visual_context_mask
    features = _flat_float(context, name=strategy_name)
    if visual_mask is None or visual_mask.ndim != 1 or len(visual_mask) != len(features):
        raise ValueError(f"{strategy_name} requires a full-context visual mask")
    visual_positions = visual_mask.nonzero(as_tuple=False).flatten()
    if not len(visual_positions):
        raise ValueError(f"{strategy_name} requires visual tokens")
    positions = torch.arange(len(features), device=features.device)
    text_after_image = (~visual_mask) & (positions > visual_positions[-1])
    text = features[text_after_image]
    if not len(text):
        text = features[~visual_mask]
    if not len(text):
        raise ValueError(f"{strategy_name} requires at least one text token")
    return text


def _divprune_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Clean-room max-min diversity selection based on the DivPrune paper."""

    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    source = request.value_states if request.value_states is not None else request.features
    features = _flat_float(source, name="divprune")
    candidates = torch.nn.functional.normalize(features, dim=-1)
    target = request.keep_count
    if target == len(candidates):
        selected = torch.arange(target, device=request.device)
    else:
        distances = 1.0 - candidates @ candidates.T
        if len(candidates) == 1:
            selected = torch.zeros(1, dtype=torch.long, device=request.device)
        else:
            nearest_other = distances.topk(2, dim=0, largest=False).values[1]
            first = nearest_other.argmax()
            chosen = [first]
            chosen_mask = torch.zeros(len(candidates), dtype=torch.bool, device=request.device)
            chosen_mask[first] = True
            while len(chosen) < target:
                minimum_distance = distances.index_select(0, torch.stack(chosen)).min(dim=0).values
                minimum_distance.masked_fill_(chosen_mask, -float("inf"))
                next_index = minimum_distance.argmax()
                chosen.append(next_index)
                chosen_mask[next_index] = True
            selected = torch.stack(chosen)
    return selected.sort().values


def _dart_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """DART-style duplication removal adapted to an exact fixed budget."""

    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    visual_keys = _flat_float(request.key_states, name="dart")
    visual_values = torch.nn.functional.normalize(
        _flat_float(request.value_states, name="dart"),
        dim=-1,
    )
    text_keys = _decoder_text_features(
        request,
        source="key",
        strategy_name="dart",
    )
    text_values = torch.nn.functional.normalize(
        _decoder_text_features(
            request,
            source="value",
            strategy_name="dart",
        ),
        dim=-1,
    )

    requested_image_pivots = int(request.options.get("pivot_image_tokens", 4))
    requested_text_pivots = int(request.options.get("pivot_text_tokens", 4))
    if requested_image_pivots <= 0 or requested_text_pivots < 0:
        raise ValueError("dart pivot counts must satisfy image > 0 and text >= 0")
    # The released DART defaults use four image and four text pivots.  At 5%
    # that exceeds the complete budget, so scale image pivots down and leave
    # room for at least one duplication-aware farthest-token selection.
    image_pivot_count = min(
        requested_image_pivots,
        max(1, request.keep_count // 2),
        request.token_count,
    )
    image_scores = visual_keys.abs().sum(dim=-1)
    image_pivots = image_scores.topk(image_pivot_count, sorted=True).indices
    text_pivot_count = min(requested_text_pivots, len(text_keys))
    text_pivots = (
        text_keys.abs().sum(dim=-1).topk(text_pivot_count, sorted=True).indices
        if text_pivot_count
        else torch.empty(0, dtype=torch.long, device=request.device)
    )

    selected_mask = torch.zeros(request.token_count, dtype=torch.bool, device=request.device)
    selected_mask[image_pivots] = True
    pivot_vectors = [visual_values[index] for index in image_pivots]
    pivot_vectors.extend(text_values[index] for index in text_pivots)
    if not pivot_vectors:
        pivot_vectors.append(visual_values[image_scores.argmax()])

    step = 0
    while int(selected_mask.sum()) < request.keep_count:
        candidates = (~selected_mask).nonzero(as_tuple=False).flatten()
        pivot = pivot_vectors[step % len(pivot_vectors)]
        similarities = visual_values.index_select(0, candidates) @ pivot
        selected_mask[candidates[similarities.argmin()]] = True
        step += 1
    return selected_mask.nonzero(as_tuple=False).flatten()


def _greedy_prune_strategy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Semantic-saliency sorting plus greedy redundancy suppression."""

    if request.keep_count == request.token_count:
        return torch.arange(request.token_count, device=request.device)
    threshold = float(request.options.get("similarity_threshold", 0.9))
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("greedy_prune similarity_threshold must be in [-1, 1]")

    visual = torch.nn.functional.normalize(
        _flat_float(request.value_states, name="greedy_prune"),
        dim=-1,
    )
    last_text = torch.nn.functional.normalize(
        _decoder_text_features(
            request,
            source="value",
            strategy_name="greedy_prune",
        )[-1],
        dim=0,
    )
    target = request.keep_count
    candidates = visual
    saliency = candidates @ last_text
    order = saliency.argsort(descending=True, stable=True)
    active = torch.ones(len(candidates), dtype=torch.bool, device=request.device)
    selected: list[torch.Tensor] = []
    for pivot in order:
        if not bool(active[pivot]):
            continue
        selected.append(pivot)
        if len(selected) == target:
            break
        redundant = (candidates @ candidates[pivot]) > threshold
        active &= ~redundant

    # The paper permits early termination when no candidates remain.  The
    # experiment protocol requires an exact K for controlled ratio comparisons,
    # so deterministically fill from the remaining saliency order if necessary.
    if len(selected) < target:
        selected_ids = torch.zeros(len(candidates), dtype=torch.bool, device=request.device)
        if selected:
            selected_ids[torch.stack(selected)] = True
        for candidate in order:
            if not bool(selected_ids[candidate]):
                selected.append(candidate)
                selected_ids[candidate] = True
                if len(selected) == target:
                    break

    return torch.stack(selected).sort().values


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
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        token_count = len(features)
        effective_ratio = self.config.keep_ratio if keep_ratio is None else float(keep_ratio)
        keep_count = compute_keep_count(token_count, effective_ratio)
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
        context_query_states: torch.Tensor | None = None,
        context_key_states: torch.Tensor | None = None,
        context_value_states: torch.Tensor | None = None,
        visual_context_mask: torch.Tensor | None = None,
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        token_count = len(key_states)
        effective_ratio = self.config.keep_ratio if keep_ratio is None else float(keep_ratio)
        keep_count = compute_keep_count(token_count, effective_ratio)
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
            context_query_states=context_query_states,
            context_key_states=context_key_states,
            context_value_states=context_value_states,
            visual_context_mask=visual_context_mask,
            layer_index=layer_index,
            options=self.config.selector_kwargs,
        )
        return run_vision_token_strategy(self.config.selector, request)


register_vision_token_strategy("random", _random_strategy)
register_vision_token_strategy("uniform", _uniform_strategy)
register_vision_token_strategy("key_norm", _key_norm_strategy)
register_vision_token_strategy("embedding_norm", _embedding_norm_strategy)
register_vision_token_strategy("dart", _dart_strategy)
register_vision_token_strategy("divprune", _divprune_strategy)
register_vision_token_strategy("greedy_prune", _greedy_prune_strategy)
