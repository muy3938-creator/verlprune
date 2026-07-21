"""Hydra-facing config. Constructs and freezes PruningSpec — the real plan."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .curriculum import validate_keep_ratio_schedule

_RESERVED_SELECTOR_KWARGS = {
    "token_count",
    "keep_count",
    "device",
    "generator",
    "features",
    "grid_thw",
    "query_states",
    "key_states",
    "value_states",
    "context_query_states",
    "context_key_states",
    "context_value_states",
    "visual_context_mask",
    "layer_index",
}


def compute_selector_fingerprint(selector: str, selector_kwargs: Mapping[str, Any] | None = None) -> str:
    """Stable identity for one fully configured policy."""

    normalized_name = selector.strip()
    if not normalized_name:
        raise ValueError("selector must be a non-empty name")
    options = dict(selector_kwargs or {})
    conflicts = _RESERVED_SELECTOR_KWARGS.intersection(options)
    if conflicts:
        raise ValueError(f"selector_kwargs cannot override reserved inputs: {sorted(conflicts)}")
    try:
        payload = json.dumps(
            {"selector": normalized_name, "selector_kwargs": options},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selector_kwargs must be JSON serializable") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_policy_kwargs(selector: str, kwargs: Mapping[str, Any]) -> None:
    """Fail fast on unknown / invalid policy options (no silent ignore)."""

    if selector == "vision_pulse":
        allowed = {
            "budget_mode",
            "temperature",
            "top_p",
            "budget_schedule",
            "min_keep_ratio",
            "max_keep_ratio",
            "capture_capacity",
        }
        unknown = set(kwargs).difference(allowed)
        if unknown:
            raise ValueError(f"unsupported vision_pulse selector_kwargs: {sorted(unknown)}")
        budget_mode = str(kwargs.get("budget_mode", "visual_mass"))
        if budget_mode not in {"fixed", "visual_mass", "top_p"}:
            raise ValueError("vision_pulse budget_mode must be 'fixed', 'visual_mass', or 'top_p'")
        top_p = float(kwargs.get("top_p", 0.95))
        if not 0.0 < top_p <= 1.0:
            raise ValueError("vision_pulse top_p must be in (0, 1]")
        schedule = kwargs.get("budget_schedule", ())
        if schedule is None:
            schedule = ()
        if not isinstance(schedule, (list, tuple)) or not schedule:
            if schedule not in ((), []):
                raise ValueError("vision_pulse budget_schedule must be a non-empty list")
        else:
            schedule = tuple(float(value) for value in schedule)
            if any(not 0.0 < value <= 1.0 for value in schedule):
                raise ValueError("vision_pulse budget_schedule values must be in (0, 1]")
        if float(kwargs.get("temperature", 0.1)) <= 0:
            raise ValueError("vision_pulse temperature must be positive")
        minimum = float(kwargs.get("min_keep_ratio", 0.0))
        maximum = float(kwargs.get("max_keep_ratio", 1.0))
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("vision_pulse keep-ratio clamps must satisfy 0 <= min <= max <= 1")
        if int(kwargs.get("capture_capacity", 64)) <= 0:
            raise ValueError("vision_pulse capture_capacity must be positive")
        return
    if selector == "dart":
        allowed = {"pivot_image_tokens", "pivot_text_tokens"}
        unknown = set(kwargs).difference(allowed)
        if unknown:
            raise ValueError(f"unsupported dart selector_kwargs: {sorted(unknown)}")
        image_pivots = kwargs.get("pivot_image_tokens", 4)
        text_pivots = kwargs.get("pivot_text_tokens", 4)
        if (
            not isinstance(image_pivots, int)
            or isinstance(image_pivots, bool)
            or image_pivots <= 0
            or not isinstance(text_pivots, int)
            or isinstance(text_pivots, bool)
            or text_pivots < 0
        ):
            raise ValueError("dart pivot counts must satisfy image > 0 and text >= 0")
        return
    if selector == "greedy_prune":
        unknown = set(kwargs).difference({"similarity_threshold"})
        if unknown:
            raise ValueError(f"unsupported greedy_prune selector_kwargs: {sorted(unknown)}")
        threshold = float(kwargs.get("similarity_threshold", 0.9))
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("greedy_prune similarity_threshold must be in [-1, 1]")
        return
    if selector == "divprune" and kwargs:
        raise ValueError(f"unsupported divprune selector_kwargs: {sorted(kwargs)}")


@dataclass
class VisionTokenPruningConfig:
    """Flat fields for Hydra. Execution plan is ``self.spec`` (PruningSpec).

    Field meanings (map to stages, do not invent a second logic tree):

    - ``prune_after_layer=-1`` + vision embeddings → ``physical_pre_decoder``
    - ``prune_after_layer=L>=0`` → ``boundary_once`` or ``decode_query``
    - ``prefill_keep_ratio`` set → two stages (prefill + decode_query)
    """

    enabled: bool = False
    keep_ratio: float = 0.5
    prune_after_layer: int = -1
    layerwise_backend: str = "flex"
    pre_pruning_backend: str = "flex"
    selector_input: str = "vision_embedding"
    selector: str = "embedding_norm"
    selector_kwargs: dict[str, Any] = field(default_factory=dict)
    keep_ratio_schedule: dict[str, Any] = field(default_factory=dict)
    prefill_keep_ratio: float | None = None
    prefill_selector: str = "embedding_norm"
    prefill_selector_kwargs: dict[str, Any] = field(default_factory=dict)
    prefill_prune_after_layer: int = -1

    def __post_init__(self) -> None:
        if not self.selector or not self.selector.strip():
            raise ValueError("vision token pruning selector must be a non-empty name")
        self.selector = self.selector.strip()
        self.selector_kwargs = dict(self.selector_kwargs)
        self.keep_ratio_schedule = dict(self.keep_ratio_schedule or {})
        validate_keep_ratio_schedule(self.keep_ratio_schedule)
        self.prefill_selector = self.prefill_selector.strip()
        self.prefill_selector_kwargs = dict(self.prefill_selector_kwargs)

        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"vision token pruning keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.enabled and self.keep_ratio == 1.0:
            raise ValueError("enabled vision token pruning requires keep_ratio < 1")
        if self.prefill_keep_ratio is not None:
            if not 0.0 < self.prefill_keep_ratio < 1.0:
                raise ValueError("prefill_keep_ratio must be in (0, 1)")
            if not self.prefill_selector:
                raise ValueError("prefill_selector must be a non-empty name")
        if self.prune_after_layer < -1:
            raise ValueError("vision token pruning prune_after_layer must be >= -1")
        if self.prefill_prune_after_layer < -1:
            raise ValueError("prefill_prune_after_layer must be >= -1")
        if self.prefill_keep_ratio is None and self.prefill_prune_after_layer != -1:
            raise ValueError(
                "prefill_prune_after_layer requires prefill_keep_ratio to enable two-stage pruning"
            )
        if self.layerwise_backend not in {"flex", "compact_flash"}:
            raise ValueError("vision token pruning layerwise_backend must be 'flex' or 'compact_flash'")
        if self.keep_ratio_schedule and self.layerwise_backend != "flex":
            raise ValueError("keep_ratio_schedule currently requires layerwise_backend='flex'")
        if self.pre_pruning_backend not in {"flex", "flash"}:
            raise ValueError("vision token pruning pre_pruning_backend must be 'flex' or 'flash'")
        if self.pre_pruning_backend == "flash":
            if self.prune_after_layer < 0:
                raise ValueError("pre_pruning_backend='flash' requires prune_after_layer >= 0")
            if self.layerwise_backend != "flex":
                raise ValueError("pre_pruning_backend='flash' requires layerwise_backend='flex'")
        if self.selector_input not in {"vision_embedding", "decoder_key", "decode_query"}:
            raise ValueError(
                "vision token pruning selector_input must be 'vision_embedding', "
                "'decoder_key', or 'decode_query'"
            )
        if self.selector_input in {"decoder_key", "decode_query"} and self.prune_after_layer < 0:
            raise ValueError(f"{self.selector_input} selection requires prune_after_layer >= 0")
        if self.selector_input == "decoder_key" and self.layerwise_backend != "flex":
            raise ValueError("decoder_key selection requires layerwise_backend='flex'")
        if self.selector_input == "decode_query" and self.layerwise_backend != "flex":
            raise ValueError("decode_query selection requires layerwise_backend='flex'")
        if self.selector_input == "decode_query" and self.selector != "vision_pulse":
            raise ValueError("decode_query selection currently requires selector='vision_pulse'")
        if self.selector == "vision_pulse" and self.selector_input != "decode_query":
            raise ValueError("selector='vision_pulse' requires selector_input='decode_query'")
        if self.prefill_keep_ratio is not None:
            if self.prune_after_layer < 0:
                raise ValueError("two-stage pruning requires prune_after_layer >= 0")
            if self.layerwise_backend != "flex":
                raise ValueError("two-stage pruning requires layerwise_backend='flex'")
            if self.selector_input != "decode_query" or self.selector != "vision_pulse":
                raise ValueError(
                    "two-stage pruning requires selector_input='decode_query' "
                    "and selector='vision_pulse'"
                )
            if self.prefill_prune_after_layer > self.prune_after_layer:
                raise ValueError(
                    "two-stage pruning requires prefill_prune_after_layer <= prune_after_layer"
                )
            if self.prefill_selector in {"vision_pulse", "dart", "greedy_prune", "key_norm"}:
                raise ValueError(
                    f"prefill_selector={self.prefill_selector!r} cannot select from vision embeddings"
                )
        if self.selector in {"dart", "greedy_prune"} and self.selector_input != "decoder_key":
            raise ValueError(f"selector={self.selector!r} requires selector_input='decoder_key'")

        _validate_policy_kwargs(self.selector, self.selector_kwargs)
        if self.prefill_keep_ratio is not None:
            _validate_policy_kwargs(self.prefill_selector, self.prefill_selector_kwargs)
        compute_selector_fingerprint(self.selector, self.selector_kwargs)
        compute_selector_fingerprint(self.prefill_selector, self.prefill_selector_kwargs)

        # Single source of truth for execution semantics.
        from .stages import pruning_spec_from_legacy_config

        object.__setattr__(self, "_spec", pruning_spec_from_legacy_config(self))

    @property
    def spec(self):
        """Frozen PruningSpec — the only execution plan."""

        return self._spec

    def to_pruning_spec(self):
        return self.spec

    # --- derived from spec (no second validation matrix) ---

    @property
    def uses_layerwise_backend(self) -> bool:
        return self.spec.uses_layerwise_backend

    @property
    def uses_dynamic_decode_selection(self) -> bool:
        return self.spec.uses_dynamic_decode_selection

    @property
    def uses_keep_ratio_schedule(self) -> bool:
        return self.spec.uses_keep_ratio_schedule

    @property
    def uses_two_stage_pruning(self) -> bool:
        return self.spec.uses_two_stage_pruning

    @property
    def uses_physical_prefill_pruning(self) -> bool:
        return self.spec.uses_physical_prefill_pruning

    @property
    def uses_delayed_prefill_pruning(self) -> bool:
        return self.spec.uses_delayed_prefill_pruning

    @property
    def backend_name(self) -> str:
        return self.spec.backend_name

    @property
    def selection_layer(self) -> int:
        """User-facing boundary; zero means immediately before decoder layer 0."""

        if not self.uses_layerwise_backend:
            return 0
        return self.prune_after_layer

    @property
    def selector_fingerprint(self) -> str:
        return compute_selector_fingerprint(self.selector, self.selector_kwargs)

    def to_backend_payload(self) -> dict[str, Any]:
        payload = {
            "keep_ratio": self.keep_ratio,
            "selector": self.selector,
            "selector_kwargs": dict(self.selector_kwargs),
        }
        if self.keep_ratio_schedule:
            payload["keep_ratio_schedule"] = dict(self.keep_ratio_schedule)
        if self.uses_layerwise_backend:
            payload["prune_after_layer"] = self.prune_after_layer
            payload["layerwise_backend"] = self.layerwise_backend
            payload["pre_pruning_backend"] = self.pre_pruning_backend
            payload["selector_input"] = self.selector_input
        if self.uses_two_stage_pruning:
            payload["prefill_keep_ratio"] = self.prefill_keep_ratio
            payload["prefill_selector"] = self.prefill_selector
            payload["prefill_selector_kwargs"] = dict(self.prefill_selector_kwargs)
            payload["prefill_prune_after_layer"] = self.prefill_prune_after_layer
        return payload


def coerce_vision_token_pruning_config(value: Any) -> VisionTokenPruningConfig:
    if value is None:
        return VisionTokenPruningConfig()
    if isinstance(value, VisionTokenPruningConfig):
        return value
    if isinstance(value, Mapping):
        return VisionTokenPruningConfig(**dict(value))
    raise TypeError(f"unsupported vision token pruning config type: {type(value).__name__}")
