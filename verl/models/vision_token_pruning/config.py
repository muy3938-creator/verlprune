import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    """Return a stable identity for one fully configured selection strategy."""

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


@dataclass
class VisionTokenPruningConfig:
    """Backend-neutral visual-token pruning configuration.

    ``prune_after_layer=-1`` keeps the validated layer-0 physical-pruning
    baseline. Non-negative values select the experimental layerwise
    Attention/KV implementation and apply pruning starting at the next layer.
    ``selector`` names the rollout-side algorithm; the actor only replays the
    resulting indices and therefore stays independent of that algorithm.
    """

    enabled: bool = False
    keep_ratio: float = 0.5
    prune_after_layer: int = -1
    layerwise_backend: str = "flex"
    selector_input: str = "vision_embedding"
    selector: str = "random"
    selector_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selector or not self.selector.strip():
            raise ValueError("vision token pruning selector must be a non-empty name")
        self.selector = self.selector.strip()
        self.selector_kwargs = dict(self.selector_kwargs)
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"vision token pruning keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.enabled and self.keep_ratio == 1.0:
            raise ValueError("enabled vision token pruning requires keep_ratio < 1")
        if self.prune_after_layer < -1:
            raise ValueError("vision token pruning prune_after_layer must be >= -1")
        if self.layerwise_backend not in {"flex", "compact_flash"}:
            raise ValueError("vision token pruning layerwise_backend must be 'flex' or 'compact_flash'")
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
        if self.selector in {"dart", "greedy_prune"} and self.selector_input != "decoder_key":
            raise ValueError(
                f"selector={self.selector!r} requires selector_input='decoder_key'"
            )
        if self.selector_input == "decode_query":
            allowed_dynamic_options = {
                "budget_mode",
                "temperature",
                "min_keep_ratio",
                "max_keep_ratio",
                "capture_capacity",
            }
            unknown = set(self.selector_kwargs).difference(allowed_dynamic_options)
            if unknown:
                raise ValueError(
                    "unsupported vision_pulse selector_kwargs: "
                    f"{sorted(unknown)}"
                )
            budget_mode = str(self.selector_kwargs.get("budget_mode", "visual_mass"))
            if budget_mode not in {"fixed", "visual_mass"}:
                raise ValueError(
                    "vision_pulse budget_mode must be 'fixed' or 'visual_mass'"
                )
            temperature = float(self.selector_kwargs.get("temperature", 0.1))
            if temperature <= 0:
                raise ValueError("vision_pulse temperature must be positive")
            minimum = float(self.selector_kwargs.get("min_keep_ratio", 0.0))
            maximum = float(self.selector_kwargs.get("max_keep_ratio", 1.0))
            if not 0.0 <= minimum <= maximum <= 1.0:
                raise ValueError(
                    "vision_pulse keep-ratio clamps must satisfy 0 <= min <= max <= 1"
                )
            capture_capacity = int(self.selector_kwargs.get("capture_capacity", 64))
            if capture_capacity <= 0:
                raise ValueError("vision_pulse capture_capacity must be positive")
        elif self.selector == "dart":
            allowed_dart_options = {"pivot_image_tokens", "pivot_text_tokens"}
            unknown = set(self.selector_kwargs).difference(allowed_dart_options)
            if unknown:
                raise ValueError(f"unsupported dart selector_kwargs: {sorted(unknown)}")
            image_pivots = self.selector_kwargs.get("pivot_image_tokens", 4)
            text_pivots = self.selector_kwargs.get("pivot_text_tokens", 4)
            if (
                not isinstance(image_pivots, int)
                or isinstance(image_pivots, bool)
                or image_pivots <= 0
                or not isinstance(text_pivots, int)
                or isinstance(text_pivots, bool)
                or text_pivots < 0
            ):
                raise ValueError("dart pivot counts must satisfy image > 0 and text >= 0")
        elif self.selector == "greedy_prune":
            unknown = set(self.selector_kwargs).difference({"similarity_threshold"})
            if unknown:
                raise ValueError(
                    f"unsupported greedy_prune selector_kwargs: {sorted(unknown)}"
                )
            threshold = float(self.selector_kwargs.get("similarity_threshold", 0.9))
            if not -1.0 <= threshold <= 1.0:
                raise ValueError("greedy_prune similarity_threshold must be in [-1, 1]")
        elif self.selector == "divprune" and self.selector_kwargs:
            raise ValueError(
                f"unsupported divprune selector_kwargs: {sorted(self.selector_kwargs)}"
            )
        compute_selector_fingerprint(self.selector, self.selector_kwargs)

    @property
    def uses_layerwise_backend(self) -> bool:
        return self.enabled and self.prune_after_layer >= 0

    @property
    def uses_dynamic_decode_selection(self) -> bool:
        return self.uses_layerwise_backend and self.selector_input == "decode_query"

    @property
    def backend_name(self) -> str:
        if not self.uses_layerwise_backend:
            return "physical"
        return f"layerwise_{self.layerwise_backend}"

    @property
    def selector_fingerprint(self) -> str:
        return compute_selector_fingerprint(self.selector, self.selector_kwargs)

    def to_backend_payload(self) -> dict[str, Any]:
        payload = {
            "keep_ratio": self.keep_ratio,
            "selector": self.selector,
            "selector_kwargs": dict(self.selector_kwargs),
        }
        if self.uses_layerwise_backend:
            payload["prune_after_layer"] = self.prune_after_layer
            payload["layerwise_backend"] = self.layerwise_backend
            payload["selector_input"] = self.selector_input
        return payload


def coerce_vision_token_pruning_config(value: Any) -> VisionTokenPruningConfig:
    """Normalize Hydra/dict/dataclass inputs at integration boundaries."""

    if value is None:
        return VisionTokenPruningConfig()
    if isinstance(value, VisionTokenPruningConfig):
        return value
    if isinstance(value, Mapping):
        return VisionTokenPruningConfig(**dict(value))
    raise TypeError(f"unsupported vision token pruning config type: {type(value).__name__}")
