import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_RESERVED_SELECTOR_KWARGS = {"token_count", "keep_count", "device", "generator", "features", "grid_thw"}


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
    selector: str = "random"
    selector_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"vision token pruning keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.enabled and self.keep_ratio == 1.0:
            raise ValueError("enabled vision token pruning requires keep_ratio < 1")
        if self.prune_after_layer < -1:
            raise ValueError("vision token pruning prune_after_layer must be >= -1")
        if not self.selector or not self.selector.strip():
            raise ValueError("vision token pruning selector must be a non-empty name")
        self.selector = self.selector.strip()
        self.selector_kwargs = dict(self.selector_kwargs)
        compute_selector_fingerprint(self.selector, self.selector_kwargs)

    @property
    def uses_layerwise_backend(self) -> bool:
        return self.enabled and self.prune_after_layer >= 0

    @property
    def backend_name(self) -> str:
        return "layerwise" if self.uses_layerwise_backend else "physical"

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
