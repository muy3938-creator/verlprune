from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"vision token pruning keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.enabled and self.keep_ratio == 1.0:
            raise ValueError("enabled vision token pruning requires keep_ratio < 1")
        if self.prune_after_layer < -1:
            raise ValueError("vision token pruning prune_after_layer must be >= -1")
        if not self.selector or not self.selector.strip():
            raise ValueError("vision token pruning selector must be a non-empty name")

    @property
    def uses_layerwise_backend(self) -> bool:
        return self.enabled and self.prune_after_layer >= 0


def coerce_vision_token_pruning_config(value: Any) -> VisionTokenPruningConfig:
    """Normalize Hydra/dict/dataclass inputs at integration boundaries."""

    if value is None:
        return VisionTokenPruningConfig()
    if isinstance(value, VisionTokenPruningConfig):
        return value
    if isinstance(value, Mapping):
        return VisionTokenPruningConfig(**dict(value))
    raise TypeError(f"unsupported vision token pruning config type: {type(value).__name__}")
