from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class VisionTokenPruningConfig:
    """Layer-0 random visual-token pruning configuration."""

    enabled: bool = False
    keep_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"vision token pruning keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.enabled and self.keep_ratio == 1.0:
            raise ValueError("enabled vision token pruning requires keep_ratio < 1")


def coerce_vision_token_pruning_config(value: Any) -> VisionTokenPruningConfig:
    """Normalize Hydra/dict/dataclass inputs at integration boundaries."""

    if value is None:
        return VisionTokenPruningConfig()
    if isinstance(value, VisionTokenPruningConfig):
        return value
    if isinstance(value, Mapping):
        return VisionTokenPruningConfig(**dict(value))
    raise TypeError(f"unsupported vision token pruning config type: {type(value).__name__}")
