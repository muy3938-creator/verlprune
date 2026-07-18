from dataclasses import dataclass


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
