"""Shared adapter utilities for physical and experimental layerwise plugins."""

from __future__ import annotations

from typing import Any

from verl.models.vision_token_pruning.config import (
    VisionTokenPruningConfig,
    coerce_vision_token_pruning_config,
)


def pruning_config_from_hf(hf_config: Any) -> VisionTokenPruningConfig:
    raw_config = getattr(hf_config, "vision_token_pruning", None)
    if not isinstance(raw_config, dict):
        raise ValueError("pruned vLLM model requires hf_config.vision_token_pruning")
    normalized = dict(raw_config)
    normalized["enabled"] = True
    return coerce_vision_token_pruning_config(normalized)
