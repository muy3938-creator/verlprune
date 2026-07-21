"""Visual-token pruning platform.

Public surface for experiments and training:

- ``VisionTokenPruningConfig`` / ``PruningSpec`` — experiment plan
- ``VisionTokenSelectionRequest`` + policy registry — algorithms
- ``prepare_actor_pruning_inputs`` / transport — training wire (import from submodules)
"""

from .config import VisionTokenPruningConfig, compute_selector_fingerprint
from .protocol import DynamicVisionTokenSelection, TwoStageVisionTokenSelection, VisionTokenSelection
from .request import VisionTokenSelectionRequest
from .stages import InputKind, PruningSpec, RuntimeKind, StageKind, StageSpec, pruning_spec_from_legacy_config
from .strategy import (
    VisionTokenSelectionEngine,
    available_vision_token_strategies,
    register_vision_token_strategy,
    run_vision_token_strategy,
)

__all__ = [
    "InputKind",
    "PruningSpec",
    "RuntimeKind",
    "StageKind",
    "StageSpec",
    "VisionTokenPruningConfig",
    "VisionTokenSelectionEngine",
    "VisionTokenSelectionRequest",
    "VisionTokenSelection",
    "DynamicVisionTokenSelection",
    "TwoStageVisionTokenSelection",
    "available_vision_token_strategies",
    "compute_selector_fingerprint",
    "pruning_spec_from_legacy_config",
    "register_vision_token_strategy",
    "run_vision_token_strategy",
]
