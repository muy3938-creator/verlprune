"""FlexAttention layerwise pruning helpers.

The monolithic plugin still lives at ``layerwise_flex_vision_token_pruning.py``
for registration compatibility. Shared observe/apply utilities are gradually
moved here so algorithm work does not require editing the 900-line file.
"""

from .plan import FORWARD_CONTEXT_KEY, LayerwiseFlexPruningPlan, active_plan, layer_index_from_name

__all__ = [
    "FORWARD_CONTEXT_KEY",
    "LayerwiseFlexPruningPlan",
    "active_plan",
    "layer_index_from_name",
]
