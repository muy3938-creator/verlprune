"""Compatibility imports for the pre-milestone runtime module.

New integrations should import rollout replay from ``training`` and physical
feature compaction from ``embeddings``.
"""

from .embeddings import KEEP_MASK_KEY, prune_visual_embedding_outputs, prune_visual_embeddings
from .training import (
    SELECTION_WIRE_KEY,
    PreparedActorPruningInputs,
    attach_selection_to_multi_modal_inputs,
    prepare_actor_pruning_inputs,
    replay_rollout_selection_on_attention_mask,
    selection_to_keep_mask,
    strip_pruning_metadata,
    strip_selection_metadata,
)

__all__ = [
    "KEEP_MASK_KEY",
    "SELECTION_WIRE_KEY",
    "PreparedActorPruningInputs",
    "attach_selection_to_multi_modal_inputs",
    "prepare_actor_pruning_inputs",
    "prune_visual_embedding_outputs",
    "prune_visual_embeddings",
    "replay_rollout_selection_on_attention_mask",
    "selection_to_keep_mask",
    "strip_pruning_metadata",
    "strip_selection_metadata",
]
