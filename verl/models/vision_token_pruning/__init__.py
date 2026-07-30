"""Minimal Pre-LLM vision token pruning.

This package provides a single-stage Pre-LLM pruning pipeline that
physically removes visual tokens before they enter the language model
backbone, reducing FLOPs and KV-cache memory across all decoder layers.
"""

from .pre_llm_pruner import select_vision_tokens  # noqa: F401
from .sequence_compressor import (  # noqa: F401
    KEEP_MASK_KEY,
    compress_sequence,
    indices_to_keep_mask,
    prune_visual_embeddings,
    prune_visual_embedding_outputs,
)
from .training_replay import (  # noqa: F401
    apply_pre_llm_pruning_to_attention_mask,
    attach_keep_indices_to_sample,
    strip_pruning_from_sample,
    validate_replay_alignment,
)
from .vllm_transport import (  # noqa: F401
    clear_request_selection_registry,
    decode_vllm_selection_capture,
    pop_request_selection,
    register_request_selection,
)
