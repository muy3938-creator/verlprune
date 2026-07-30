"""Training-side replay of rollout pruning decisions.

During rollout (vLLM), the pruning module selects which visual tokens to
keep and records the decision as ``keep_indices``.  During actor training,
this module *replays* the exact same decision so that the actor forward
pass sees the identical pruned sequence, guaranteeing logprob alignment.

Typical flow::

    # --- Rollout side (after generation) ---
    keep_indices = select_vision_tokens(image_embeds, keep_ratio=0.5)
    attach_keep_indices_to_sample(sample_mm_inputs, keep_indices, n_visual)

    # --- Actor side (before forward) ---
    # The keep_mask flows automatically through extract_multi_modal_inputs
    # → model(**multi_modal_inputs) → _get_input_embeds(vision_token_keep_mask=...)
"""

from __future__ import annotations

from typing import Any

import torch

from .sequence_compressor import KEEP_MASK_KEY, indices_to_keep_mask

# Key used to store the serialised selection record in multi_modal_inputs
SELECTION_KEY = "vision_token_selection"


# ---------------------------------------------------------------------------
# Rollout → sample attachment
# ---------------------------------------------------------------------------

def attach_keep_indices_to_sample(
    multi_modal_inputs: dict[str, Any],
    keep_indices: torch.Tensor,
    original_visual_token_count: int,
    *,
    keep_ratio: float | None = None,
    method: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach pruning decision to a single sample's multi_modal_inputs.

    This should be called during rollout after ``select_vision_tokens``
    produces ``keep_indices``.  The keep_mask (a tensor) is added to
    ``multi_modal_inputs`` so it flows through ``extract_multi_modal_inputs``
    → ``torch.cat`` → ``model(**multi_modal_inputs)`` naturally.

    A separate *selection_record* dict is returned for storage in
    ``meta_info`` (not in ``multi_modal_inputs``, because
    ``extract_multi_modal_inputs`` applies ``torch.cat`` to all values
    and a dict would break).

    Args:
        multi_modal_inputs: The per-sample multi_modal_inputs dict (will be
            modified in place and returned).
        keep_indices: Sorted 1-D tensor of kept visual token indices.
        original_visual_token_count: Total number of visual tokens before
            pruning (used for validation during replay).
        keep_ratio: Optional record of the keep ratio used.
        method: Optional record of the selection method used.

    Returns:
        A tuple of ``(multi_modal_inputs, selection_record)``.
        - ``multi_modal_inputs``: now contains ``vision_token_keep_mask``.
        - ``selection_record``: a plain dict for storage in ``meta_info``.
    """
    if KEEP_MASK_KEY in multi_modal_inputs:
        raise ValueError("vision_token_keep_mask is already attached to this sample")

    keep_mask = indices_to_keep_mask(keep_indices, original_visual_token_count)
    multi_modal_inputs[KEEP_MASK_KEY] = keep_mask

    # Lightweight selection record for debugging / validation.
    # Store in sample meta_info, NOT in multi_modal_inputs.
    selection_record = {
        "kept_indices": keep_indices.cpu().tolist(),
        "original_count": original_visual_token_count,
        "kept_count": int(keep_indices.numel()),
        "keep_ratio": keep_ratio,
        "method": method,
    }
    return multi_modal_inputs, selection_record


def strip_pruning_from_sample(
    multi_modal_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Remove all pruning metadata — used for unpruned teacher forward."""
    return {
        key: value
        for key, value in multi_modal_inputs.items()
        if key not in {KEEP_MASK_KEY, SELECTION_KEY}
    }


# ---------------------------------------------------------------------------
# Actor-side validation
# ---------------------------------------------------------------------------

def validate_replay_alignment(
    input_ids: torch.Tensor,
    multi_modal_inputs: dict[str, Any],
    image_token_id: int,
    sample_index: int = 0,
    selection_record: dict[str, Any] | None = None,
) -> None:
    """Validate that the keep_mask in multi_modal_inputs is consistent.

    Call this in debug mode to verify that rollout → actor replay is correct.

    Args:
        input_ids: Token IDs for this sample.
        multi_modal_inputs: The per-sample multimodal inputs containing
            ``vision_token_keep_mask``.
        image_token_id: The integer ID of ``<|image_pad|>``.
        sample_index: For error messages.
        selection_record: Optional selection record from ``meta_info``
            (returned by ``attach_keep_indices_to_sample``).

    Raises:
        ValueError: If the keep_mask length doesn't match the number of
            image tokens in the input sequence.
    """
    if KEEP_MASK_KEY not in multi_modal_inputs:
        return  # No pruning metadata, nothing to validate

    keep_mask = multi_modal_inputs[KEEP_MASK_KEY]
    n_image_tokens = (input_ids == image_token_id).sum().item()

    if keep_mask.numel() != n_image_tokens:
        raise ValueError(
            f"Sample {sample_index}: vision_token_keep_mask length "
            f"({keep_mask.numel()}) does not match the number of image "
            f"tokens in input_ids ({n_image_tokens})"
        )

    if selection_record is not None:
        expected_count = selection_record.get("original_count")
        if expected_count is not None and expected_count != n_image_tokens:
            raise ValueError(
                f"Sample {sample_index}: rollout recorded "
                f"{expected_count} visual tokens but actor input has "
                f"{n_image_tokens}"
            )


def apply_pre_llm_pruning_to_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs_list: list[dict[str, Any] | None],
    image_token_id: int,
) -> torch.Tensor:
    """Zero out dropped image token positions in attention_mask before unpad_input.

    This ensures that unpad_input (and FlashAttention) in dp_actor physically
    removes dropped visual tokens, reducing sequence length and saving FLOPs.

    Args:
        input_ids: Micro-batch token IDs tensor (B, S).
        attention_mask: Micro-batch attention mask tensor (B, S).
        multi_modal_inputs_list: List of per-sample multi_modal_inputs dicts.
        image_token_id: The integer token ID of ``<|image_pad|>``.

    Returns:
        A modified attention_mask tensor with 0s at dropped image token positions.
    """
    if attention_mask is None:
        return None

    attention_mask = attention_mask.clone()
    for sample_idx, mm_inputs in enumerate(multi_modal_inputs_list):
        if mm_inputs is None or KEEP_MASK_KEY not in mm_inputs:
            continue
        keep_mask = mm_inputs[KEEP_MASK_KEY]
        sample_ids = input_ids[sample_idx]
        image_positions = ((sample_ids == image_token_id) & attention_mask[sample_idx].bool()).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != keep_mask.numel():
            raise ValueError(
                f"Sample {sample_idx}: image token count in input_ids ({image_positions.numel()}) "
                f"does not match keep_mask length ({keep_mask.numel()})"
            )
        drop_indices = image_positions[~keep_mask.to(image_positions.device)]
        attention_mask[sample_idx, drop_indices] = 0

    return attention_mask


