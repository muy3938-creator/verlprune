from __future__ import annotations

from typing import Any

import torch

from .protocol import VisionTokenSelection

KEEP_MASK_KEY = "vision_token_keep_mask"
SELECTION_WIRE_KEY = "vision_token_selection"


def selection_to_keep_mask(selection: VisionTokenSelection, *, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros(selection.original_visual_token_count, dtype=torch.bool, device=device)
    mask[list(selection.kept_visual_indices)] = True
    return mask


def attach_selection_to_multi_modal_inputs(
    multi_modal_inputs: dict[str, Any],
    selection_wire: dict[str, Any],
) -> dict[str, Any]:
    if KEEP_MASK_KEY in multi_modal_inputs or SELECTION_WIRE_KEY in multi_modal_inputs:
        raise ValueError("visual-token pruning metadata is already attached")
    selection = VisionTokenSelection.from_wire(selection_wire)
    return {
        **multi_modal_inputs,
        KEEP_MASK_KEY: selection_to_keep_mask(selection),
        SELECTION_WIRE_KEY: selection.to_wire(),
    }


def strip_pruning_metadata(multi_modal_inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove pruning fields before an unpruned teacher forward."""

    return {
        key: value
        for key, value in multi_modal_inputs.items()
        if key not in {KEEP_MASK_KEY, SELECTION_WIRE_KEY}
    }


def strip_selection_metadata(multi_modal_inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the model-facing mask and remove its serialized rollout record."""

    return {key: value for key, value in multi_modal_inputs.items() if key != SELECTION_WIRE_KEY}


def prune_visual_embeddings(
    embeddings: torch.Tensor,
    keep_mask: torch.Tensor | None,
    *,
    name: str = KEEP_MASK_KEY,
) -> torch.Tensor:
    if keep_mask is None:
        return embeddings
    if embeddings.ndim != 2:
        raise ValueError("visual embeddings must be a rank-2 tensor")
    if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a rank-1 bool tensor")
    if keep_mask.numel() != embeddings.shape[0]:
        raise ValueError(
            f"{name} covers {keep_mask.numel()} visual tokens, but model produced {embeddings.shape[0]} features"
        )
    return embeddings[keep_mask.to(embeddings.device)]


def apply_rollout_pruning_to_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    per_sample_multi_modal_inputs: list[dict[str, Any] | None],
    *,
    image_token_id: int,
    expected_keep_ratio: float,
) -> torch.Tensor:
    """Remove the exact rollout-selected image tokens from the actor sequence."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have identical rank-2 shapes")
    if len(per_sample_multi_modal_inputs) != input_ids.shape[0]:
        raise ValueError("multi_modal_inputs length must match batch size")

    output = attention_mask.clone()
    for sample_index, multi_modal_inputs in enumerate(per_sample_multi_modal_inputs):
        if multi_modal_inputs is None:
            raise ValueError(f"sample {sample_index} is missing multimodal inputs")
        if KEEP_MASK_KEY not in multi_modal_inputs or SELECTION_WIRE_KEY not in multi_modal_inputs:
            raise ValueError(f"sample {sample_index} is missing rollout visual-token selection")

        keep_mask = multi_modal_inputs[KEEP_MASK_KEY]
        selection = VisionTokenSelection.from_wire(multi_modal_inputs[SELECTION_WIRE_KEY])
        if selection.keep_ratio != expected_keep_ratio:
            raise ValueError(
                f"sample {sample_index} rollout keep_ratio {selection.keep_ratio} "
                f"does not match actor keep_ratio {expected_keep_ratio}"
            )
        if not torch.equal(keep_mask.cpu(), selection_to_keep_mask(selection)):
            raise ValueError(f"sample {sample_index} keep mask does not match rollout selection")

        image_positions = (
            (input_ids[sample_index] == image_token_id) & output[sample_index].bool()
        ).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != selection.original_visual_token_count:
            raise ValueError(
                f"sample {sample_index} selection covers {selection.original_visual_token_count} visual tokens, "
                f"but input contains {image_positions.numel()} image tokens"
            )
        output[sample_index, image_positions[~keep_mask.to(image_positions.device)]] = 0
    return output
