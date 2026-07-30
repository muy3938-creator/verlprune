"""Physical sequence compression after Pre-LLM visual token selection.

Given a keep_mask (or keep_indices) of which visual tokens survived pruning,
this module drops the corresponding ``<|image_pad|>`` positions from
``input_ids``, ``inputs_embeds``, ``attention_mask``, and ``position_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


KEEP_MASK_KEY = "vision_token_keep_mask"


@dataclass
class CompressedSequence:
    """Result of physical sequence compression."""

    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    n_dropped: int


def prune_visual_embeddings(
    embeddings: torch.Tensor,
    keep_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a boolean keep_mask to flattened visual embeddings (N, D)."""
    if keep_mask is None:
        return embeddings
    if embeddings.ndim != 2:
        raise ValueError("visual embeddings must be rank-2")
    if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be a rank-1 bool tensor")
    if keep_mask.numel() != embeddings.shape[0]:
        raise ValueError(
            f"keep_mask length ({keep_mask.numel()}) != "
            f"embeddings count ({embeddings.shape[0]})"
        )
    return embeddings[keep_mask.to(embeddings.device)]


def prune_visual_embedding_outputs(
    embeddings: torch.Tensor,
    auxiliary_embeddings: Optional[list[torch.Tensor]],
    keep_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """Prune a primary visual output and any aligned deep-stack outputs."""
    pruned_embeddings = prune_visual_embeddings(embeddings, keep_mask)
    if auxiliary_embeddings is None:
        return pruned_embeddings, None
    return pruned_embeddings, [
        prune_visual_embeddings(auxiliary, keep_mask)
        for auxiliary in auxiliary_embeddings
    ]


def indices_to_keep_mask(indices: torch.Tensor, total: int) -> torch.Tensor:
    """Convert sorted kept-indices to a boolean mask of length *total*."""
    mask = torch.zeros(total, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask


def compress_sequence(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    image_token_id: int,
    keep_mask: torch.Tensor,
) -> CompressedSequence:
    """Physically drop pruned image-pad tokens from the packed sequence.

    Works on a single flattened (batch=1) sequence. For Qwen2-VL the
    ``position_ids`` are 3-D mROPE tensors of shape ``(3, 1, seq_len)`` or
    ``(3, seq_len)``.

    Args:
        input_ids: Token IDs, shape ``(1, seq_len)``.
        inputs_embeds: Embeddings, shape ``(1, seq_len, D)``.
        attention_mask: Shape ``(1, seq_len)``.
        position_ids: Shape ``(3, 1, seq_len)`` or ``(3, seq_len)`` or
            ``(1, seq_len)``.
        image_token_id: The integer ID of ``<|image_pad|>``.
        keep_mask: Boolean mask of length ``N_visual`` (total image tokens
            in the sequence), ``True`` for tokens to keep.

    Returns:
        A :class:`CompressedSequence` with shorter seq_len.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    bsz, seq_len = input_ids.shape
    if bsz != 1:
        raise ValueError("compress_sequence currently supports batch_size=1")

    # Locate all image token positions in the flat sequence
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    n_image = image_positions.numel()

    if keep_mask.numel() != n_image:
        raise ValueError(
            f"keep_mask length ({keep_mask.numel()}) does not match "
            f"number of image tokens ({n_image})"
        )

    # Positions to DROP = image positions where keep_mask is False
    drop_mask_seq = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
    dropped_positions = image_positions[~keep_mask.to(image_positions.device)]
    drop_mask_seq[dropped_positions] = False

    # Build the compressed sequence
    new_input_ids = input_ids[:, drop_mask_seq].contiguous()
    new_embeds = inputs_embeds[:, drop_mask_seq, :].contiguous()
    new_attention_mask = attention_mask[:, drop_mask_seq].contiguous()

    # Handle position_ids: mROPE (3/4, 1, seq) or standard (1, seq)
    if position_ids.ndim == 3:
        # (3, 1, seq) -> select on last dim
        new_position_ids = position_ids[:, :, drop_mask_seq].contiguous()
    elif position_ids.ndim == 2:
        new_position_ids = position_ids[:, drop_mask_seq].contiguous()
    else:
        new_position_ids = position_ids[drop_mask_seq].contiguous()

    return CompressedSequence(
        input_ids=new_input_ids,
        inputs_embeds=new_embeds,
        attention_mask=new_attention_mask,
        position_ids=new_position_ids,
        n_dropped=int(dropped_positions.numel()),
    )
