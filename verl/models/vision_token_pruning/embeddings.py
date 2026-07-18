"""Model-facing physical compaction of aligned visual features."""

from __future__ import annotations

import torch

KEEP_MASK_KEY = "vision_token_keep_mask"


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
            f"{name} covers {keep_mask.numel()} visual tokens, "
            f"but model produced {embeddings.shape[0]} features"
        )
    return embeddings[keep_mask.to(embeddings.device)]


def prune_visual_embedding_outputs(
    embeddings: torch.Tensor,
    auxiliary_embeddings: list[torch.Tensor] | None,
    keep_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Prune a primary visual output and any aligned deep-stack outputs."""

    pruned_embeddings = prune_visual_embeddings(embeddings, keep_mask)
    if auxiliary_embeddings is None:
        return pruned_embeddings, None
    return pruned_embeddings, [
        prune_visual_embeddings(auxiliary, keep_mask, name=f"auxiliary visual keep mask #{index}")
        for index, auxiliary in enumerate(auxiliary_embeddings)
    ]
