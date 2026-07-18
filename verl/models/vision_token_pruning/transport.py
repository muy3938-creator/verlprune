"""Backend transport adapters for exact visual-token selections.

The stable protocol does not know about vLLM. This module contains the one
temporary integration detail: selections travel through vLLM 0.18's routed
expert capture channel and through two low-range embedding metadata columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .config import compute_selector_fingerprint
from .protocol import VisionTokenSelection

EMBEDDING_METADATA_RADIX = 256


def encode_embedding_selection_metadata(indices: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    encoded = indices + 1
    if encoded.numel() and int(encoded.max()) >= EMBEDDING_METADATA_RADIX**2:
        raise ValueError("visual-token selection transport supports fewer than 65536 image tokens")
    return torch.stack(
        [
            encoded.remainder(EMBEDDING_METADATA_RADIX),
            encoded.div(EMBEDDING_METADATA_RADIX, rounding_mode="floor"),
        ],
        dim=1,
    ).to(dtype=dtype)


def decode_embedding_selection_metadata(annotated_embeddings: torch.Tensor) -> torch.Tensor:
    if annotated_embeddings.ndim != 2 or annotated_embeddings.shape[1] < 2:
        raise ValueError("annotated visual embeddings must contain two metadata columns")
    return (
        annotated_embeddings[:, -2].long()
        + annotated_embeddings[:, -1].long() * EMBEDDING_METADATA_RADIX
    )


def decode_vllm_selection_capture(
    capture: Any,
    *,
    keep_ratio: float,
    original_visual_token_count: int,
    selector: str,
    selector_kwargs: Mapping[str, Any] | None = None,
) -> VisionTokenSelection:
    """Decode positive one-based indices from the vLLM compatibility channel."""

    values = capture.tolist() if hasattr(capture, "tolist") else capture
    if values is None:
        raise ValueError("vLLM visual-token pruning plugin did not return selection metadata")
    try:
        encoded = [int(token_layers[0][0]) for token_layers in values]
    except (IndexError, TypeError) as exc:
        raise ValueError("invalid vLLM selection capture shape") from exc
    kept_indices = tuple(sorted({value - 1 for value in encoded if value > 0}))
    return VisionTokenSelection(
        keep_ratio=keep_ratio,
        selector=selector,
        selector_fingerprint=compute_selector_fingerprint(selector, selector_kwargs),
        original_visual_token_count=original_visual_token_count,
        kept_visual_indices=kept_indices,
    )
