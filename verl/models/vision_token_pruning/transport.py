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
from .protocol import DynamicVisionTokenSelection, TwoStageVisionTokenSelection, VisionTokenSelection

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
        binary_transport = any(
            len(token_layers) > 1 and int(token_layers[-1][0]) > 0
            for token_layers in values
        )
        if binary_transport:
            if any(len(token_layers) < 33 for token_layers in values):
                raise ValueError("binary vLLM selection capture requires at least 33 layers")
            encoded = [
                sum(int(token_layers[bit][0]) << bit for bit in range(16))
                if int(token_layers[-1][0]) > 0
                else 0
                for token_layers in values
            ]
            original_counts = {
                sum(int(token_layers[16 + bit][0]) << bit for bit in range(16))
                for token_layers in values
                if int(token_layers[-1][0]) > 0
            }
            if len(original_counts) != 1 or next(iter(original_counts)) <= 0:
                raise ValueError("binary vLLM selection capture has inconsistent token counts")
            original_visual_token_count = next(iter(original_counts))
        else:
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


def decode_vllm_dynamic_selection_capture(
    capture: Any,
    *,
    nominal_keep_ratio: float,
    original_visual_token_count: int,
    selector: str,
    selector_kwargs: Mapping[str, Any] | None = None,
) -> DynamicVisionTokenSelection:
    """Decode one padded selected-index vector for every scheduled query."""

    values = capture.tolist() if hasattr(capture, "tolist") else capture
    if values is None:
        raise ValueError("vLLM dynamic visual-token pruning did not return routing metadata")
    query_rows: list[tuple[int, ...]] = []
    try:
        for token_layers in values:
            encoded = [int(value) for value in token_layers[0]]
            selected = tuple(sorted({value - 1 for value in encoded if value > 0}))
            query_rows.append(selected)
    except (IndexError, TypeError) as exc:
        raise ValueError("invalid vLLM dynamic selection capture shape") from exc
    return DynamicVisionTokenSelection(
        nominal_keep_ratio=nominal_keep_ratio,
        selector=selector,
        selector_fingerprint=compute_selector_fingerprint(selector, selector_kwargs),
        original_visual_token_count=original_visual_token_count,
        query_kept_visual_indices=tuple(query_rows),
    )


def decode_vllm_two_stage_selection_capture(
    capture: Any,
    *,
    prefill_keep_ratio: float,
    prefill_selector: str,
    prefill_selector_kwargs: Mapping[str, Any] | None,
    decode_keep_ratio: float,
    decode_selector: str,
    decode_selector_kwargs: Mapping[str, Any] | None,
) -> TwoStageVisionTokenSelection:
    """Decode physical-prefill metadata and dynamic routes from one channel.

    Layer-zero capture rows whose final capacity slot is one describe a
    physically retained visual token: slot zero stores its one-based original
    index and slot one stores the original visual-token count. Other rows are
    ordinary per-query dynamic selections relative to the retained subset.
    """

    values = capture.tolist() if hasattr(capture, "tolist") else capture
    if values is None:
        raise ValueError("vLLM two-stage pruning did not return selection metadata")
    prefill_indices: list[int] = []
    original_counts: set[int] = set()
    query_rows: list[tuple[int, ...]] = []
    try:
        for token_layers in values:
            row = [int(value) for value in token_layers[0]]
            if len(row) < 3:
                raise ValueError("two-stage selection capture requires capacity >= 3")
            if row[-1] > 0:
                if row[0] <= 0 or row[1] <= 0:
                    raise ValueError("invalid two-stage prefill metadata row")
                prefill_indices.append(row[0] - 1)
                original_counts.add(row[1])
                query_rows.append(())
            else:
                query_rows.append(tuple(sorted({value - 1 for value in row[:-1] if value > 0})))
    except (IndexError, TypeError) as exc:
        raise ValueError("invalid vLLM two-stage selection capture shape") from exc
    if len(original_counts) != 1:
        raise ValueError("two-stage capture has missing or inconsistent original token counts")
    original_count = next(iter(original_counts))
    prefill = VisionTokenSelection(
        keep_ratio=prefill_keep_ratio,
        selector=prefill_selector,
        selector_fingerprint=compute_selector_fingerprint(prefill_selector, prefill_selector_kwargs),
        original_visual_token_count=original_count,
        kept_visual_indices=tuple(sorted(set(prefill_indices))),
    )
    decode = DynamicVisionTokenSelection(
        nominal_keep_ratio=decode_keep_ratio,
        selector=decode_selector,
        selector_fingerprint=compute_selector_fingerprint(decode_selector, decode_selector_kwargs),
        original_visual_token_count=len(prefill.kept_visual_indices),
        query_kept_visual_indices=tuple(query_rows),
    )
    return TwoStageVisionTokenSelection(prefill=prefill, decode=decode)
