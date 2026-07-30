"""Side-channel Request ID transport for Pre-LLM visual token pruning.

This module provides a thread-safe side-channel registry mapping
vLLM ``request_id`` to selected ``keep_indices``.

This completely eliminates:
1. Embedding column modifications (no radix-256 columns).
2. vLLM ``RoutedExpertsCapturer`` MoE buffer hacks.
3. Bitwise 33-layer capture dependencies.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

_REGISTRY_LOCK = threading.Lock()
_PRUNING_REQUEST_REGISTRY: dict[str, torch.Tensor] = {}


def register_request_selection(request_id: str, keep_indices: torch.Tensor) -> None:
    """Store selection indices for a given vLLM request_id in thread-safe registry."""
    if not request_id:
        return
    with _REGISTRY_LOCK:
        _PRUNING_REQUEST_REGISTRY[request_id] = keep_indices.cpu().detach()


def pop_request_selection(request_id: str) -> torch.Tensor | None:
    """Retrieve and remove selection indices for a given vLLM request_id."""
    if not request_id:
        return None
    with _REGISTRY_LOCK:
        return _PRUNING_REQUEST_REGISTRY.pop(request_id, None)


def clear_request_selection_registry() -> None:
    """Clear all pending selections in the side-channel registry."""
    with _REGISTRY_LOCK:
        _PRUNING_REQUEST_REGISTRY.clear()


def decode_vllm_selection_capture(capture: Any, request_id: str | None = None) -> torch.Tensor | None:
    """Decode kept visual token indices via side-channel registry or legacy capture.

    Args:
        capture: Legacy capture output or request_id string.
        request_id: Optional request_id to look up in the side-channel registry.

    Returns:
        Sorted 1-D :class:`torch.Tensor` of kept token indices (0-based).
    """
    if request_id is not None:
        indices = pop_request_selection(request_id)
        if indices is not None:
            return indices

    if isinstance(capture, str):
        indices = pop_request_selection(capture)
        if indices is not None:
            return indices

    # Fallback to legacy array decoding if provided
    if capture is None:
        return None
    if not hasattr(capture, "tolist") and not isinstance(capture, list):
        return None
    values = capture.tolist() if hasattr(capture, "tolist") else capture

    try:
        encoded = [int(token_layers[0][0]) for token_layers in values]
        kept_indices = sorted({value - 1 for value in encoded if value > 0})
        return torch.tensor(kept_indices, dtype=torch.long)
    except (IndexError, TypeError, ValueError):
        return None
