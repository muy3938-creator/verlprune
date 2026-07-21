"""Shared helpers for built-in policies. Fail fast on missing tensors."""

from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest


def flat_float(features: torch.Tensor | None, *, name: str) -> torch.Tensor:
    if features is None:
        raise ValueError(f"{name} requires features/decoder states")
    if features.ndim < 2:
        raise ValueError(f"{name} states must have a token dimension")
    return features.float().reshape(len(features), -1)


def decoder_text_features(
    request: VisionTokenSelectionRequest,
    *,
    source: str,
    strategy_name: str,
) -> torch.Tensor:
    context = getattr(request, f"context_{source}_states")
    visual_mask = request.visual_context_mask
    features = flat_float(context, name=strategy_name)
    if visual_mask is None or visual_mask.ndim != 1 or len(visual_mask) != len(features):
        raise ValueError(f"{strategy_name} requires a full-context visual mask")
    visual_positions = visual_mask.nonzero(as_tuple=False).flatten()
    if not len(visual_positions):
        raise ValueError(f"{strategy_name} requires visual tokens")
    positions = torch.arange(len(features), device=features.device)
    text_after_image = (~visual_mask) & (positions > visual_positions[-1])
    text = features[text_after_image]
    if not len(text):
        text = features[~visual_mask]
    if not len(text):
        raise ValueError(f"{strategy_name} requires at least one text token")
    return text


def full_arange(request: VisionTokenSelectionRequest) -> torch.Tensor:
    return torch.arange(request.token_count, device=request.device)
