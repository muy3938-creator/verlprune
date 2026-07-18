"""Minimal external strategies for the visual-token pruning experiment API."""

from __future__ import annotations

import torch

from verl.models.vision_token_pruning.strategy import VisionTokenSelectionRequest


def feature_norm(request: VisionTokenSelectionRequest) -> torch.Tensor:
    """Keep high-norm visual features plus the required final MRoPE anchor.

    ``request.options["channel_start"]`` can exclude leading feature channels
    from the score without requiring any actor, trainer, or vLLM plugin edits.
    """

    if request.features is None:
        raise ValueError("feature_norm requires rollout-side visual features")
    if request.keep_count == 1:
        return torch.tensor([request.token_count - 1], device=request.device)

    channel_start = int(request.options.get("channel_start", 0))
    features = request.features[:-1, channel_start:].float()
    if features.shape[1] == 0:
        raise ValueError("channel_start must leave at least one feature channel")
    scores = features.square().sum(dim=1)
    selected = scores.topk(request.keep_count - 1).indices.sort().values
    return torch.cat([selected, selected.new_tensor([request.token_count - 1])])
