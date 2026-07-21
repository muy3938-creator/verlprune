from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest


def uniform_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    return torch.linspace(
        0,
        request.token_count - 1,
        steps=request.keep_count,
        device=request.device,
    ).round().long()
