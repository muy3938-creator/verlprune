from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import full_arange


def random_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return full_arange(request)
    indices = torch.randperm(
        request.token_count,
        device=request.device,
        generator=request.generator,
    )[: request.keep_count]
    return indices.sort().values
