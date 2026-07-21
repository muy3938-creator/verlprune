from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import full_arange


def key_norm_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.key_states is None:
        raise ValueError("key_norm requires decoder key states")
    if request.keep_count == request.token_count:
        return full_arange(request)
    scores = request.key_states.float().square().sum(dim=tuple(range(1, request.key_states.ndim)))
    return scores.topk(request.keep_count, sorted=False).indices.sort().values
