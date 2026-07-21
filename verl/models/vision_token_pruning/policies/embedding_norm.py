from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import flat_float, full_arange


def embedding_norm_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return full_arange(request)
    features = flat_float(request.features, name="embedding_norm")
    scores = features.square().sum(dim=-1)
    return scores.topk(request.keep_count, sorted=False).indices.sort().values
