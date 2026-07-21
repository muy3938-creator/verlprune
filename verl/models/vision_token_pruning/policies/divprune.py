from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import flat_float, full_arange


def divprune_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return full_arange(request)
    source = request.value_states if request.value_states is not None else request.features
    features = flat_float(source, name="divprune")
    candidates = torch.nn.functional.normalize(features, dim=-1)
    target = request.keep_count
    if target == len(candidates):
        selected = torch.arange(target, device=request.device)
    else:
        distances = 1.0 - candidates @ candidates.T
        if len(candidates) == 1:
            selected = torch.zeros(1, dtype=torch.long, device=request.device)
        else:
            nearest_other = distances.topk(2, dim=0, largest=False).values[1]
            first = nearest_other.argmax()
            chosen = [first]
            chosen_mask = torch.zeros(len(candidates), dtype=torch.bool, device=request.device)
            chosen_mask[first] = True
            while len(chosen) < target:
                minimum_distance = distances.index_select(0, torch.stack(chosen)).min(dim=0).values
                minimum_distance.masked_fill_(chosen_mask, -float("inf"))
                next_index = minimum_distance.argmax()
                chosen.append(next_index)
                chosen_mask[next_index] = True
            selected = torch.stack(chosen)
    return selected.sort().values
