from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import decoder_text_features, flat_float, full_arange


def greedy_prune_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return full_arange(request)
    threshold = float(request.options.get("similarity_threshold", 0.9))
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("greedy_prune similarity_threshold must be in [-1, 1]")

    visual = torch.nn.functional.normalize(
        flat_float(request.value_states, name="greedy_prune"),
        dim=-1,
    )
    last_text = torch.nn.functional.normalize(
        decoder_text_features(request, source="value", strategy_name="greedy_prune")[-1],
        dim=0,
    )
    target = request.keep_count
    candidates = visual
    saliency = candidates @ last_text
    order = saliency.argsort(descending=True, stable=True)
    active = torch.ones(len(candidates), dtype=torch.bool, device=request.device)
    selected: list[torch.Tensor] = []
    for pivot in order:
        if not bool(active[pivot]):
            continue
        selected.append(pivot)
        if len(selected) == target:
            break
        redundant = (candidates @ candidates[pivot]) > threshold
        active &= ~redundant

    if len(selected) < target:
        selected_ids = torch.zeros(len(candidates), dtype=torch.bool, device=request.device)
        if selected:
            selected_ids[torch.stack(selected)] = True
        for candidate in order:
            if not bool(selected_ids[candidate]):
                selected.append(candidate)
                selected_ids[candidate] = True
                if len(selected) == target:
                    break

    return torch.stack(selected).sort().values
