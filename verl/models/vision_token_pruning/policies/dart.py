from __future__ import annotations

import torch

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

from ._common import decoder_text_features, flat_float, full_arange


def dart_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.keep_count == request.token_count:
        return full_arange(request)
    visual_keys = flat_float(request.key_states, name="dart")
    visual_values = torch.nn.functional.normalize(
        flat_float(request.value_states, name="dart"),
        dim=-1,
    )
    text_keys = decoder_text_features(request, source="key", strategy_name="dart")
    text_values = torch.nn.functional.normalize(
        decoder_text_features(request, source="value", strategy_name="dart"),
        dim=-1,
    )

    requested_image_pivots = int(request.options.get("pivot_image_tokens", 4))
    requested_text_pivots = int(request.options.get("pivot_text_tokens", 4))
    if requested_image_pivots <= 0 or requested_text_pivots < 0:
        raise ValueError("dart pivot counts must satisfy image > 0 and text >= 0")
    image_pivot_count = min(
        requested_image_pivots,
        max(1, request.keep_count // 2),
        request.token_count,
    )
    image_scores = visual_keys.abs().sum(dim=-1)
    image_pivots = image_scores.topk(image_pivot_count, sorted=True).indices
    text_pivot_count = min(requested_text_pivots, len(text_keys))
    text_pivots = (
        text_keys.abs().sum(dim=-1).topk(text_pivot_count, sorted=True).indices
        if text_pivot_count
        else torch.empty(0, dtype=torch.long, device=request.device)
    )

    selected_mask = torch.zeros(request.token_count, dtype=torch.bool, device=request.device)
    selected_mask[image_pivots] = True
    pivot_vectors = [visual_values[index] for index in image_pivots]
    pivot_vectors.extend(text_values[index] for index in text_pivots)
    if not pivot_vectors:
        pivot_vectors.append(visual_values[image_scores.argmax()])

    step = 0
    while int(selected_mask.sum()) < request.keep_count:
        candidates = (~selected_mask).nonzero(as_tuple=False).flatten()
        pivot = pivot_vectors[step % len(pivot_vectors)]
        similarities = visual_values.index_select(0, candidates) @ pivot
        selected_mask[candidates[similarities.argmin()]] = True
        step += 1
    return selected_mask.nonzero(as_tuple=False).flatten()
