"""Observe-side helpers: boundary QKV selection for boundary_once stages.

These pure helpers keep algorithm invocation out of the FlexAttention forward
body so new observe sources can be added without rewriting score_mod logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

if TYPE_CHECKING:
    from verl.vllm_plugins.flex.plan import LayerwiseFlexPruningPlan


def select_from_decoder_states(
    plan: LayerwiseFlexPruningPlan,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: FlexAttentionMetadata,
    *,
    layer_index: int,
) -> None:
    """Run the pure-PyTorch selector once and retain its packed-query mask."""

    if plan.query_keep_mask is not None or not bool((plan.candidate_ids > 0).any()):
        return
    actual_tokens = metadata.num_actual_tokens
    candidate_ids = plan.candidate_ids[:actual_tokens].to(key.device)
    keep = torch.ones(actual_tokens, dtype=torch.bool, device=key.device)
    capture = torch.zeros((actual_tokens, 1), dtype=torch.int32, device=key.device)

    for request_index in range(len(metadata.query_start_loc) - 1):
        start = int(metadata.query_start_loc[request_index].item())
        end = int(metadata.query_start_loc[request_index + 1].item())
        image_positions = (candidate_ids[start:end] > 0).nonzero(as_tuple=False).flatten() + start
        if not len(image_positions):
            continue
        selected_relative = plan.selection_engine.select_decoder_states(
            query_states=query[:actual_tokens].index_select(0, image_positions),
            key_states=key[:actual_tokens].index_select(0, image_positions),
            value_states=value[:actual_tokens].index_select(0, image_positions),
            context_query_states=query[start:end],
            context_key_states=key[start:end],
            context_value_states=value[start:end],
            visual_context_mask=candidate_ids[start:end] > 0,
            layer_index=layer_index,
            keep_ratio=plan.keep_ratio_override,
        )
        selected_positions = image_positions.index_select(0, selected_relative)
        keep[image_positions] = False
        keep[selected_positions] = True
        capture[selected_positions, 0] = candidate_ids[selected_positions].to(torch.int32)

    plan.query_keep_mask = keep
    plan.capture_values = capture
    capturer = RoutedExpertsCapturer.get_instance()
    if capturer is not None:
        capturer.capture(0, capture)
