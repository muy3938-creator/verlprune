"""Instance-local visual KV pruning for Transformers generation experiments.

This adapter intentionally uses the public Hugging Face model and cache.  It
does not patch vLLM or global Transformers classes.  Layers through the anchor
run with the model's configured FlashAttention backend; later layers gather
the retained Q/K/V rows and use packed varlen FlashAttention by default.  The
older SDPA mask remains an explicit numerical/debug reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from .config import VisionTokenPruningConfig
from .dynamic import select_dynamic_visual_kv
from .protocol import DynamicVisionTokenSelection, VisionTokenSelection
from .strategy import VisionTokenSelectionEngine


@dataclass
class TransformersPruningState:
    config: VisionTokenPruningConfig
    visual_mask: torch.Tensor
    engines: list[VisionTokenSelectionEngine]
    static_indices: list[torch.Tensor] | None = None
    current_dynamic_indices: list[torch.Tensor] | None = None
    dynamic_rows: list[list[tuple[int, ...]]] = field(default_factory=list)
    post_pruning_backend: str = "sdpa"

    def __post_init__(self) -> None:
        if self.visual_mask.ndim != 2 or self.visual_mask.dtype != torch.bool:
            raise ValueError("visual_mask must have shape [batch, prompt_length] and bool dtype")
        if len(self.engines) != len(self.visual_mask):
            raise ValueError("one selection engine is required per batch element")
        if self.post_pruning_backend not in {"sdpa", "flash_varlen"}:
            raise ValueError("post_pruning_backend must be 'sdpa' or 'flash_varlen'")
        if not self.dynamic_rows:
            self.dynamic_rows = [
                [tuple() for _ in range(self.prompt_length)]
                for _ in range(len(self.visual_mask))
            ]

    @property
    def prompt_length(self) -> int:
        return int(self.visual_mask.shape[1])

    def context_visual_mask(self, key_length: int, device: torch.device) -> torch.Tensor:
        if key_length < self.prompt_length:
            raise ValueError("KV context is shorter than the pruning prompt")
        suffix = torch.zeros(
            (len(self.visual_mask), key_length - self.prompt_length),
            dtype=torch.bool,
            device=device,
        )
        return torch.cat((self.visual_mask.to(device), suffix), dim=1)

    def static_key_keep(self, key_length: int, device: torch.device) -> torch.Tensor:
        if self.static_indices is None:
            raise RuntimeError("static selection has not been computed at the anchor layer")
        visual = self.context_visual_mask(key_length, device)
        keep = ~visual
        for batch_index, indices in enumerate(self.static_indices):
            positions = visual[batch_index].nonzero(as_tuple=False).flatten()
            keep[batch_index, positions[indices.to(device)]] = True
        return keep

    def dynamic_key_keep(self, key_length: int, device: torch.device) -> torch.Tensor:
        if self.current_dynamic_indices is None:
            raise RuntimeError("dynamic selection has not been computed at the anchor layer")
        visual = self.context_visual_mask(key_length, device)
        keep = ~visual
        for batch_index, indices in enumerate(self.current_dynamic_indices):
            positions = visual[batch_index].nonzero(as_tuple=False).flatten()
            keep[batch_index, positions[indices.to(device)]] = True
        return keep

    def selections(self) -> list[VisionTokenSelection | DynamicVisionTokenSelection]:
        output: list[VisionTokenSelection | DynamicVisionTokenSelection] = []
        visual_count = int(self.visual_mask[0].sum().item())
        if self.config.uses_dynamic_decode_selection:
            for rows in self.dynamic_rows:
                output.append(
                    DynamicVisionTokenSelection(
                        nominal_keep_ratio=self.config.keep_ratio,
                        original_visual_token_count=visual_count,
                        query_kept_visual_indices=tuple(rows),
                        selector=self.config.selector,
                        selector_fingerprint=self.config.selector_fingerprint,
                    )
                )
            return output
        if self.static_indices is None:
            raise RuntimeError("generation ended before static selection was computed")
        for indices in self.static_indices:
            output.append(
                VisionTokenSelection(
                    keep_ratio=self.config.keep_ratio,
                    original_visual_token_count=visual_count,
                    kept_visual_indices=tuple(int(index) for index in indices.cpu().tolist()),
                    selector=self.config.selector,
                    selector_fingerprint=self.config.selector_fingerprint,
                )
            )
        return output


def _select_static(
    state: TransformersPruningState,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_index: int,
) -> None:
    visual_masks = state.context_visual_mask(key_states.shape[2], key_states.device)
    selected = []
    for batch_index, engine in enumerate(state.engines):
        visual = visual_masks[batch_index]
        selected.append(
            engine.select_decoder_states(
                query_states=query_states[batch_index, :, visual, :].transpose(0, 1),
                key_states=key_states[batch_index, :, visual, :].transpose(0, 1),
                value_states=value_states[batch_index, :, visual, :].transpose(0, 1),
                context_query_states=query_states[batch_index].transpose(0, 1),
                context_key_states=key_states[batch_index].transpose(0, 1),
                context_value_states=value_states[batch_index].transpose(0, 1),
                visual_context_mask=visual,
                layer_index=layer_index,
            )
        )
    state.static_indices = selected


def _select_dynamic(
    state: TransformersPruningState,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
) -> None:
    options = state.config.selector_kwargs
    visual_masks = state.context_visual_mask(key_states.shape[2], key_states.device)
    selected = []
    for batch_index in range(len(query_states)):
        result = select_dynamic_visual_kv(
            query_states[batch_index, :, -1, :],
            key_states[batch_index].transpose(0, 1),
            visual_masks[batch_index],
            softmax_scale=scaling,
            temperature=float(options.get("temperature", 0.1)),
            budget_mode=str(options.get("budget_mode", "fixed")),
            fixed_keep_ratio=state.config.keep_ratio,
            top_p=float(options.get("top_p", 0.95)),
            min_keep_ratio=float(options.get("min_keep_ratio", 0.0)),
            max_keep_ratio=float(options.get("max_keep_ratio", 1.0)),
        )
        indices = result.kept_visual_indices
        selected.append(indices)
        state.dynamic_rows[batch_index].append(tuple(int(index) for index in indices.cpu().tolist()))
    state.current_dynamic_indices = selected


def _pruned_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_keep: torch.Tensor,
    *,
    scaling: float,
    dropout: float,
    dynamic_last_query_only: bool,
) -> torch.Tensor:
    query_length = query_states.shape[2]
    key_length = key_states.shape[2]
    query_positions = torch.arange(
        key_length - query_length,
        key_length,
        device=query_states.device,
    )
    key_positions = torch.arange(key_length, device=query_states.device)
    allowed = key_positions[None, None, :] <= query_positions[None, :, None]
    allowed = allowed.expand(len(query_states), -1, -1).clone()
    if dynamic_last_query_only and query_length > 1:
        allowed[:, -1, :] &= key_keep
    else:
        allowed &= key_keep[:, None, :]
    output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=allowed[:, None, :, :],
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )
    return output.transpose(1, 2).contiguous()


def _packed_token_indices(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return flat packed indices, int32 cu-seqlens, and maximum row length."""

    if keep.ndim != 2 or keep.dtype != torch.bool:
        raise ValueError("packed FlashAttention keep mask must be rank-2 bool")
    lengths = keep.sum(dim=1, dtype=torch.int32)
    if not bool((lengths > 0).all()):
        raise ValueError("every packed FlashAttention request must retain at least one token")
    indices = keep.reshape(-1).nonzero(as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, int(lengths.max().item())


def _pruned_flash_varlen(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_keep: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
    flash_attn_varlen_func,
) -> torch.Tensor:
    """Run arbitrary retained KV rows through packed varlen FlashAttention.

    The full Transformers DynamicCache remains intact.  Only the current
    attention inputs are gathered, so static and per-query dynamic selections
    can share the same implementation without changing cache bookkeeping.
    """

    batch_size, query_heads, query_length, head_dim = query_states.shape
    if key_states.ndim != 4 or value_states.shape != key_states.shape:
        raise ValueError("key/value states must have matching rank-4 shapes")
    if key_keep.shape != (batch_size, key_states.shape[2]):
        raise ValueError("key_keep must match the batch and KV sequence dimensions")
    key_keep = key_keep.to(device=key_states.device, dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != key_keep.shape:
            raise ValueError("varlen FlashAttention expects a rank-2 padding mask matching KV")
        key_keep = key_keep & attention_mask.to(device=key_keep.device, dtype=torch.bool)

    # Generation queries are the final query_length positions in their KV
    # context.  Static prefill therefore drops the corresponding visual query
    # rows, while decode query_length=1 always retains the newly generated row.
    query_keep = key_keep[:, -query_length:]
    query_indices, cu_seqlens_q, max_seqlen_q = _packed_token_indices(query_keep)
    key_indices, cu_seqlens_k, max_seqlen_k = _packed_token_indices(key_keep)

    query_token_major = query_states.transpose(1, 2).contiguous()
    key_token_major = key_states.transpose(1, 2).contiguous()
    value_token_major = value_states.transpose(1, 2).contiguous()
    packed_query = query_token_major.view(-1, query_heads, head_dim).index_select(
        0, query_indices
    )
    packed_key = key_token_major.view(
        -1,
        key_states.shape[1],
        head_dim,
    ).index_select(0, key_indices)
    packed_value = value_token_major.view(
        -1,
        value_states.shape[1],
        head_dim,
    ).index_select(0, key_indices)
    packed_output = flash_attn_varlen_func(
        q=packed_query,
        k=packed_key,
        v=packed_value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    output = packed_output.new_zeros(
        (batch_size * query_length, query_heads, head_dim)
    )
    output.index_copy_(0, query_indices, packed_output)
    return output.view(batch_size, query_length, query_heads, head_dim)


def _make_attention_forward(
    state: TransformersPruningState,
    flash_attention,
    flash_attn_varlen_func=None,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
    )

    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        del output_attentions, use_cache
        batch_size, query_length, _ = hidden_states.shape
        query_states = module.q_proj(hidden_states).view(
            batch_size, query_length, -1, module.head_dim
        ).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(
            batch_size, query_length, -1, module.head_dim
        ).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(
            batch_size, query_length, -1, module.head_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("Qwen2.5-VL generation requires position embeddings")
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            getattr(module, "rope_scaling", module.config.rope_scaling)["mrope_section"],
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                module.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        at_anchor = module.layer_idx == state.config.prune_after_layer
        after_anchor = module.layer_idx > state.config.prune_after_layer
        dynamic = state.config.uses_dynamic_decode_selection
        if at_anchor:
            if dynamic:
                if query_length == 1:
                    _select_dynamic(state, query_states, key_states, module.scaling)
                else:
                    # The prefill predicts the first response token without
                    # decode-time routing. Selection begins when that token is
                    # fed back through the anchor layer.
                    state.current_dynamic_indices = None
            elif state.static_indices is None:
                _select_static(state, query_states, key_states, value_states, module.layer_idx)

        dropout = 0.0 if not module.training else module.attention_dropout
        if after_anchor and (not dynamic or state.current_dynamic_indices is not None):
            key_keep = (
                state.dynamic_key_keep(key_states.shape[2], key_states.device)
                if dynamic
                else state.static_key_keep(key_states.shape[2], key_states.device)
            )
            if state.post_pruning_backend == "flash_varlen":
                if flash_attn_varlen_func is None:
                    raise RuntimeError("flash_varlen backend was installed without FlashAttention")
                attn_output = _pruned_flash_varlen(
                    query_states,
                    key_states,
                    value_states,
                    key_keep,
                    attention_mask=attention_mask,
                    scaling=module.scaling,
                    dropout=dropout,
                    flash_attn_varlen_func=flash_attn_varlen_func,
                )
            else:
                attn_output = _pruned_sdpa(
                    query_states,
                    key_states,
                    value_states,
                    key_keep,
                    scaling=module.scaling,
                    dropout=dropout,
                    dynamic_last_query_only=dynamic,
                )
            attn_weights = None
        else:
            attn_output, attn_weights = flash_attention(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                position_ids=position_ids,
                **kwargs,
            )
        attn_output = attn_output.reshape(batch_size, query_length, -1).contiguous()
        return module.o_proj(attn_output), attn_weights

    return forward


def install_transformers_pruning(
    model,
    *,
    input_ids: torch.Tensor,
    image_token_id: int,
    config: VisionTokenPruningConfig,
    seed: int = 0,
    post_pruning_backend: str = "flash_varlen",
) -> TransformersPruningState:
    """Install pruning on one Qwen2.5-VL model instance and return its state."""

    if not config.enabled or not config.uses_layerwise_backend:
        raise ValueError("Transformers sampler requires enabled layerwise pruning")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, prompt_length]")
    visual_mask = input_ids.eq(image_token_id).cpu()
    visual_counts = visual_mask.sum(dim=1)
    if not bool((visual_counts > 0).all()) or not bool((visual_counts == visual_counts[0]).all()):
        raise ValueError("every prompt must contain the same positive number of visual tokens")

    state = TransformersPruningState(
        config=config,
        visual_mask=visual_mask,
        engines=[VisionTokenSelectionEngine(config, seed=seed + index) for index in range(len(input_ids))],
        post_pruning_backend=post_pruning_backend,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    flash_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    flash_varlen = None
    if post_pruning_backend == "flash_varlen":
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as exc:
            raise ImportError(
                "post_pruning_backend='flash_varlen' requires flash-attn"
            ) from exc
        flash_varlen = flash_attn_varlen_func
    forward = _make_attention_forward(state, flash_attention, flash_varlen)
    layers = model.model.language_model.layers
    if config.prune_after_layer >= len(layers) - 1:
        raise ValueError("prune_after_layer must leave at least one pruned layer")
    for layer in layers:
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
    return state
