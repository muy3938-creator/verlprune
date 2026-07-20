import torch
import torch.nn.functional as F

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine
from verl.models.vision_token_pruning.transformers_sampler import (
    TransformersPruningState,
    _pruned_flash_varlen,
    _pruned_sdpa,
)


def _state(selector: str = "uniform") -> TransformersPruningState:
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=1,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decoder_key",
        selector=selector,
    )
    return TransformersPruningState(
        config=config,
        visual_mask=torch.tensor([[False, True, True, True, True, False]]),
        engines=[VisionTokenSelectionEngine(config, seed=0)],
        static_indices=[torch.tensor([0, 3])],
    )


def test_static_key_keep_preserves_text_and_selected_visual_tokens():
    state = _state()
    assert state.static_key_keep(8, torch.device("cpu")).tolist() == [
        [True, True, False, False, True, True, True, True]
    ]


def test_dynamic_rows_start_with_unpruned_prompt_queries():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=1,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decode_query",
        selector="vision_pulse",
        selector_kwargs={"budget_mode": "fixed"},
    )
    state = TransformersPruningState(
        config=config,
        visual_mask=torch.tensor([[False, True, True, False]]),
        engines=[VisionTokenSelectionEngine(config, seed=0)],
    )
    assert state.dynamic_rows == [[(), (), (), ()]]


def test_pruned_sdpa_matches_explicit_boolean_mask():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    keep = torch.tensor([[True, False, True]])
    actual = _pruned_sdpa(
        query,
        key,
        value,
        keep,
        scaling=0.5,
        dropout=0.0,
        dynamic_last_query_only=False,
    )
    causal = torch.ones(3, 3, dtype=torch.bool).tril() & keep[:, None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal[:, None],
        scale=0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def _fake_flash_attn_varlen_func(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
):
    assert causal
    assert max_seqlen_q == int(cu_seqlens_q.diff().max())
    assert max_seqlen_k == int(cu_seqlens_k.diff().max())
    outputs = []
    for batch_index in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[batch_index : batch_index + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_index : batch_index + 2].tolist()
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        value = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        query_positions = torch.arange(k_end - k_start - (q_end - q_start), k_end - k_start)
        key_positions = torch.arange(k_end - k_start)
        allowed = key_positions[None, :] <= query_positions[:, None]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[None, None],
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def _explicit_compacted_reference(query, key, value, keep, scale):
    batch_size, heads, query_length, head_dim = query.shape
    output = query.new_zeros((batch_size, query_length, heads, head_dim))
    query_keep = keep[:, -query_length:]
    for batch_index in range(batch_size):
        compact_query = query[batch_index, :, query_keep[batch_index]].unsqueeze(0)
        compact_key = key[batch_index, :, keep[batch_index]].unsqueeze(0)
        compact_value = value[batch_index, :, keep[batch_index]].unsqueeze(0)
        query_positions = torch.arange(
            compact_key.shape[2] - compact_query.shape[2],
            compact_key.shape[2],
        )
        key_positions = torch.arange(compact_key.shape[2])
        allowed = key_positions[None, :] <= query_positions[:, None]
        compact_output = F.scaled_dot_product_attention(
            compact_query,
            compact_key,
            compact_value,
            attn_mask=allowed[None, None],
            scale=scale,
        ).squeeze(0).transpose(0, 1)
        output[batch_index, query_keep[batch_index]] = compact_output
    return output


def test_pruned_flash_varlen_packs_variable_prefill_rows_and_scatters_output():
    torch.manual_seed(1)
    query = torch.randn(2, 2, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    value = torch.randn(2, 2, 6, 4)
    keep = torch.tensor(
        [
            [True, True, False, True, False, True],
            [True, False, True, False, True, True],
        ]
    )

    actual = _pruned_flash_varlen(
        query,
        key,
        value,
        keep,
        attention_mask=None,
        scaling=0.5,
        dropout=0.0,
        flash_attn_varlen_func=_fake_flash_attn_varlen_func,
    )
    expected = _explicit_compacted_reference(query, key, value, keep, 0.5)

    torch.testing.assert_close(actual, expected)
    assert bool((actual[~keep] == 0).all())


def test_pruned_flash_varlen_decode_uses_one_query_and_selected_cached_kv():
    torch.manual_seed(2)
    query = torch.randn(2, 2, 1, 4)
    key = torch.randn(2, 2, 7, 4)
    value = torch.randn(2, 2, 7, 4)
    keep = torch.tensor(
        [
            [True, False, True, False, True, True, True],
            [True, True, False, False, True, True, True],
        ]
    )

    actual = _pruned_flash_varlen(
        query,
        key,
        value,
        keep,
        attention_mask=None,
        scaling=0.5,
        dropout=0.0,
        flash_attn_varlen_func=_fake_flash_attn_varlen_func,
    )
    expected = _explicit_compacted_reference(query, key, value, keep, 0.5)

    torch.testing.assert_close(actual, expected)
