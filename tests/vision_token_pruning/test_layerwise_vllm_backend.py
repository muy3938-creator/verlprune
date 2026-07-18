import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm.forward_context  # noqa: E402
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata  # noqa: E402

from verl.vllm_plugins.layerwise_vision_token_pruning import (  # noqa: E402
    LayerwisePrunedFlashAttentionImpl,
    LayerwisePruningPlan,
    _compact_metadata,
    _logical_slot_mapping,
    layer_index_from_name,
)


def make_metadata(token_count: int) -> FlashAttentionMetadata:
    return FlashAttentionMetadata(
        num_actual_tokens=token_count,
        max_query_len=token_count,
        query_start_loc=torch.tensor([0, token_count], dtype=torch.int32),
        max_seq_len=token_count,
        seq_lens=torch.tensor([token_count], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.arange(token_count, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=0,
        dcp_context_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
    )


def make_context(layer_name: str, metadata: FlashAttentionMetadata, plan: LayerwisePruningPlan):
    return vllm.forward_context.ForwardContext(
        no_compile_layers={},
        attn_metadata={layer_name: metadata},
        slot_mapping={layer_name: metadata.slot_mapping},
        virtual_engine=0,
        additional_kwargs={"verl_layerwise_vision_pruning": plan},
    )


def make_impl(num_heads=2, head_dim=4):
    return LayerwisePrunedFlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0,
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def test_layer_name_parser_uses_decoder_layer_index():
    assert layer_index_from_name("model.language_model.layers.17.self_attn.attn") == 17


def test_logical_slots_follow_paged_block_table():
    metadata = SimpleNamespace(block_table=torch.tensor([[9, 3], [7, 5]], dtype=torch.int32))

    slots = _logical_slot_mapping(
        metadata,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 15, 16, 18]),
        block_size=16,
    )

    assert slots.tolist() == [144, 159, 80, 82]


def test_compact_metadata_preserves_flash_attention_int32_cu_seqlens():
    metadata = make_metadata(6)

    compact = _compact_metadata(
        metadata,
        compact_query_lengths=torch.tensor([4], dtype=torch.int32),
        compact_sequence_lengths=torch.tensor([4], dtype=torch.int32),
    )

    assert compact.query_start_loc.dtype == torch.int32
    assert compact.query_start_loc.tolist() == [0, 4]


def test_prefill_cache_update_writes_only_retained_tokens_to_compact_slots():
    layer_name = "model.language_model.layers.2.self_attn.attn"
    layer = MagicMock(layer_name=layer_name, _k_scale=torch.tensor(1.0), _v_scale=torch.tensor(1.0))
    metadata = make_metadata(6)
    plan = LayerwisePruningPlan(
        prune_after_layer=1,
        query_keep_mask=torch.tensor([True, True, False, False, True, True]),
        pruned_token_counts=torch.tensor([2], dtype=torch.int32),
    )
    context = make_context(layer_name, metadata, plan)
    key = torch.arange(48, dtype=torch.float32).reshape(6, 2, 4)
    value = key + 100
    kv_cache = torch.zeros(2, 2, 16, 2, 4)

    with (
        patch("vllm.forward_context.get_forward_context", return_value=context),
        patch("verl.vllm_plugins.layerwise_vision_token_pruning.reshape_and_cache_flash") as cache_op,
    ):
        make_impl().do_kv_cache_update(layer, key, value, kv_cache, metadata.slot_mapping)

    assert torch.equal(cache_op.call_args.args[0], key[[0, 1, 4, 5]])
    assert torch.equal(cache_op.call_args.args[1], value[[0, 1, 4, 5]])
    assert cache_op.call_args.args[4].tolist() == [0, 1, 2, 3]


def test_prefill_forward_gathers_queries_and_scatters_retained_outputs():
    layer_name = "model.language_model.layers.2.self_attn.attn"
    layer = MagicMock(layer_name=layer_name)
    metadata = make_metadata(6)
    keep_mask = torch.tensor([True, True, False, False, True, True])
    plan = LayerwisePruningPlan(
        prune_after_layer=1,
        query_keep_mask=keep_mask,
        pruned_token_counts=torch.tensor([2], dtype=torch.int32),
    )
    context = make_context(layer_name, metadata, plan)
    query = torch.arange(48, dtype=torch.float32).reshape(6, 2, 4)
    key = query + 100
    value = query + 200
    output = torch.empty_like(query)
    kv_cache = torch.zeros(2, 2, 16, 2, 4)

    def fake_flash_forward(
        _self,
        _layer,
        compact_query,
        _key,
        _value,
        _cache,
        compact_metadata,
        compact_output,
        *_args,
    ):
        assert compact_metadata.num_actual_tokens == 4
        assert compact_metadata.seq_lens.tolist() == [4]
        compact_output.copy_(compact_query)
        return compact_output

    with (
        patch("vllm.forward_context.get_forward_context", return_value=context),
        patch.object(FlashAttentionImpl, "forward", autospec=True, side_effect=fake_flash_forward),
        patch(
            "verl.vllm_plugins.layerwise_vision_token_pruning.RoutedExpertsCapturer.get_instance",
            return_value=None,
        ),
    ):
        result = make_impl().forward(layer, query, key, value, kv_cache, metadata, output)

    assert torch.equal(result[keep_mask], query[keep_mask])
    assert torch.count_nonzero(result[~keep_mask]) == 0


def test_multi_request_prefill_compacts_each_sequence_independently():
    layer_name = "model.language_model.layers.5.self_attn.attn"
    layer = MagicMock(layer_name=layer_name)
    metadata = dataclasses.replace(
        make_metadata(9),
        max_query_len=5,
        query_start_loc=torch.tensor([0, 4, 9], dtype=torch.int32),
        max_seq_len=5,
        seq_lens=torch.tensor([4, 5], dtype=torch.int32),
        block_table=torch.tensor([[8, 9], [3, 4]], dtype=torch.int32),
    )
    keep_mask = torch.tensor([True, False, True, True, True, False, False, True, True])
    plan = LayerwisePruningPlan(
        prune_after_layer=3,
        query_keep_mask=keep_mask,
        pruned_token_counts=torch.tensor([1, 2], dtype=torch.int32),
    )
    context = make_context(layer_name, metadata, plan)
    query = torch.arange(72, dtype=torch.float32).reshape(9, 2, 4)
    output = torch.empty_like(query)

    def fake_flash_forward(
        _self,
        _layer,
        compact_query,
        _key,
        _value,
        _cache,
        compact_metadata,
        compact_output,
        *_args,
    ):
        assert compact_metadata.query_start_loc.tolist() == [0, 3, 6]
        assert compact_metadata.seq_lens.tolist() == [3, 3]
        compact_output.copy_(compact_query)
        return compact_output

    with (
        patch("vllm.forward_context.get_forward_context", return_value=context),
        patch.object(FlashAttentionImpl, "forward", autospec=True, side_effect=fake_flash_forward),
        patch(
            "verl.vllm_plugins.layerwise_vision_token_pruning.RoutedExpertsCapturer.get_instance",
            return_value=None,
        ),
    ):
        result = make_impl().forward(
            layer,
            query,
            query,
            query,
            torch.zeros(2, 10, 16, 2, 4),
            metadata,
            output,
        )

    assert torch.equal(result[keep_mask], query[keep_mask])
    assert torch.count_nonzero(result[~keep_mask]) == 0
