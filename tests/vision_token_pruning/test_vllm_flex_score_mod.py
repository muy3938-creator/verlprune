"""Proof that vLLM's native Flex backend can own paged KV for pruning masks."""

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

if not torch.cuda.is_available():
    pytest.skip("vLLM FlexAttention requires CUDA", allow_module_level=True)

from flash_attn import flash_attn_func  # noqa: E402
from vllm.v1.attention.backends.flex_attention import (  # noqa: E402
    FlexAttentionImpl,
    FlexAttentionMetadata,
    physical_to_logical_mapping,
)


def _install_request_keep_score_mod(metadata, keep_by_request):
    request_lookup = metadata.doc_ids
    assert request_lookup is not None

    def pruning_score_mod(score, _batch, _head, logical_query, logical_key, *, physical_q):
        request = request_lookup[physical_q]
        allowed = (
            keep_by_request[request, logical_query]
            & keep_by_request[request, logical_key]
        )
        return torch.where(allowed, score, -float("inf"))

    metadata.score_mod = pruning_score_mod
    metadata.transformed_score_mod = metadata.get_transformed_score_mod()


def _install_physical_slot_keep_score_mod(metadata, physical_slot_keep):
    def pruning_score_mod(score, _batch, _head, _query, physical_key):
        return torch.where(physical_slot_keep[physical_key], score, -float("inf"))

    metadata.transformed_score_mod = pruning_score_mod


def test_runtime_score_mod_matches_compact_flash_for_multiple_requests():
    """A score modifier masks each request while vLLM owns cache addressing."""

    request_lengths = (4, 5)
    token_count = sum(request_lengths)
    block_size = 16
    block_table = torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
    sequence_lengths = torch.tensor(request_lengths, dtype=torch.int32, device="cuda")
    metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=token_count,
        max_query_len=max(request_lengths),
        query_start_loc=torch.tensor([0, request_lengths[0], token_count], dtype=torch.int32, device="cuda"),
        max_seq_len=max(request_lengths),
        seq_lens=sequence_lengths,
        block_table=block_table,
        slot_mapping=torch.cat(
            [
                torch.arange(block_size, block_size + request_lengths[0], device="cuda"),
                torch.arange(2 * block_size, 2 * block_size + request_lengths[1], device="cuda"),
            ]
        ),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=3 * block_size,
        block_size=block_size,
        max_possible_sequence_length=block_size,
        num_reqs=2,
        physical_to_logical=physical_to_logical_mapping(
            block_table,
            sequence_lengths,
            block_size,
            total_blocks=3,
        ),
        decode_offset=torch.zeros(2, dtype=torch.int32, device="cuda"),
        num_blocks_per_seq=torch.ones(2, dtype=torch.int32, device="cuda"),
    )
    implementation = FlexAttentionImpl(
        num_heads=2,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    torch.manual_seed(11)
    query = torch.randn(token_count, 2, 16, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    kv_cache = torch.zeros(2, 3, block_size, 2, 16, dtype=torch.bfloat16, device="cuda")
    implementation.do_kv_cache_update(layer, key, value, kv_cache, metadata.slot_mapping)

    keep_by_request = torch.tensor(
        [
            [True, False, True, True, False],
            [True, True, False, False, True],
        ],
        device="cuda",
    )
    keep = torch.cat([keep_by_request[0, : request_lengths[0]], keep_by_request[1]])
    _install_request_keep_score_mod(metadata, keep_by_request)
    output = torch.empty_like(query)
    actual = implementation.forward(layer, query, key, value, kv_cache, metadata, output)

    expected_parts = []
    request_start = 0
    for request, request_length in enumerate(request_lengths):
        local_keep = keep_by_request[request, :request_length]
        local_retained = local_keep.nonzero(as_tuple=False).flatten() + request_start
        expected_parts.append(
            flash_attn_func(
                query.index_select(0, local_retained).unsqueeze(0),
                key.index_select(0, local_retained).unsqueeze(0),
                value.index_select(0, local_retained).unsqueeze(0),
                causal=True,
            ).squeeze(0)
        )
        request_start += request_length
    expected = torch.cat(expected_parts)

    assert torch.count_nonzero(actual[~keep]) == 0
    torch.testing.assert_close(actual[keep].float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_prefill_slot_sidecar_is_reused_by_next_decode_query():
    """A physical-slot sidecar carries selection into decode without request state."""

    block_size = 16
    prompt_lengths = (4, 5)
    block_table = torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
    implementation = FlexAttentionImpl(
        num_heads=2,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    torch.manual_seed(29)
    prompt_key = torch.randn(sum(prompt_lengths), 2, 16, dtype=torch.bfloat16, device="cuda")
    prompt_value = torch.randn_like(prompt_key)
    kv_cache = torch.zeros(2, 3, block_size, 2, 16, dtype=torch.bfloat16, device="cuda")
    prompt_slots = torch.cat(
        [
            torch.arange(block_size, block_size + prompt_lengths[0], device="cuda"),
            torch.arange(2 * block_size, 2 * block_size + prompt_lengths[1], device="cuda"),
        ]
    )
    implementation.do_kv_cache_update(layer, prompt_key, prompt_value, kv_cache, prompt_slots)

    decode_query = torch.randn(2, 2, 16, dtype=torch.bfloat16, device="cuda")
    decode_key = torch.randn_like(decode_query)
    decode_value = torch.randn_like(decode_query)
    decode_sequence_lengths = torch.tensor(
        [prompt_lengths[0] + 1, prompt_lengths[1] + 1],
        dtype=torch.int32,
        device="cuda",
    )
    decode_metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        max_seq_len=int(decode_sequence_lengths.max()),
        seq_lens=decode_sequence_lengths,
        block_table=block_table,
        slot_mapping=torch.tensor(
            [block_size + prompt_lengths[0], 2 * block_size + prompt_lengths[1]],
            device="cuda",
        ),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=3 * block_size,
        block_size=block_size,
        max_possible_sequence_length=block_size,
        num_reqs=2,
        physical_to_logical=physical_to_logical_mapping(
            block_table,
            decode_sequence_lengths,
            block_size,
            total_blocks=3,
        ),
        decode_offset=torch.tensor(prompt_lengths, dtype=torch.int32, device="cuda"),
        num_blocks_per_seq=torch.ones(2, dtype=torch.int32, device="cuda"),
    )
    implementation.do_kv_cache_update(
        layer,
        decode_key,
        decode_value,
        kv_cache,
        decode_metadata.slot_mapping,
    )

    # Prompt selection is produced once at the boundary. The generic adapter
    # scatters it to a sidecar using vLLM's existing slot mapping; no cache
    # address is changed and algorithms never see the physical layout.
    keep_by_request = torch.tensor(
        [
            [True, False, True, True, True, False],
            [True, True, False, False, True, True],
        ],
        device="cuda",
    )
    flat_prompt_keep = torch.cat(
        [keep_by_request[0, : prompt_lengths[0]], keep_by_request[1, : prompt_lengths[1]]]
    )
    physical_slot_keep = torch.ones(3 * block_size, dtype=torch.bool, device="cuda")
    physical_slot_keep[prompt_slots] = flat_prompt_keep
    _install_physical_slot_keep_score_mod(decode_metadata, physical_slot_keep)
    actual = implementation.forward(
        layer,
        decode_query,
        decode_key,
        decode_value,
        kv_cache,
        decode_metadata,
        torch.empty_like(decode_query),
    )

    expected_parts = []
    prompt_start = 0
    for request, prompt_length in enumerate(prompt_lengths):
        retained_prompt = keep_by_request[request, :prompt_length].nonzero(as_tuple=False).flatten()
        request_key = torch.cat(
            [
                prompt_key[prompt_start : prompt_start + prompt_length].index_select(0, retained_prompt),
                decode_key[request : request + 1],
            ]
        )
        request_value = torch.cat(
            [
                prompt_value[prompt_start : prompt_start + prompt_length].index_select(0, retained_prompt),
                decode_value[request : request + 1],
            ]
        )
        expected_parts.append(
            F.scaled_dot_product_attention(
                decode_query[request].unsqueeze(0).unsqueeze(2),
                request_key.transpose(0, 1).unsqueeze(0),
                request_value.transpose(0, 1).unsqueeze(0),
            ).squeeze(0).transpose(0, 1)
        )
        prompt_start += prompt_length
    expected = torch.cat(expected_parts)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
