"""GPU lifecycle checks for the layerwise native-Flex vLLM adapter."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

if not torch.cuda.is_available():
    pytest.skip("vLLM FlexAttention requires CUDA", allow_module_level=True)

import vllm.forward_context  # noqa: E402
from vllm.v1.attention.backends.flex_attention import (  # noqa: E402
    FlexAttentionMetadata,
    physical_to_logical_mapping,
)

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine  # noqa: E402
from verl.vllm_plugins.layerwise_flex_vision_token_pruning import (  # noqa: E402
    LayerwiseFlexPruningPlan,
    LayerwisePrunedFlexAttentionImpl,
)


def _metadata(*, query_tokens: int, sequence_length: int, slot_start: int) -> FlexAttentionMetadata:
    block_size = 16
    block_table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    sequence_lengths = torch.tensor([sequence_length], dtype=torch.int32, device="cuda")
    return FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=query_tokens,
        max_query_len=query_tokens,
        query_start_loc=torch.tensor([0, query_tokens], dtype=torch.int32, device="cuda"),
        max_seq_len=sequence_length,
        seq_lens=sequence_lengths,
        block_table=block_table,
        slot_mapping=torch.arange(slot_start, slot_start + query_tokens, device="cuda"),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=2 * block_size,
        block_size=block_size,
        max_possible_sequence_length=block_size,
        num_reqs=1,
        physical_to_logical=physical_to_logical_mapping(
            block_table,
            sequence_lengths,
            block_size,
            total_blocks=2,
        ),
        decode_offset=torch.tensor([sequence_length - query_tokens], dtype=torch.int32, device="cuda"),
        num_blocks_per_seq=torch.ones(1, dtype=torch.int32, device="cuda"),
    )


def _context(layer_name: str, metadata: FlexAttentionMetadata, plan: LayerwiseFlexPruningPlan):
    return vllm.forward_context.ForwardContext(
        no_compile_layers={},
        attn_metadata={layer_name: metadata},
        slot_mapping={layer_name: metadata.slot_mapping},
        virtual_engine=0,
        additional_kwargs={"verl_layerwise_flex_vision_pruning": plan},
    )


def _implementation() -> LayerwisePrunedFlexAttentionImpl:
    return LayerwisePrunedFlexAttentionImpl(
        num_heads=2,
        head_size=16,
        scale=16**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def test_boundary_key_selection_persists_into_decode_physical_slots():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=15,
        selector="key_norm",
        selector_input="decoder_key",
    )
    plan = LayerwiseFlexPruningPlan(
        prune_after_layer=15,
        selector_input="decoder_key",
        # Tokens 1..4 are image tokens; their positive values are the exact
        # one-based indices returned through Vision-OPD's capture protocol.
        candidate_ids=torch.tensor([0, 1, 2, 3, 4, 0], device="cuda"),
        query_keep_mask=None,
        selection_engine=VisionTokenSelectionEngine(config, seed=7),
    )
    prompt_metadata = _metadata(query_tokens=6, sequence_length=6, slot_start=16)
    torch.manual_seed(17)
    query = torch.randn(6, 2, 16, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    # Force visual token 2 to have the strongest non-anchor key norm.
    key[2].fill_(10)

    boundary_name = "model.language_model.layers.15.self_attn.attn"
    boundary_layer = SimpleNamespace(
        layer_name=boundary_name,
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    boundary_impl = _implementation()
    boundary_cache = torch.zeros(2, 2, 16, 2, 16, dtype=torch.bfloat16, device="cuda")
    boundary_impl.do_kv_cache_update(
        boundary_layer,
        key,
        value,
        boundary_cache,
        prompt_metadata.slot_mapping,
    )
    with (
        patch(
            "vllm.forward_context.get_forward_context",
            return_value=_context(boundary_name, prompt_metadata, plan),
        ),
        patch(
            "verl.vllm_plugins.layerwise_flex_vision_token_pruning."
            "RoutedExpertsCapturer.get_instance",
            return_value=None,
        ),
    ):
        boundary_impl.forward(
            boundary_layer,
            query,
            key,
            value,
            boundary_cache,
            prompt_metadata,
            torch.empty_like(query),
        )

    assert plan.selection_engine.selection_count == 1
    assert plan.query_keep_mask is not None
    assert plan.query_keep_mask.tolist() == [True, False, True, False, True, True]

    post_name = "model.language_model.layers.16.self_attn.attn"
    post_layer = SimpleNamespace(
        layer_name=post_name,
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    post_impl = _implementation()
    post_cache = torch.zeros_like(boundary_cache)
    post_impl.do_kv_cache_update(post_layer, key, value, post_cache, prompt_metadata.slot_mapping)
    prompt_output = torch.empty_like(query)
    with patch(
        "vllm.forward_context.get_forward_context",
        return_value=_context(post_name, prompt_metadata, plan),
    ):
        post_impl.forward(
            post_layer,
            query,
            key,
            value,
            post_cache,
            prompt_metadata,
            prompt_output,
        )

    assert post_impl._physical_slot_keep is not None
    assert post_impl._physical_slot_keep[16:22].tolist() == plan.query_keep_mask.tolist()
    assert torch.count_nonzero(prompt_output[~plan.query_keep_mask]) == 0

    decode_metadata = _metadata(query_tokens=1, sequence_length=7, slot_start=22)
    decode_plan = LayerwiseFlexPruningPlan(
        prune_after_layer=15,
        selector_input="decoder_key",
        candidate_ids=torch.zeros(1, dtype=torch.int64, device="cuda"),
        query_keep_mask=None,
        selection_engine=plan.selection_engine,
    )
    decode_query = torch.randn(1, 2, 16, dtype=torch.bfloat16, device="cuda")
    decode_key = torch.randn_like(decode_query)
    decode_value = torch.randn_like(decode_query)
    post_impl.do_kv_cache_update(
        post_layer,
        decode_key,
        decode_value,
        post_cache,
        decode_metadata.slot_mapping,
    )
    with patch(
        "vllm.forward_context.get_forward_context",
        return_value=_context(post_name, decode_metadata, decode_plan),
    ):
        post_impl.forward(
            post_layer,
            decode_query,
            decode_key,
            decode_value,
            post_cache,
            decode_metadata,
            torch.empty_like(decode_query),
        )

    assert post_impl._physical_slot_keep[16:22].tolist() == plan.query_keep_mask.tolist()
    assert bool(post_impl._physical_slot_keep[22])
    assert plan.selection_engine.selection_count == 1


def test_each_decode_query_reselects_visual_kv_at_the_anchor_layer():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.25,
        prune_after_layer=15,
        selector="vision_pulse",
        selector_input="decode_query",
        selector_kwargs={
            "budget_mode": "fixed",
            "temperature": 0.1,
            "capture_capacity": 4,
        },
    )
    engine = VisionTokenSelectionEngine(config, seed=7)
    prompt_plan = LayerwiseFlexPruningPlan(
        prune_after_layer=15,
        selector_input="decode_query",
        candidate_ids=torch.tensor([0, 1, 2, 3, 4, 0], device="cuda"),
        query_keep_mask=None,
        selection_engine=engine,
    )
    prompt_metadata = _metadata(query_tokens=6, sequence_length=6, slot_start=16)
    prompt_query = torch.zeros(6, 2, 16, dtype=torch.bfloat16, device="cuda")
    prompt_key = torch.zeros_like(prompt_query)
    prompt_value = torch.randn_like(prompt_query)
    prompt_query[-1, :, 0] = 1
    prompt_key[1, :, 0] = 10  # visual index 0
    prompt_key[2, :, 1] = 10  # visual index 1

    boundary_name = "model.language_model.layers.15.self_attn.attn"
    boundary_layer = SimpleNamespace(
        layer_name=boundary_name,
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    boundary_impl = _implementation()
    boundary_cache = torch.zeros(2, 2, 16, 2, 16, dtype=torch.bfloat16, device="cuda")
    with (
        patch(
            "vllm.forward_context.get_forward_context",
            return_value=_context(boundary_name, prompt_metadata, prompt_plan),
        ),
        patch(
            "verl.vllm_plugins.layerwise_flex_vision_token_pruning."
            "RoutedExpertsCapturer.get_instance",
            return_value=None,
        ),
    ):
        boundary_impl.forward(
            boundary_layer,
            prompt_query,
            prompt_key,
            prompt_value,
            boundary_cache,
            prompt_metadata,
            torch.empty_like(prompt_query),
        )

    assert prompt_plan.dynamic_query_active.tolist() == [False, False, False, False, False, True]
    assert prompt_plan.capture_values[-1].tolist() == [1, 0, 0, 0]
    assert prompt_plan.dynamic_request_slot_keep[0, 17:21].tolist() == [True, False, False, False]

    post_name = "model.language_model.layers.16.self_attn.attn"
    post_layer = SimpleNamespace(
        layer_name=post_name,
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    post_impl = _implementation()
    post_cache = torch.zeros_like(boundary_cache)
    with patch(
        "vllm.forward_context.get_forward_context",
        return_value=_context(post_name, prompt_metadata, prompt_plan),
    ):
        post_impl.forward(
            post_layer,
            prompt_query,
            prompt_key,
            prompt_value,
            post_cache,
            prompt_metadata,
            torch.empty_like(prompt_query),
        )
    prompt_score_mod = prompt_metadata.transformed_score_mod
    assert prompt_score_mod is not None
    zero = torch.tensor(0.0, device="cuda")
    assert not torch.isneginf(
        prompt_score_mod(
            zero,
            zero,
            zero,
            torch.tensor(0, device="cuda"),
            torch.tensor(18, device="cuda"),
        )
    )
    assert torch.isneginf(
        prompt_score_mod(
            zero,
            zero,
            zero,
            torch.tensor(5, device="cuda"),
            torch.tensor(18, device="cuda"),
        )
    )

    decode_metadata = _metadata(query_tokens=1, sequence_length=7, slot_start=22)
    decode_plan = LayerwiseFlexPruningPlan(
        prune_after_layer=15,
        selector_input="decode_query",
        candidate_ids=torch.zeros(1, dtype=torch.int64, device="cuda"),
        query_keep_mask=None,
        selection_engine=engine,
    )
    decode_query = torch.zeros(1, 2, 16, dtype=torch.bfloat16, device="cuda")
    decode_query[0, :, 1] = 1
    decode_key = torch.zeros_like(decode_query)
    decode_value = torch.randn_like(decode_query)
    with (
        patch(
            "vllm.forward_context.get_forward_context",
            return_value=_context(boundary_name, decode_metadata, decode_plan),
        ),
        patch(
            "verl.vllm_plugins.layerwise_flex_vision_token_pruning."
            "RoutedExpertsCapturer.get_instance",
            return_value=None,
        ),
    ):
        boundary_impl.forward(
            boundary_layer,
            decode_query,
            decode_key,
            decode_value,
            boundary_cache,
            decode_metadata,
            torch.empty_like(decode_query),
        )

    assert decode_plan.dynamic_query_active.tolist() == [True]
    assert decode_plan.capture_values[0].tolist() == [2, 0, 0, 0]
    assert decode_plan.dynamic_request_slot_keep[0, 17:21].tolist() == [False, True, False, False]

    with patch(
        "vllm.forward_context.get_forward_context",
        return_value=_context(post_name, decode_metadata, decode_plan),
    ):
        post_impl.forward(
            post_layer,
            decode_query,
            decode_key,
            decode_value,
            post_cache,
            decode_metadata,
            torch.empty_like(decode_query),
        )
    decode_score_mod = decode_metadata.transformed_score_mod
    assert decode_score_mod is not None
    query_zero = torch.tensor(0, device="cuda")
    assert torch.isneginf(
        decode_score_mod(zero, zero, zero, query_zero, torch.tensor(17, device="cuda"))
    )
    assert not torch.isneginf(
        decode_score_mod(zero, zero, zero, query_zero, torch.tensor(18, device="cuda"))
    )
