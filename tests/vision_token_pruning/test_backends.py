import pytest

from verl.models.vision_token_pruning.backends import (
    build_vllm_pruning_launch_options,
    resolve_vllm_backend,
)
from verl.models.vision_token_pruning.config import VisionTokenPruningConfig


def test_physical_backend_is_default_and_supports_qwen3_vl():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.5)

    assert resolve_vllm_backend(config).name == "physical"
    options = build_vllm_pruning_launch_options(
        config,
        model_type="qwen3_vl",
        routing_replay_enabled=False,
    )
    assert options.hf_overrides["architectures"] == ["VerlPrunedQwen3VLForConditionalGeneration"]
    assert options.cli_args["enforce_eager"] is True


def test_layerwise_backend_is_eager_only_and_rejects_unsupported_model():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.5, prune_after_layer=2)

    profile = resolve_vllm_backend(config)
    assert profile.name == "layerwise_flex"
    assert profile.requires_eager is True
    assert profile.supports_chunked_prefill is False
    with pytest.raises(ValueError, match="does not support"):
        build_vllm_pruning_launch_options(
            config,
            model_type="qwen3_vl",
            routing_replay_enabled=False,
        )

    options = build_vllm_pruning_launch_options(
        config,
        model_type="qwen2_5_vl",
        routing_replay_enabled=False,
    )
    assert options.cli_args["attention_config"] == {"backend": "FLEX_ATTENTION"}
    assert options.hf_overrides["architectures"] == [
        "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration"
    ]


def test_compact_flash_reference_backend_does_not_request_flex_attention():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=2,
        layerwise_backend="compact_flash",
    )

    options = build_vllm_pruning_launch_options(
        config,
        model_type="qwen2_5_vl",
        routing_replay_enabled=False,
    )

    assert "attention_config" not in options.cli_args
    assert options.hf_overrides["architectures"] == [
        "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration"
    ]


def test_dynamic_decode_reserves_capture_width():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.05,
        prune_after_layer=15,
        selector="vision_pulse",
        selector_input="decode_query",
        selector_kwargs={"capture_capacity": 32},
    )

    options = build_vllm_pruning_launch_options(
        config,
        model_type="qwen2_5_vl",
        routing_replay_enabled=False,
    )

    assert options.hf_overrides["text_config"]["num_experts_per_tok"] == 32
