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
    assert profile.requires_eager is True
    assert profile.supports_chunked_prefill is False
    with pytest.raises(ValueError, match="does not support"):
        build_vllm_pruning_launch_options(
            config,
            model_type="qwen3_vl",
            routing_replay_enabled=False,
        )
