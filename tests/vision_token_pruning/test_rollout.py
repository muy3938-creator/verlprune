import pytest

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.rollout import VisionTokenPruningRollout


def make_rollout(*, enabled=True, prune_after_layer=-1, selector="random", selector_kwargs=None):
    return VisionTokenPruningRollout(
        VisionTokenPruningConfig(
            enabled=enabled,
            keep_ratio=0.5,
            prune_after_layer=prune_after_layer,
            selector=selector,
            selector_kwargs=selector_kwargs or {},
        ),
        model_type="qwen2_5_vl",
        image_token_id=99,
    )


def test_rollout_policy_owns_vllm_launch_overrides():
    options = make_rollout().build_launch_options(routing_replay_enabled=False)

    assert options.hf_overrides["architectures"] == ["VerlPrunedQwen2_5VLForConditionalGeneration"]
    assert options.hf_overrides["vision_token_pruning"] == {
        "keep_ratio": 0.5,
        "selector": "random",
        "selector_kwargs": {},
    }
    assert options.cli_args["video_pruning_rate"] == 0.5
    assert options.cli_args["enable_return_routed_experts"] is True


def test_rollout_policy_validates_and_decodes_one_image_request():
    rollout = make_rollout()
    original_count = rollout.inspect_request(
        prompt_ids=[1, 99, 99, 99, 99, 2],
        image_data=[object()],
        video_data=None,
    )
    selection = rollout.decode_selection(
        [[[1]], [[4]]],
        original_token_count=original_count,
    )

    assert selection["original_visual_token_count"] == 4
    assert selection["kept_visual_indices"] == [0, 3]
    assert selection["selector"] == "random"


def test_rollout_policy_rejects_replay_channel_conflict():
    with pytest.raises(ValueError, match="cannot share routed_experts"):
        make_rollout().build_launch_options(routing_replay_enabled=True)


def test_layerwise_rollout_selects_batch_capable_oot_backend():
    options = make_rollout(prune_after_layer=15).build_launch_options(routing_replay_enabled=False)

    assert options.hf_overrides["architectures"] == [
        "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration"
    ]
    assert options.hf_overrides["vision_token_pruning"] == {
        "keep_ratio": 0.5,
        "selector": "random",
        "selector_kwargs": {},
        "prune_after_layer": 15,
    }
    assert options.cli_args["enable_chunked_prefill"] is False
    assert options.cli_args["enable_prefix_caching"] is False
    assert options.cli_args["enforce_eager"] is True
    assert "max_num_seqs" not in options.cli_args


def test_rollout_passes_selector_to_vllm_and_selection_protocol():
    rollout = make_rollout(selector="uniform")
    options = rollout.build_launch_options(routing_replay_enabled=False)
    selection = rollout.decode_selection([[[1]], [[4]]], original_token_count=4)

    assert options.hf_overrides["vision_token_pruning"]["selector"] == "uniform"
    assert selection["selector"] == "uniform"


def test_rollout_preserves_strategy_options_and_identity():
    rollout = make_rollout(
        selector="examples.vision_token_pruning.custom_strategies:feature_norm",
        selector_kwargs={"offset": 2, "nested": {"b": 2, "a": 1}},
    )
    options = rollout.build_launch_options(routing_replay_enabled=False)
    selection = rollout.decode_selection([[[1]], [[4]]], original_token_count=4)

    assert options.hf_overrides["vision_token_pruning"]["selector_kwargs"] == {
        "offset": 2,
        "nested": {"b": 2, "a": 1},
    }
    assert len(selection["selector_fingerprint"]) == 64
