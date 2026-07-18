import pytest

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.rollout import VisionTokenPruningRollout


def make_rollout(*, enabled=True):
    return VisionTokenPruningRollout(
        VisionTokenPruningConfig(enabled=enabled, keep_ratio=0.5),
        model_type="qwen2_5_vl",
        image_token_id=99,
    )


def test_rollout_policy_owns_vllm_launch_overrides():
    options = make_rollout().build_launch_options(routing_replay_enabled=False)

    assert options.hf_overrides["architectures"] == ["VerlRandomPrunedQwen2_5VLForConditionalGeneration"]
    assert options.hf_overrides["vision_token_pruning"] == {"keep_ratio": 0.5}
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


def test_rollout_policy_rejects_replay_channel_conflict():
    with pytest.raises(ValueError, match="cannot share routed_experts"):
        make_rollout().build_launch_options(routing_replay_enabled=True)
