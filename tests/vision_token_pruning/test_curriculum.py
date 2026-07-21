from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.curriculum import resolve_keep_ratio  # noqa: E402
from verl.models.vision_token_pruning.protocol import VisionTokenSelection  # noqa: E402
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine  # noqa: E402
from verl.models.vision_token_pruning.training import (  # noqa: E402
    SELECTION_WIRE_KEY,
    replay_rollout_selection_on_attention_mask,
)
from verl.models.vision_token_pruning.embeddings import KEEP_MASK_KEY  # noqa: E402


def test_piecewise_keep_ratio_schedule_interpolates_and_clamps():
    schedule = {"milestones": [[0, 0.50], [80, 0.10], [100, 0.05]]}

    assert resolve_keep_ratio(schedule, 0, fallback=0.05) == pytest.approx(0.50)
    assert resolve_keep_ratio(schedule, 40, fallback=0.05) == pytest.approx(0.30)
    assert resolve_keep_ratio(schedule, 80, fallback=0.05) == pytest.approx(0.10)
    assert resolve_keep_ratio(schedule, 100, fallback=0.05) == pytest.approx(0.05)
    assert resolve_keep_ratio(schedule, 500, fallback=0.05) == pytest.approx(0.05)


def test_schedule_without_global_step_fails_fast():
    schedule = {"milestones": [[0, 0.50], [80, 0.10]]}
    with pytest.raises(ValueError, match="global_step is missing"):
        resolve_keep_ratio(schedule, None, fallback=0.05)


def test_empty_schedule_uses_fallback_without_step():
    assert resolve_keep_ratio({}, None, fallback=0.42) == 0.42
    assert resolve_keep_ratio(None, None, fallback=0.42) == 0.42


def test_schedule_is_carried_to_vllm_payload():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.05,
        prune_after_layer=0,
        selector="random",
        selector_input="vision_embedding",
        keep_ratio_schedule={"milestones": [[0, 0.5], [80, 0.1], [100, 0.05]]},
    )

    assert config.uses_keep_ratio_schedule
    assert config.to_backend_payload()["keep_ratio_schedule"]["milestones"][1] == [80, 0.1]


def test_actor_replays_actual_scheduled_ratio_instead_of_static_config_ratio():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        selector="random",
        original_visual_token_count=4,
        kept_visual_indices=(0, 2),
    )
    inputs = torch.tensor([[99, 99, 99, 99, 7]])
    attention = torch.ones_like(inputs)
    output = replay_rollout_selection_on_attention_mask(
        inputs,
        attention,
        [
            {
                KEEP_MASK_KEY: torch.tensor([True, False, True, False]),
                SELECTION_WIRE_KEY: selection.to_wire(),
            }
        ],
        image_token_id=99,
        expected_keep_ratio=None,
        expected_selector="random",
    )

    assert output.tolist() == [[1, 0, 1, 0, 1]]


def test_runtime_ratio_overrides_static_selector_budget():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.05,
        prune_after_layer=0,
        selector="random",
        selector_input="vision_embedding",
    )
    engine = VisionTokenSelectionEngine(config, seed=7)
    selected = engine.select(torch.randn(20, 4), grid_thw=None, keep_ratio=0.50)

    assert len(selected) == 10


@pytest.mark.parametrize(
    "schedule",
    [
        {"milestones": [[0, 0.5], [0, 0.1]]},
        {"milestones": [[0, 0.5], [10, 1.0]]},
        {"milestones": [[0, 0.5], [10, 0.1]], "interpolation": "cosine"},
    ],
)
def test_invalid_keep_ratio_schedule_is_rejected(schedule):
    with pytest.raises(ValueError):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=0,
            selector="random",
            selector_input="vision_embedding",
            keep_ratio_schedule=schedule,
        )
