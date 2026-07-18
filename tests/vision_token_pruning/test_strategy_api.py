import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.strategy import (  # noqa: E402
    VisionTokenSelectionEngine,
    VisionTokenSelectionRequest,
    run_vision_token_strategy,
)


def test_request_strategy_receives_options_and_features_by_module_path():
    features = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [4.0, 4.0],
            [3.0, 3.0],
            [0.0, 0.0],
        ]
    )
    request = VisionTokenSelectionRequest(
        token_count=5,
        keep_count=3,
        device=torch.device("cpu"),
        features=features,
        options={"channel_start": 1},
    )

    selected = run_vision_token_strategy(
        "examples.vision_token_pruning.custom_strategies:feature_norm",
        request,
    )

    assert selected.tolist() == [2, 3, 4]


def test_selection_engine_owns_deterministic_per_request_seeds():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.5, selector="random")
    features = torch.zeros(8, 2)
    first_engine = VisionTokenSelectionEngine(config, seed=17)
    second_engine = VisionTokenSelectionEngine(config, seed=17)

    first_sequence = [first_engine.select(features, grid_thw=[1, 2, 4]) for _ in range(2)]
    second_sequence = [second_engine.select(features, grid_thw=[1, 2, 4]) for _ in range(2)]

    assert all(
        torch.equal(first, second)
        for first, second in zip(first_sequence, second_sequence, strict=True)
    )
    assert first_engine.selection_count == 2
    assert not torch.equal(first_sequence[0], first_sequence[1])


def test_legacy_module_strategy_remains_compatible():
    request = VisionTokenSelectionRequest(
        token_count=6,
        keep_count=3,
        device=torch.device("cpu"),
    )

    selected = run_vision_token_strategy(
        "verl.models.vision_token_pruning.selectors:select_uniform_visual_tokens",
        request,
    )

    assert selected.tolist() == [0, 2, 5]
