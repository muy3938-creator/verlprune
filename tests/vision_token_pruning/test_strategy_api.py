import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.strategy import (  # noqa: E402
    VisionTokenSelectionEngine,
    VisionTokenSelectionRequest,
    available_vision_token_strategies,
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


def test_decoder_key_norm_strategy_uses_layer_state_and_keeps_anchor():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=15,
        selector="key_norm",
        selector_input="decoder_key",
    )
    engine = VisionTokenSelectionEngine(config, seed=3)
    key = torch.tensor([[[1.0]], [[4.0]], [[2.0]], [[100.0]]])

    selected = engine.select_decoder_states(
        query_states=torch.zeros_like(key),
        key_states=key,
        value_states=torch.zeros_like(key),
        layer_index=15,
    )

    # The final token is always the MRoPE anchor, not a score competitor.
    assert selected.tolist() == [1, 3]
    assert engine.selection_count == 1


def test_four_paper_methods_are_exposed_as_builtin_experiment_modes():
    assert {"dart", "divprune", "greedy_prune"}.issubset(
        available_vision_token_strategies()
    )


def test_divprune_matches_manual_max_min_diversity_fixture():
    values = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.5, 0.5],
        ]
    )
    selected = run_vision_token_strategy(
        "divprune",
        VisionTokenSelectionRequest(
            token_count=5,
            keep_count=3,
            device=torch.device("cpu"),
            features=values,
            value_states=values,
        ),
    )

    # Token 3 has the largest nearest-neighbour distance; token 0 is then
    # farthest from it. Token 4 is the platform's mandatory MRoPE anchor.
    assert selected.tolist() == [0, 3, 4]


def test_dart_uses_key_norm_pivot_then_keeps_its_farthest_duplicate_candidate():
    visual_keys = torch.tensor(
        [[10.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.9, 0.0], [0.0, -1.0], [0.1, 0.1]]
    )
    visual_values = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.0, -1.0], [0.5, 0.5]]
    )
    context_keys = torch.cat((visual_keys, torch.tensor([[0.0, 2.0], [0.0, 3.0]])))
    context_values = torch.cat((visual_values, torch.tensor([[0.0, 1.0], [0.0, -1.0]])))
    selected = run_vision_token_strategy(
        "dart",
        VisionTokenSelectionRequest(
            token_count=6,
            keep_count=3,
            device=torch.device("cpu"),
            key_states=visual_keys,
            value_states=visual_values,
            context_key_states=context_keys,
            context_value_states=context_values,
            visual_context_mask=torch.tensor([True] * 6 + [False, False]),
        ),
    )

    assert selected.tolist() == [0, 1, 5]


def test_greedy_prune_combines_last_text_saliency_with_redundancy_suppression():
    visual_values = torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]]
    )
    context_values = torch.cat((visual_values, torch.tensor([[1.0, 0.0]])))
    selected = run_vision_token_strategy(
        "greedy_prune",
        VisionTokenSelectionRequest(
            token_count=5,
            keep_count=3,
            device=torch.device("cpu"),
            value_states=visual_values,
            context_value_states=context_values,
            visual_context_mask=torch.tensor([True] * 5 + [False]),
            options={"similarity_threshold": 0.9},
        ),
    )

    # Token 1 is more than 0.9-similar to the most salient token 0, so the
    # next selected semantic region is token 2. Token 4 is the anchor.
    assert selected.tolist() == [0, 2, 4]
