"""Policy contract tests via the canonical strategy API (no selectors facade)."""

import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest  # noqa: E402
from verl.models.vision_token_pruning.strategy import (  # noqa: E402
    available_vision_token_strategies,
    register_vision_token_strategy,
    run_vision_token_strategy,
)


def _run(name: str, token_count: int, keep_count: int, **kwargs):
    return run_vision_token_strategy(
        name,
        VisionTokenSelectionRequest(
            token_count=token_count,
            keep_count=keep_count,
            device=torch.device("cpu"),
            **kwargs,
        ),
    )


def test_random_selector_is_seedable_sorted_and_has_no_forced_anchor():
    gen1 = torch.Generator().manual_seed(1)
    gen2 = torch.Generator().manual_seed(1)
    first = _run("random", 8, 4, generator=gen1)
    second = _run("random", 8, 4, generator=gen2)

    assert torch.equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert len(first) == 4
    assert 7 not in first.tolist()


def test_single_retained_token_is_sampled_without_a_forced_position():
    selected = _run("random", 5, 1, generator=torch.Generator().manual_seed(7))
    assert len(selected) == 1


def test_builtin_uniform_selector_is_available_and_validated():
    selected = _run("uniform", 8, 4)
    assert {"random", "uniform"}.issubset(available_vision_token_strategies())
    assert selected.tolist() == [0, 2, 5, 7]


def test_custom_selector_receives_features_and_grid_without_actor_changes():
    observed = {}

    def keep_tail(token_count, keep_count, *, device, features, grid_thw, **_):
        observed.update(features=features, grid_thw=grid_thw)
        return torch.arange(token_count - keep_count, token_count, device=device)

    register_vision_token_strategy("test_keep_tail", keep_tail, replace=True)
    features = torch.randn(6, 4)
    selected = _run(
        "test_keep_tail",
        6,
        3,
        features=features,
        grid_thw=[1, 4, 6],
    )

    assert selected.tolist() == [3, 4, 5]
    assert observed["features"] is features
    assert observed["grid_thw"] == [1, 4, 6]


def test_selector_output_contract_accepts_any_valid_subset():
    register_vision_token_strategy(
        "test_bad_anchor",
        lambda token_count, keep_count, *, device, **_: torch.arange(keep_count, device=device),
        replace=True,
    )
    selected = _run("test_bad_anchor", 8, 4)
    assert selected.tolist() == [0, 1, 2, 3]


def test_selector_can_be_loaded_by_dotted_path_for_vllm_workers():
    selected = _run(
        "verl.models.vision_token_pruning.policies.uniform:uniform_policy",
        6,
        3,
    )
    assert selected.tolist() == [0, 2, 5]
