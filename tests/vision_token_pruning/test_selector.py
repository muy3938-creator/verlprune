import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.selectors import (  # noqa: E402
    available_vision_token_selectors,
    register_vision_token_selector,
    select_random_visual_tokens,
    select_visual_tokens,
)


def test_random_selector_is_seedable_sorted_and_has_no_forced_anchor():
    first = select_random_visual_tokens(
        8,
        4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    second = select_random_visual_tokens(
        8,
        4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert len(first) == 4
    assert 7 not in first.tolist()


def test_single_retained_token_is_sampled_without_a_forced_position():
    selected = select_random_visual_tokens(
        5,
        1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert len(selected) == 1


def test_builtin_uniform_selector_is_available_and_validated():
    selected = select_visual_tokens(
        "uniform",
        8,
        4,
        device=torch.device("cpu"),
    )

    assert {"random", "uniform"}.issubset(available_vision_token_selectors())
    assert selected.tolist() == [0, 2, 5, 7]


def test_custom_selector_receives_features_and_grid_without_actor_changes():
    observed = {}

    def keep_tail(token_count, keep_count, *, device, features, grid_thw, **_):
        observed.update(features=features, grid_thw=grid_thw)
        return torch.arange(token_count - keep_count, token_count, device=device)

    register_vision_token_selector("test_keep_tail", keep_tail)
    features = torch.randn(6, 4)
    selected = select_visual_tokens(
        "test_keep_tail",
        6,
        3,
        device=torch.device("cpu"),
        features=features,
        grid_thw=[1, 4, 6],
    )

    assert selected.tolist() == [3, 4, 5]
    assert observed["features"] is features
    assert observed["grid_thw"] == [1, 4, 6]


def test_selector_output_contract_accepts_any_valid_subset():
    register_vision_token_selector(
        "test_bad_anchor",
        lambda token_count, keep_count, *, device, **_: torch.arange(keep_count, device=device),
    )

    selected = select_visual_tokens("test_bad_anchor", 8, 4, device=torch.device("cpu"))

    assert selected.tolist() == [0, 1, 2, 3]


def test_selector_can_be_loaded_by_dotted_path_for_vllm_workers():
    selected = select_visual_tokens(
        "verl.models.vision_token_pruning.selectors:select_uniform_visual_tokens",
        6,
        3,
        device=torch.device("cpu"),
    )

    assert selected.tolist() == [0, 2, 5]
