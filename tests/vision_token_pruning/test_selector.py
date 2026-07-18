import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.selectors import select_random_visual_tokens  # noqa: E402


def test_random_selector_is_seedable_sorted_and_keeps_mrope_anchor():
    first = select_random_visual_tokens(
        8,
        4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    second = select_random_visual_tokens(
        8,
        4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert len(first) == 4
    assert first[-1].item() == 7


def test_single_retained_token_is_mrope_anchor():
    selected = select_random_visual_tokens(5, 1, device=torch.device("cpu"))

    assert selected.tolist() == [4]
