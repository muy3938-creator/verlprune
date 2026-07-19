import math

import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.dynamic import select_dynamic_visual_kv  # noqa: E402


def test_visual_mass_softmax_uses_complete_context():
    query = torch.tensor([[1.0]])
    # Two visual keys have zero logits; a strong text key takes almost all mass.
    keys = torch.tensor([[[0.0]], [[0.0]], [[8.0]]])
    visual = torch.tensor([True, True, False])

    result = select_dynamic_visual_kv(
        query,
        keys,
        visual,
        softmax_scale=1.0,
        temperature=1.0,
        budget_mode="visual_mass",
        fixed_keep_ratio=0.5,
    )

    expected_mass = 2.0 / (2.0 + math.exp(8.0))
    assert float(result.visual_mass_max) == pytest.approx(expected_mass)
    assert result.keep_count == 1


def test_fixed_budget_reselects_tokens_for_each_query_and_supports_gqa():
    keys = torch.tensor(
        [
            [[4.0, 0.0]],
            [[0.0, 4.0]],
            [[1.0, 1.0]],
            [[0.0, 0.0]],
        ]
    )
    visual = torch.tensor([True, True, True, False])

    first = select_dynamic_visual_kv(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        keys,
        visual,
        softmax_scale=1.0,
        temperature=0.5,
        budget_mode="fixed",
        fixed_keep_ratio=0.2,
    )
    second = select_dynamic_visual_kv(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        keys,
        visual,
        softmax_scale=1.0,
        temperature=0.5,
        budget_mode="fixed",
        fixed_keep_ratio=0.2,
    )

    assert first.kept_visual_indices.tolist() == [0]
    assert second.kept_visual_indices.tolist() == [1]


def test_visual_mass_budget_respects_ratio_clamps():
    query = torch.tensor([[1.0]])
    keys = torch.ones(10, 1, 1)
    visual = torch.tensor([True] * 8 + [False] * 2)

    result = select_dynamic_visual_kv(
        query,
        keys,
        visual,
        softmax_scale=1.0,
        temperature=1.0,
        budget_mode="visual_mass",
        fixed_keep_ratio=0.5,
        min_keep_ratio=0.25,
        max_keep_ratio=0.5,
    )

    # Unclamped mass is 0.8 -> ceil(6.4)=7; the 50% cap keeps four.
    assert result.keep_count == 4
