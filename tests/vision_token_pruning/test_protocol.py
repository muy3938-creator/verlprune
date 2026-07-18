import pytest

from verl.models.vision_token_pruning.protocol import (
    VisionTokenSelection,
    compute_keep_count,
    decode_rollout_selection,
)


def test_selection_round_trip():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=6,
        kept_visual_indices=(0, 2, 5),
    )

    assert VisionTokenSelection.from_wire(selection.to_wire()) == selection


def test_keep_count_rounds_and_never_drops_every_token():
    assert compute_keep_count(5, 0.5) == 2
    assert compute_keep_count(2, 0.1) == 1


def test_selection_requires_sorted_unique_indices():
    with pytest.raises(ValueError, match="sorted and unique"):
        VisionTokenSelection(
            keep_ratio=0.5,
            original_visual_token_count=4,
            kept_visual_indices=(2, 1),
        )


def test_selection_requires_final_mrope_anchor():
    with pytest.raises(ValueError, match="MRoPE anchor"):
        VisionTokenSelection(
            keep_ratio=0.5,
            original_visual_token_count=4,
            kept_visual_indices=(0, 2),
        )


def test_decode_vllm_metadata():
    selection = decode_rollout_selection(
        [[[0]], [[1]], [[3]], [[4]], [[0]]],
        keep_ratio=0.75,
        original_visual_token_count=4,
    )

    assert selection.kept_visual_indices == (0, 2, 3)


def test_decode_vllm_metadata_requires_exact_count():
    with pytest.raises(ValueError, match="expected 2"):
        decode_rollout_selection(
            [[[1]]],
            keep_ratio=0.5,
            original_visual_token_count=4,
        )
