import pytest

from verl.models.vision_token_pruning.protocol import (
    DynamicVisionTokenSelection,
    VisionTokenSelection,
    compute_keep_count,
    decode_rollout_selection,
    selection_from_wire,
)


def test_selection_round_trip():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=6,
        kept_visual_indices=(0, 2, 5),
    )

    assert VisionTokenSelection.from_wire(selection.to_wire()) == selection
    assert selection.to_wire()["selector"] == "random"
    assert len(selection.to_wire()["selector_fingerprint"]) == 64


def test_selection_default_fingerprint_tracks_non_default_selector():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        selector="uniform",
        original_visual_token_count=4,
        kept_visual_indices=(0, 3),
    )

    assert selection.selector_fingerprint != VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=4,
        kept_visual_indices=(0, 3),
    ).selector_fingerprint


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


def test_selection_allows_dropping_the_final_visual_token():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=4,
        kept_visual_indices=(0, 2),
    )

    assert selection.kept_visual_indices == (0, 2)


def test_decode_vllm_metadata():
    selection = decode_rollout_selection(
        [[[0]], [[1]], [[3]], [[4]], [[0]]],
        keep_ratio=0.75,
        original_visual_token_count=4,
    )

    assert selection.kept_visual_indices == (0, 2, 3)


def test_decode_records_the_rollout_selector():
    selection = decode_rollout_selection(
        [[[1]], [[4]]],
        keep_ratio=0.5,
        original_visual_token_count=4,
        selector="uniform",
    )

    assert selection.selector == "uniform"


def test_decode_vllm_metadata_requires_exact_count():
    with pytest.raises(ValueError, match="expected 2"):
        decode_rollout_selection(
            [[[1]]],
            keep_ratio=0.5,
            original_visual_token_count=4,
        )


def test_dynamic_selection_round_trip_allows_variable_per_query_budgets():
    selection = DynamicVisionTokenSelection(
        nominal_keep_ratio=0.05,
        original_visual_token_count=8,
        query_kept_visual_indices=((), (1, 6), (0, 2, 5), (7,)),
        selector="vision_pulse",
    )

    assert selection_from_wire(selection.to_wire()) == selection
