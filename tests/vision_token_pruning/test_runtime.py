import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.protocol import VisionTokenSelection  # noqa: E402
from verl.models.vision_token_pruning.runtime import (  # noqa: E402
    KEEP_MASK_KEY,
    attach_selection_to_multi_modal_inputs,
    prepare_actor_pruning_inputs,
    prune_visual_embeddings,
    replay_rollout_selection_on_attention_mask,
    strip_pruning_metadata,
    strip_selection_metadata,
)


def test_rollout_selection_physically_compacts_actor_tokens_and_features():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=4,
        kept_visual_indices=(0, 3),
    )
    inputs = attach_selection_to_multi_modal_inputs({"pixel_values": torch.ones(1)}, selection.to_wire())
    input_ids = torch.tensor([[7, 99, 99, 99, 99, 8]])
    attention_mask = torch.ones((1, 6), dtype=torch.long)

    pruned_attention_mask = replay_rollout_selection_on_attention_mask(
        input_ids,
        attention_mask,
        [inputs],
        image_token_id=99,
        expected_keep_ratio=0.5,
    )
    compact_input_ids = input_ids[pruned_attention_mask.bool()]
    embeddings = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    compact_embeddings = prune_visual_embeddings(
        embeddings,
        strip_selection_metadata(inputs)[KEEP_MASK_KEY],
    )

    assert pruned_attention_mask.tolist() == [[1, 1, 0, 0, 1, 1]]
    assert compact_input_ids.tolist() == [7, 99, 99, 8]
    assert compact_embeddings.tolist() == [[0.0, 1.0], [6.0, 7.0]]
    assert KEEP_MASK_KEY not in strip_pruning_metadata(inputs)


def test_actor_rejects_tampered_keep_mask():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=4,
        kept_visual_indices=(0, 3),
    )
    inputs = attach_selection_to_multi_modal_inputs({}, selection.to_wire())
    inputs[KEEP_MASK_KEY] = torch.tensor([False, True, False, True])

    with pytest.raises(ValueError, match="does not match rollout selection"):
        replay_rollout_selection_on_attention_mask(
            torch.tensor([[99, 99, 99, 99]]),
            torch.ones((1, 4), dtype=torch.long),
            [inputs],
            image_token_id=99,
            expected_keep_ratio=0.5,
        )


def test_teacher_preparation_removes_all_pruning_protocol_fields():
    selection = VisionTokenSelection(
        keep_ratio=0.5,
        original_visual_token_count=2,
        kept_visual_indices=(1,),
    )
    inputs = attach_selection_to_multi_modal_inputs({"pixel_values": torch.ones(1)}, selection.to_wire())

    prepared = prepare_actor_pruning_inputs(
        input_ids=torch.tensor([[99, 99]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        per_sample_multi_modal_inputs=[inputs],
        image_token_id=99,
        config=VisionTokenPruningConfig(enabled=True, keep_ratio=0.5),
        apply_pruning=False,
    )

    assert KEEP_MASK_KEY not in prepared.per_sample_multi_modal_inputs[0]
    assert torch.equal(prepared.attention_mask, torch.ones((1, 2), dtype=torch.long))
