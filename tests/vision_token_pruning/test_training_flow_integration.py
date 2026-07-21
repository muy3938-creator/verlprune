import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.embeddings import (  # noqa: E402
    KEEP_MASK_KEY,
    prune_visual_embeddings,
)
from verl.models.vision_token_pruning.protocol import decode_rollout_selection  # noqa: E402
from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest  # noqa: E402
from verl.models.vision_token_pruning.strategy import run_vision_token_strategy  # noqa: E402
from verl.models.vision_token_pruning.training import (  # noqa: E402
    attach_selection_to_multi_modal_inputs,
    replay_rollout_selection_on_attention_mask,
)


def test_rollout_to_actor_physical_pruning_supports_a_backward_step():
    kept_indices = run_vision_token_strategy(
        "random",
        VisionTokenSelectionRequest(
            token_count=6,
            keep_count=3,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(11),
        ),
    )
    routed_experts = [[[index + 1]] for index in kept_indices.tolist()]
    selection = decode_rollout_selection(
        routed_experts,
        keep_ratio=0.5,
        original_visual_token_count=6,
    )
    multimodal_inputs = attach_selection_to_multi_modal_inputs({}, selection.to_wire())

    input_ids = torch.tensor([[10, 99, 99, 99, 99, 99, 99, 11, 12]])
    attention_mask = replay_rollout_selection_on_attention_mask(
        input_ids,
        torch.ones_like(input_ids),
        [multimodal_inputs],
        image_token_id=99,
        expected_keep_ratio=0.5,
    )
    compact_ids = input_ids[attention_mask.bool()]
    full_visual_features = torch.randn(6, 8)
    compact_visual_features = prune_visual_embeddings(
        full_visual_features,
        multimodal_inputs[KEEP_MASK_KEY],
    )
    rollout_compact_ids = torch.tensor([10, 99, 99, 99, 11, 12])
    full_positions = torch.arange(input_ids.numel())
    actor_compact_positions = full_positions[attention_mask.squeeze(0).bool()]
    rollout_compact_positions = torch.cat(
        (
            torch.tensor([0]),
            kept_indices + 1,
            torch.tensor([7, 8]),
        )
    )

    token_embedding = torch.nn.Embedding(128, 8)
    lm_head = torch.nn.Linear(8, 128)
    hidden = token_embedding(compact_ids)
    hidden[compact_ids == 99] = compact_visual_features
    loss = torch.nn.functional.cross_entropy(lm_head(hidden[:-1]), compact_ids[1:])
    loss.backward()

    assert torch.equal(compact_ids, rollout_compact_ids)
    assert torch.equal(actor_compact_positions, rollout_compact_positions)
    assert compact_visual_features.shape == (3, 8)
    assert token_embedding.weight.grad is not None
    assert lm_head.weight.grad is not None
