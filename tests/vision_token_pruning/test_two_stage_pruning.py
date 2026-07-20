import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import (  # noqa: E402
    VisionTokenPruningConfig,
    compute_selector_fingerprint,
)
from verl.models.vision_token_pruning.protocol import (  # noqa: E402
    DynamicVisionTokenSelection,
    TwoStageVisionTokenSelection,
    VisionTokenSelection,
    selection_from_wire,
)
from verl.models.vision_token_pruning.training import (  # noqa: E402
    attach_selection_to_multi_modal_inputs,
    prepare_actor_pruning_inputs,
)
from verl.models.vision_token_pruning.transport import (  # noqa: E402
    decode_vllm_two_stage_selection_capture,
)


OPTIONS = {"budget_mode": "fixed", "temperature": 0.1, "capture_capacity": 6}


def _config() -> VisionTokenPruningConfig:
    return VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        selector="vision_pulse",
        selector_kwargs=OPTIONS,
        selector_input="decode_query",
        prune_after_layer=3,
        prefill_keep_ratio=0.5,
        prefill_selector="embedding_norm",
    )


def _capture_row(*values: int) -> list[list[int]]:
    row = list(values) + [0] * (6 - len(values))
    return [row]


def test_two_stage_config_exposes_relative_and_effective_budgets():
    config = _config()

    assert config.uses_two_stage_pruning
    assert config.uses_dynamic_decode_selection
    assert config.backend_name == "prefill_physical_then_dynamic_decode_flex"
    assert config.prefill_keep_ratio * config.keep_ratio == pytest.approx(0.25)
    assert config.to_backend_payload()["prefill_selector"] == "embedding_norm"


def test_two_stage_config_rejects_non_dynamic_second_stage():
    with pytest.raises(ValueError, match="two-stage pruning requires"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            selector="embedding_norm",
            selector_input="vision_embedding",
            prune_after_layer=3,
            prefill_keep_ratio=0.5,
        )


def test_two_stage_protocol_round_trip_and_effective_ratio():
    selection = TwoStageVisionTokenSelection(
        prefill=VisionTokenSelection(
            keep_ratio=0.5,
            selector="embedding_norm",
            original_visual_token_count=4,
            kept_visual_indices=(0, 2),
        ),
        decode=DynamicVisionTokenSelection(
            nominal_keep_ratio=0.5,
            selector="vision_pulse",
            original_visual_token_count=2,
            query_kept_visual_indices=((), (1,), (0,)),
        ),
    )

    assert selection_from_wire(selection.to_wire()) == selection
    assert selection.effective_nominal_keep_ratio == pytest.approx(0.25)


def test_two_stage_capture_and_actor_replay_use_decode_indices_relative_to_prefill_subset():
    # Compact rollout order: text, original visual 0, original visual 2,
    # text query, decode query. The last top-k slot marks prefill metadata.
    capture = [
        _capture_row(),
        _capture_row(1, 4, 0, 0, 0, 1),
        _capture_row(3, 4, 0, 0, 0, 1),
        _capture_row(2),
        _capture_row(1),
    ]
    selection = decode_vllm_two_stage_selection_capture(
        capture,
        prefill_keep_ratio=0.5,
        prefill_selector="embedding_norm",
        prefill_selector_kwargs={},
        decode_keep_ratio=0.5,
        decode_selector="vision_pulse",
        decode_selector_kwargs=OPTIONS,
    )

    assert selection.prefill.kept_visual_indices == (0, 2)
    assert selection.decode.original_visual_token_count == 2
    assert selection.decode.query_kept_visual_indices == ((), (), (), (1,), (0,))

    attached = attach_selection_to_multi_modal_inputs({}, selection.to_wire())
    input_ids = torch.tensor([[7, 99, 99, 99, 99, 8, 10]])
    attention_mask = torch.ones_like(input_ids)
    prepared = prepare_actor_pruning_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        per_sample_multi_modal_inputs=[attached],
        image_token_id=99,
        config=_config(),
        apply_pruning=True,
    )

    assert prepared.attention_mask.tolist() == [[1, 1, 0, 1, 0, 1, 1]]
    dynamic = prepared.dynamic_layerwise_attention_mask
    assert dynamic is not None
    assert dynamic[0, 5, [1, 3]].tolist() == [False, True]
    assert dynamic[0, 6, [1, 3]].tolist() == [True, False]
    assert attached["vision_token_keep_mask"].tolist() == [True, False, True, False]
    assert selection.decode.selector_fingerprint == compute_selector_fingerprint(
        "vision_pulse",
        OPTIONS,
    )
