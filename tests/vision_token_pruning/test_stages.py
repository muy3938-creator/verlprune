"""StageSpec / PruningSpec: explicit three-semantic experiment model."""

from __future__ import annotations

import pytest

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.stages import (
    InputKind,
    PruningSpec,
    RuntimeKind,
    StageKind,
    StageSpec,
    pruning_spec_from_legacy_config,
    stage_specs_from_mapping,
)


def test_physical_legacy_config_maps_to_single_physical_stage():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.25)
    spec = config.to_pruning_spec()

    assert spec.enabled
    assert spec.runtime is RuntimeKind.PHYSICAL_FIXED
    assert len(spec.stages) == 1
    stage = spec.stages[0]
    assert stage.kind is StageKind.PHYSICAL_PRE_DECODER
    assert stage.input_kind is InputKind.VISION_EMBEDDING
    assert stage.policy == "embedding_norm"
    assert stage.keep_ratio == 0.25
    assert stage.observe_layer is None


def test_boundary_once_legacy_maps_observe_and_apply_layers():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=15,
        selector="key_norm",
        selector_input="decoder_key",
    )
    spec = pruning_spec_from_legacy_config(config)

    assert spec.runtime is RuntimeKind.FLEX_MASK
    assert spec.uses_boundary_once
    stage = spec.stages[0]
    assert stage.kind is StageKind.BOUNDARY_ONCE
    assert stage.input_kind is InputKind.BOUNDARY_QKV
    assert stage.observe_layer == 15
    assert stage.apply_from_layer == 16


def test_decode_query_legacy_maps_to_decode_stage():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.05,
        prune_after_layer=15,
        selector="vision_pulse",
        selector_input="decode_query",
    )
    spec = config.to_pruning_spec()

    assert spec.uses_decode_query
    assert not spec.uses_two_stage
    stage = spec.stages[0]
    assert stage.kind is StageKind.DECODE_QUERY
    assert stage.input_kind is InputKind.DECODE_QK
    assert stage.observe_layer == 15
    assert stage.apply_from_layer == 16


def test_two_stage_physical_prefill_plus_decode():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.25,
        selector="vision_pulse",
        selector_input="decode_query",
        selector_kwargs={"budget_mode": "fixed"},
        prefill_keep_ratio=0.5,
        prefill_prune_after_layer=-1,
        prune_after_layer=15,
    )
    spec = config.to_pruning_spec()

    assert spec.uses_two_stage
    assert spec.stages[0].kind is StageKind.PHYSICAL_PRE_DECODER
    assert spec.stages[1].kind is StageKind.DECODE_QUERY
    assert spec.prefill_stage is not None
    assert spec.prefill_stage.keep_ratio == 0.5
    assert spec.decode_stage is not None
    assert spec.decode_stage.keep_ratio == 0.25


def test_two_stage_delayed_boundary_prefill():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.25,
        selector="vision_pulse",
        selector_input="decode_query",
        selector_kwargs={"budget_mode": "fixed"},
        prefill_keep_ratio=0.5,
        prefill_prune_after_layer=7,
        prune_after_layer=15,
    )
    spec = config.to_pruning_spec()

    assert spec.stages[0].kind is StageKind.BOUNDARY_ONCE
    assert spec.stages[0].observe_layer == 7
    assert spec.stages[0].apply_from_layer == 8
    assert spec.stages[1].kind is StageKind.DECODE_QUERY


def test_stage_rejects_wrong_input_for_kind():
    with pytest.raises(ValueError, match="cannot use input_kind"):
        StageSpec(
            kind=StageKind.PHYSICAL_PRE_DECODER,
            policy="key_norm",
            keep_ratio=0.5,
            input_kind=InputKind.BOUNDARY_QKV,
        )


def test_boundary_requires_explicit_layers():
    with pytest.raises(ValueError, match="observe_layer and apply_from_layer"):
        StageSpec(
            kind=StageKind.BOUNDARY_ONCE,
            policy="uniform",
            keep_ratio=0.5,
            input_kind=InputKind.VISION_EMBEDDING,
        )


def test_apply_from_must_be_observe_plus_one():
    with pytest.raises(ValueError, match="apply_from_layer == observe_layer \\+ 1"):
        StageSpec(
            kind=StageKind.BOUNDARY_ONCE,
            policy="uniform",
            keep_ratio=0.5,
            input_kind=InputKind.VISION_EMBEDDING,
            observe_layer=3,
            apply_from_layer=5,
        )


def test_physical_rejects_schedule():
    with pytest.raises(ValueError, match="keep_ratio_schedule"):
        PruningSpec(
            enabled=True,
            runtime=RuntimeKind.PHYSICAL_FIXED,
            stages=(
                StageSpec(
                    kind=StageKind.PHYSICAL_PRE_DECODER,
                    policy="random",
                    keep_ratio=0.5,
                    input_kind=InputKind.VISION_EMBEDDING,
                ),
            ),
            keep_ratio_schedule={"milestones": [[0, 0.5], [10, 0.1]]},
        )


def test_decode_query_must_be_final_stage():
    prefill = StageSpec(
        kind=StageKind.DECODE_QUERY,
        policy="vision_pulse",
        keep_ratio=0.5,
        input_kind=InputKind.DECODE_QK,
        observe_layer=1,
        apply_from_layer=2,
    )
    boundary = StageSpec(
        kind=StageKind.BOUNDARY_ONCE,
        policy="uniform",
        keep_ratio=0.5,
        input_kind=InputKind.VISION_EMBEDDING,
        observe_layer=3,
        apply_from_layer=4,
    )
    with pytest.raises(ValueError, match="decode_query must be the final stage"):
        PruningSpec(
            enabled=True,
            runtime=RuntimeKind.FLEX_MASK,
            stages=(prefill, boundary),
        )


def test_stage_specs_from_mapping_requires_fields():
    with pytest.raises(ValueError, match="missing fields"):
        stage_specs_from_mapping([{"kind": "physical_pre_decoder", "policy": "random"}])


def test_stage_specs_from_mapping_builds_physical():
    stages = stage_specs_from_mapping(
        [
            {
                "kind": "physical_pre_decoder",
                "policy": "embedding_norm",
                "keep_ratio": 0.4,
                "input_kind": "vision_embedding",
            }
        ]
    )
    assert stages[0].kind is StageKind.PHYSICAL_PRE_DECODER
    assert stages[0].keep_ratio == 0.4


def test_disabled_spec_has_no_stages():
    spec = pruning_spec_from_legacy_config(VisionTokenPruningConfig(enabled=False))
    assert not spec.enabled
    assert spec.stages == ()


def test_physical_legacy_with_schedule_fails_fast():
    with pytest.raises(ValueError, match="keep_ratio_schedule"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            prune_after_layer=-1,
            keep_ratio_schedule={"milestones": [[0, 0.5], [10, 0.1]]},
        ).to_pruning_spec()
