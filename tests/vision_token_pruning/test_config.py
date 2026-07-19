import pytest

from verl.models.vision_token_pruning.config import (
    VisionTokenPruningConfig,
    coerce_vision_token_pruning_config,
    compute_selector_fingerprint,
)


def test_enabled_config_defaults_to_layer0_random_pruning():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.25)

    assert config.keep_ratio == 0.25
    assert config.prune_after_layer == -1
    assert config.selector == "random"
    assert config.uses_layerwise_backend is False
    assert not hasattr(config, "method")
    assert not hasattr(config, "mode")


def test_non_negative_layer_selects_experimental_layerwise_backend():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.5, prune_after_layer=15)

    assert config.uses_layerwise_backend is True
    assert config.backend_name == "layerwise_flex"


def test_invalid_layer_boundary_is_rejected():
    with pytest.raises(ValueError, match="prune_after_layer"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=0.5, prune_after_layer=-2)


def test_selector_name_must_be_non_empty():
    with pytest.raises(ValueError, match="selector"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=0.5, selector="  ")


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.1])
def test_keep_ratio_must_be_in_open_closed_unit_interval(keep_ratio):
    with pytest.raises(ValueError, match="keep_ratio"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=keep_ratio)


def test_enabled_pruning_rejects_noop_ratio():
    with pytest.raises(ValueError, match="requires keep_ratio < 1"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=1.0)


def test_hydra_mapping_is_normalized_at_the_boundary():
    config = coerce_vision_token_pruning_config({"enabled": True, "keep_ratio": 0.5})

    assert config == VisionTokenPruningConfig(enabled=True, keep_ratio=0.5)


def test_selector_fingerprint_is_stable_and_mapping_order_independent():
    first = compute_selector_fingerprint("custom", {"alpha": 1, "nested": {"x": 2, "y": 3}})
    second = compute_selector_fingerprint("custom", {"nested": {"y": 3, "x": 2}, "alpha": 1})

    assert first == second
    assert len(first) == 64


def test_selector_kwargs_cannot_override_request_inputs():
    with pytest.raises(ValueError, match="reserved inputs"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            selector_kwargs={"features": "spoofed"},
        )


def test_selector_kwargs_reject_non_finite_json_values():
    with pytest.raises(ValueError, match="JSON serializable"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            selector_kwargs={"temperature": float("nan")},
        )


def test_backend_profile_is_explicit():
    assert VisionTokenPruningConfig(enabled=True, keep_ratio=0.5).backend_name == "physical"
    assert (
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            prune_after_layer=3,
        ).backend_name
        == "layerwise_flex"
    )


def test_compact_flash_remains_an_explicit_reference_backend():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=3,
        layerwise_backend="compact_flash",
    )

    assert config.backend_name == "layerwise_compact_flash"


def test_decoder_key_selection_is_layerwise_only():
    with pytest.raises(ValueError, match="decoder_key"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            selector="key_norm",
            selector_input="decoder_key",
        )
