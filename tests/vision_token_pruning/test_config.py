import pytest

from verl.models.vision_token_pruning.config import (
    VisionTokenPruningConfig,
    coerce_vision_token_pruning_config,
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
