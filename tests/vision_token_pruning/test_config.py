import pytest

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig


def test_enabled_config_is_fixed_layer0_random_pruning():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.25)

    assert config.keep_ratio == 0.25
    assert not hasattr(config, "layer")
    assert not hasattr(config, "method")
    assert not hasattr(config, "mode")


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.1])
def test_keep_ratio_must_be_in_open_closed_unit_interval(keep_ratio):
    with pytest.raises(ValueError, match="keep_ratio"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=keep_ratio)


def test_enabled_pruning_rejects_noop_ratio():
    with pytest.raises(ValueError, match="requires keep_ratio < 1"):
        VisionTokenPruningConfig(enabled=True, keep_ratio=1.0)
