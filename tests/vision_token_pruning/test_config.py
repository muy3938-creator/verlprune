import pytest

from verl.models.vision_token_pruning.config import (
    VisionTokenPruningConfig,
    coerce_vision_token_pruning_config,
    compute_selector_fingerprint,
)


def test_enabled_config_defaults_to_layer0_embedding_pruning():
    config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.25)

    assert config.keep_ratio == 0.25
    assert config.prune_after_layer == -1
    assert config.selector == "embedding_norm"
    assert config.selection_layer == 0
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
    assert (
        VisionTokenPruningConfig(enabled=True, keep_ratio=0.5).backend_name
        == "prefill_physical_shared_kv"
    )
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


def test_flash_can_accelerate_layers_through_the_flex_pruning_boundary():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.1,
        prune_after_layer=15,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
    )

    assert config.to_backend_payload()["pre_pruning_backend"] == "flash"


@pytest.mark.parametrize("pre_pruning_backend", ["sdpa", "flash_attention_2"])
def test_pre_pruning_backend_rejects_unknown_values(pre_pruning_backend):
    with pytest.raises(ValueError, match="pre_pruning_backend"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.1,
            prune_after_layer=15,
            pre_pruning_backend=pre_pruning_backend,
        )


def test_flash_pre_pruning_backend_requires_layerwise_flex():
    with pytest.raises(ValueError, match="prune_after_layer"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.1,
            pre_pruning_backend="flash",
        )
    with pytest.raises(ValueError, match="layerwise_backend='flex'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.1,
            prune_after_layer=15,
            layerwise_backend="compact_flash",
            pre_pruning_backend="flash",
        )


def test_decoder_key_selection_is_layerwise_only():
    with pytest.raises(ValueError, match="decoder_key"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            selector="key_norm",
            selector_input="decoder_key",
        )

    with pytest.raises(ValueError, match="layerwise_backend='flex'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.5,
            prune_after_layer=15,
            layerwise_backend="compact_flash",
            selector="key_norm",
            selector_input="decoder_key",
        )


@pytest.mark.parametrize("selector", ["dart", "greedy_prune"])
def test_text_conditioned_paper_selectors_require_decoder_context(selector):
    with pytest.raises(ValueError, match="selector_input='decoder_key'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.1,
            prune_after_layer=15,
            selector=selector,
            selector_input="vision_embedding",
        )


def test_decode_query_selects_dynamic_flex_mode():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.05,
        prune_after_layer=15,
        selector="vision_pulse",
        selector_input="decode_query",
    )

    assert config.uses_dynamic_decode_selection
    assert config.backend_name == "layerwise_flex"


def test_decode_query_rejects_compact_or_non_vision_pulse_modes():
    with pytest.raises(ValueError, match="layerwise_backend='flex'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=15,
            layerwise_backend="compact_flash",
            selector="vision_pulse",
            selector_input="decode_query",
        )
    with pytest.raises(ValueError, match="selector_input='decode_query'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=15,
            selector="vision_pulse",
            selector_input="decoder_key",
        )
    with pytest.raises(ValueError, match="vision_pulse"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=15,
            selector="uniform",
            selector_input="decode_query",
        )


@pytest.mark.parametrize(
    ("selector_kwargs", "message"),
    [
        ({"temprature": 0.1}, "unsupported vision_pulse selector_kwargs"),
        ({"budget_mode": "adaptive"}, "budget_mode"),
        ({"temperature": 0.0}, "temperature"),
        ({"min_keep_ratio": 0.8, "max_keep_ratio": 0.2}, "keep-ratio clamps"),
        ({"capture_capacity": 0}, "capture_capacity"),
    ],
)
def test_dynamic_decode_selection_rejects_invalid_algorithm_options(
    selector_kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.1,
            prune_after_layer=15,
            layerwise_backend="flex",
            selector_input="decode_query",
            selector="vision_pulse",
            selector_kwargs=selector_kwargs,
        )


@pytest.mark.parametrize(
    ("selector", "selector_kwargs", "message"),
    [
        ("dart", {"pivot_image_token": 4}, "unsupported dart"),
        ("dart", {"pivot_image_tokens": 0}, "pivot counts"),
        ("dart", {"pivot_text_tokens": 1.5}, "pivot counts"),
        ("divprune", {"threshold": 0.1}, "unsupported divprune"),
        ("greedy_prune", {"threshold": 0.9}, "unsupported greedy_prune"),
        ("greedy_prune", {"similarity_threshold": 1.1}, "similarity_threshold"),
    ],
)
def test_paper_selectors_reject_misspelled_or_invalid_options(
    selector,
    selector_kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=15,
            layerwise_backend="flex",
            selector_input="decoder_key",
            selector=selector,
            selector_kwargs=selector_kwargs,
        )


def test_selector_specific_validation_uses_normalized_name():
    with pytest.raises(ValueError, match="selector_input='decoder_key'"):
        VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=0.05,
            prune_after_layer=15,
            selector=" dart ",
            selector_input="vision_embedding",
        )
