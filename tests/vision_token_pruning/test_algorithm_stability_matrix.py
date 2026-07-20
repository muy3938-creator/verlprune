"""Broad deterministic coverage for research selectors and layer boundaries."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.dynamic import select_dynamic_visual_kv  # noqa: E402
from verl.models.vision_token_pruning.protocol import compute_keep_count  # noqa: E402
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine  # noqa: E402


RATIOS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.95)
LAYERS = (0, 1, 7, 15, 27, 34)
TOKEN_COUNTS = (20, 64, 257)
STATIC_METHODS = (
    ("embedding_norm", "vision_embedding", {}),
    ("dart", "decoder_key", {"pivot_image_tokens": 4, "pivot_text_tokens": 4}),
    ("divprune", "decoder_key", {}),
    ("greedy_prune", "decoder_key", {"similarity_threshold": 0.9}),
)


@pytest.mark.parametrize(("selector", "selector_input", "options"), STATIC_METHODS)
@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("token_count", TOKEN_COUNTS)
def test_static_selector_matrix_returns_stable_exact_budget(
    selector,
    selector_input,
    options,
    layer,
    ratio,
    token_count,
):
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=ratio,
        prune_after_layer=layer,
        selector=selector,
        selector_input=selector_input,
        selector_kwargs=options,
    )
    generator = torch.Generator().manual_seed(1000 + token_count + layer)
    visual = torch.randn(token_count, 2, 8, generator=generator)
    engine = VisionTokenSelectionEngine(config, seed=123)

    if selector_input == "vision_embedding":
        first = engine.select(visual, grid_thw=None)
        replay = VisionTokenSelectionEngine(config, seed=123).select(visual, grid_thw=None)
    else:
        text = torch.randn(5, 2, 8, generator=generator)
        context = torch.cat((visual, text), dim=0)
        visual_mask = torch.zeros(len(context), dtype=torch.bool)
        visual_mask[:token_count] = True
        kwargs = dict(
            query_states=visual,
            key_states=visual,
            value_states=visual,
            context_query_states=context,
            context_key_states=context,
            context_value_states=context,
            visual_context_mask=visual_mask,
            layer_index=layer,
        )
        first = engine.select_decoder_states(**kwargs)
        replay = VisionTokenSelectionEngine(config, seed=123).select_decoder_states(**kwargs)

    assert len(first) == compute_keep_count(token_count, ratio)
    assert torch.equal(first, first.sort().values)
    assert len(first.unique()) == len(first)
    assert int(first.min()) >= 0
    assert int(first.max()) < token_count
    assert torch.equal(first, replay)


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("token_count", TOKEN_COUNTS)
def test_visionpulse_fixed_budget_matrix_matches_exact_requested_ratio(layer, ratio, token_count):
    generator = torch.Generator().manual_seed(2000 + token_count + layer)
    text_count = 7
    context_length = token_count + text_count
    query = torch.randn(8, 16, generator=generator)
    keys = torch.randn(context_length, 4, 16, generator=generator)
    visual_mask = torch.zeros(context_length, dtype=torch.bool)
    visual_mask[:token_count] = True

    result = select_dynamic_visual_kv(
        query,
        keys,
        visual_mask,
        softmax_scale=16**-0.5,
        temperature=0.1,
        budget_mode="fixed",
        fixed_keep_ratio=ratio,
    )

    assert result.keep_count == compute_keep_count(token_count, ratio)
    assert torch.equal(result.kept_visual_indices, result.kept_visual_indices.sort().values)
    assert torch.isfinite(result.importance_scores).all()
    assert torch.isfinite(result.per_head_visual_mass).all()


@pytest.mark.parametrize(
    ("prefill_layer", "decode_layer"),
    ((-1, 0), (-1, 15), (0, 0), (0, 1), (0, 15), (7, 15), (15, 15), (15, 27), (27, 34)),
)
@pytest.mark.parametrize("prefill_ratio", RATIOS)
@pytest.mark.parametrize("decode_ratio", RATIOS)
def test_two_stage_boundary_and_ratio_configuration_matrix(
    prefill_layer,
    decode_layer,
    prefill_ratio,
    decode_ratio,
):
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=decode_ratio,
        prune_after_layer=decode_layer,
        selector="vision_pulse",
        selector_input="decode_query",
        selector_kwargs={"budget_mode": "fixed", "capture_capacity": 64},
        prefill_keep_ratio=prefill_ratio,
        prefill_selector="embedding_norm",
        prefill_prune_after_layer=prefill_layer,
    )

    assert config.prefill_prune_after_layer <= config.prune_after_layer
    assert config.uses_physical_prefill_pruning == (prefill_layer == -1)
    assert config.uses_delayed_prefill_pruning == (prefill_layer >= 0)
    assert 0.05**2 <= prefill_ratio * decode_ratio <= 0.95**2
