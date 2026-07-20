import torch

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine
from verl.models.vision_token_pruning.transformers_sampler import (
    TransformersPruningState,
    _pruned_sdpa,
)


def _state(selector: str = "uniform") -> TransformersPruningState:
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=1,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decoder_key",
        selector=selector,
    )
    return TransformersPruningState(
        config=config,
        visual_mask=torch.tensor([[False, True, True, True, True, False]]),
        engines=[VisionTokenSelectionEngine(config, seed=0)],
        static_indices=[torch.tensor([0, 3])],
    )


def test_static_key_keep_preserves_text_and_selected_visual_tokens():
    state = _state()
    assert state.static_key_keep(8, torch.device("cpu")).tolist() == [
        [True, True, False, False, True, True, True, True]
    ]


def test_dynamic_rows_start_with_unpruned_prompt_queries():
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        prune_after_layer=1,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decode_query",
        selector="vision_pulse",
        selector_kwargs={"budget_mode": "fixed"},
    )
    state = TransformersPruningState(
        config=config,
        visual_mask=torch.tensor([[False, True, True, False]]),
        engines=[VisionTokenSelectionEngine(config, seed=0)],
    )
    assert state.dynamic_rows == [[(), (), (), ()]]


def test_pruned_sdpa_matches_explicit_boolean_mask():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    keep = torch.tensor([[True, False, True]])
    actual = _pruned_sdpa(
        query,
        key,
        value,
        keep,
        scaling=0.5,
        dropout=0.0,
        dynamic_last_query_only=False,
    )
    causal = torch.ones(3, 3, dtype=torch.bool).tril() & keep[:, None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal[:, None],
        scale=0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
