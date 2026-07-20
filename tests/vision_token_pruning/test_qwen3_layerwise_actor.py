import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

from verl.models.transformers.qwen3_vl import qwen3_vl_attn_forward


def _attention(layer_index: int = 1):
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    return Qwen3VLTextAttention(config, layer_index)


def test_qwen3_static_layerwise_attention_accepts_subset_without_final_visual_token():
    attention = _attention()
    hidden = torch.randn(1, 4, 32)
    position_embeddings = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
    output, _ = qwen3_vl_attn_forward(
        attention,
        hidden,
        position_embeddings,
        None,
        vision_token_pruning_mask=torch.tensor([[True, False, True, False]]),
        vision_token_prune_after_layer=0,
    )

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_qwen3_dynamic_layerwise_attention_accepts_query_specific_visual_mask():
    attention = _attention()
    hidden = torch.randn(1, 4, 32)
    position_embeddings = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
    dynamic_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    dynamic_mask[:, 3, 1:] = False
    output, _ = qwen3_vl_attn_forward(
        attention,
        hidden,
        position_embeddings,
        None,
        vision_token_dynamic_attention_mask=dynamic_mask,
        vision_token_prune_after_layer=0,
    )

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_qwen3_combined_static_then_dynamic_mask_matches_explicit_reference():
    torch.manual_seed(17)
    attention = _attention(layer_index=3)
    hidden = torch.randn(1, 5, 32)
    position_embeddings = (torch.ones(1, 5, 8), torch.zeros(1, 5, 8))
    static = torch.tensor([[True, False, True, True, True]])
    dynamic = torch.ones(1, 5, 5, dtype=torch.bool)
    dynamic[:, 4, 2] = False

    combined, _ = qwen3_vl_attn_forward(
        attention,
        hidden,
        position_embeddings,
        None,
        vision_token_pruning_mask=static,
        vision_token_dynamic_attention_mask=dynamic,
        vision_token_prefill_prune_after_layer=0,
        vision_token_decode_prune_after_layer=2,
    )
    reference_dynamic = dynamic & static[:, None, :]
    reference, _ = qwen3_vl_attn_forward(
        attention,
        hidden,
        position_embeddings,
        None,
        vision_token_dynamic_attention_mask=reference_dynamic,
        vision_token_prune_after_layer=2,
    )
    reference[:, ~static[0]] = 0

    torch.testing.assert_close(combined, reference, rtol=0, atol=0)
