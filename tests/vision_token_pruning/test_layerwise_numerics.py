"""GPU numerical checks for the layerwise Transformers training path."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("layerwise FlashAttention numerics require CUDA", allow_module_level=True)


from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLTextModel

from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward


def _tiny_qwen_text_model() -> Qwen2_5_VLTextModel:
    config = Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [2, 2, 2]},
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_2"
    return Qwen2_5_VLTextModel(config).cuda().to(torch.bfloat16).eval()


def _run_forward_backward(model, input_ids, position_ids, keep_mask, prune_after_layer):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=position_ids,
        vision_token_pruning_mask=keep_mask,
        vision_token_prune_after_layer=prune_after_layer,
    ).last_hidden_state
    loss = output[:, keep_mask[0]].float().square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), loss.detach(), gradients


@pytest.mark.parametrize("prune_after_layer", [0, 1, 2])
def test_compacted_attention_matches_transformers_attention_mask_forward_and_backward(prune_after_layer):
    """Physical Q/K/V compaction must be identical to the normal HF mask path.

    The reference delegates every layer to the unmodified Transformers
    attention.  Starting after the selected boundary it supplies the replayed
    2-D keep mask through Transformers' regular FlashAttention mask handling.
    The implementation under test instead gathers retained Q/K/V tensors and
    scatters the output back into the stable hidden-state shape.
    """

    torch.manual_seed(7)
    model = _tiny_qwen_text_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 9), device="cuda")
    keep_mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 1, 1]], dtype=torch.bool, device="cuda")
    base_positions = torch.arange(input_ids.shape[1], device="cuda").view(1, 1, -1)
    # Row 0 is the packed-text position used to delimit causal sequences; the
    # remaining rows are Qwen's temporal/height/width MRoPE positions.
    position_ids = base_positions.expand(4, 1, -1).clone()

    original_forward = Qwen2_5_VLAttention.forward

    def reference_forward(self, hidden_states, attention_mask=None, **kwargs):
        pruning_mask = kwargs.pop("vision_token_pruning_mask", None)
        boundary = kwargs.pop("vision_token_prune_after_layer", None)
        if pruning_mask is not None and boundary is not None and self.layer_idx > int(boundary):
            attention_mask = pruning_mask
        return original_forward(self, hidden_states, attention_mask=attention_mask, **kwargs)

    try:
        Qwen2_5_VLAttention.forward = reference_forward
        reference_output, reference_loss, reference_gradients = _run_forward_backward(
            model, input_ids, position_ids, keep_mask, prune_after_layer
        )

        Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
        compact_output, compact_loss, compact_gradients = _run_forward_backward(
            model, input_ids, position_ids, keep_mask, prune_after_layer
        )
    finally:
        Qwen2_5_VLAttention.forward = original_forward

    retained = keep_mask[0]
    torch.testing.assert_close(compact_output[:, retained], reference_output[:, retained], rtol=0, atol=0)
    torch.testing.assert_close(compact_loss, reference_loss, rtol=0, atol=0)
    assert compact_gradients.keys() == reference_gradients.keys()
    for name in compact_gradients:
        torch.testing.assert_close(
            compact_gradients[name],
            reference_gradients[name],
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )
