from types import SimpleNamespace

from verl.utils.fsdp_utils import replace_lora_wrapper


def test_lora_base_layer_names_only_apply_to_configured_text_modules():
    config = SimpleNamespace(
        target_modules=r".*language_model.layers.*(q_proj|gate_proj)",
        exclude_modules=None,
    )

    assert (
        replace_lora_wrapper("model.language_model.layers.0.self_attn.q_proj.weight", config)
        == "model.language_model.layers.0.self_attn.q_proj.base_layer.weight"
    )
    assert (
        replace_lora_wrapper("model.visual.blocks.0.mlp.gate_proj.weight", config)
        == "model.visual.blocks.0.mlp.gate_proj.weight"
    )
