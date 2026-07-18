from pathlib import Path


def test_layer0_physical_pruning_is_wired_before_actor_unpadding():
    root = Path(__file__).resolve().parents[2]
    plugin = (root / "verl/vllm_plugins/vision_token_pruning.py").read_text()
    actor = (root / "verl/workers/actor/dp_actor.py").read_text()
    qwen25 = (root / "verl/models/transformers/qwen2_vl.py").read_text()
    qwen3 = (root / "verl/models/transformers/qwen3_vl.py").read_text()

    assert "embeddings[kept_indices]" in plugin
    assert "supports_multimodal_pruning = True" in plugin
    assert "update.replacement = pruned_replacement" in plugin
    pending_capture = plugin.index("self._pending_capture_values = capture_values")
    forward_capture = plugin.index("capturer.capture(0, capture_values)")
    assert pending_capture < forward_capture
    assert "if capturer is not None:" in plugin
    assert actor.index("apply_rollout_pruning_to_attention_mask(") < actor.index("unpad_input(")
    assert "apply_vision_token_pruning=False" in actor
    assert "prune_visual_embeddings" in qwen25
    assert "prune_visual_embeddings" in qwen3
    assert "mask_visual_embeddings" not in qwen25 + qwen3
