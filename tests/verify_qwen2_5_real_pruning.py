import sys
import os
import importlib.util
from types import ModuleType
from unittest.mock import MagicMock

# 1. Stub out ray, tensordict, and verl dependencies to avoid ModuleNotFoundError
sys.modules['ray'] = MagicMock()
sys.modules['tensordict'] = MagicMock()

# Stub verl.utils submodules imported by qwen2_vl.py
mock_compat = ModuleType("transformers_compat")
mock_compat.is_transformers_version_in_range = lambda *args, **kwargs: True
sys.modules['verl.utils.transformers_compat'] = mock_compat

mock_device = ModuleType("device")
mock_device.is_npu_available = lambda: False
sys.modules['verl.utils.device'] = mock_device

mock_ulysses = MagicMock()
sys.modules['verl.utils.ulysses'] = mock_ulysses

# Stub out sequence_compressor to avoid loading package dependencies
mock_compressor = ModuleType("sequence_compressor")
def local_prune_visual_embeddings(embeddings, keep_mask):
    if keep_mask is None:
        return embeddings
    return embeddings[keep_mask]
mock_compressor.prune_visual_embeddings = local_prune_visual_embeddings
sys.modules['verl.models.vision_token_pruning.sequence_compressor'] = mock_compressor

# 2. Dynamically load qwen2_vl.py
spec = importlib.util.spec_from_file_location("qwen2_vl", "verl/models/transformers/qwen2_vl.py")
qwen2_vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen2_vl)
_get_input_embeds = qwen2_vl._get_input_embeds
qwen2_vl_forward = qwen2_vl.qwen2_vl_forward

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

def test_real_qwen2_5_pruning():
    # 3. Create a tiny Qwen2.5-VL configuration for fast CPU instantiation
    print("--- Initializing Miniature Qwen2.5-VL Model ---")
    config = Qwen2_5_VLConfig(
        vocab_size=152000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_parameters={"mrope_section": [2, 3, 3]},
        vision_config={
            "hidden_size": 32,
            "embed_dim": 32,
            "num_hidden_layers": 1,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "tokens_per_second": 2,
        }
    )
    
    # Instantiate the real Transformers model with random weights on CPU
    model = Qwen2_5_VLForConditionalGeneration(config)
    # Bridge the naming differences between Qwen2-VL (which verl expects) and Qwen2.5-VL natively in transformers
    model.visual = model.model.visual
    print("  ✓ Real Model Instantiated Successfully!")

    # 4. Setup inputs
    # Let's say sequence length is: 4 text tokens + 16 image tokens + 4 text tokens = 24 tokens total
    t1, img_len, t2 = 4, 16, 4
    seq_len = t1 + img_len + t2
    
    input_ids = torch.randint(10, 500, (1, seq_len))
    # Replace the middle part with image token id
    image_token_id = model.config.image_token_id
    input_ids[0, t1:t1+img_len] = image_token_id
    
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    position_ids = torch.ones(4, 1, seq_len, dtype=torch.long)
    
    # Qwen2.5-VL visual encoder expects:
    # pixel_values shape: (total_patches, patch_size * patch_size * in_channels) = (128, 14 * 14 * 3)
    # where total_patches = T * H * W = 2 * 8 * 8 = 128 (since grid_thw T=1 block maps to 2 frames)
    # The output merged tokens count will be (H/2) * (W/2) * (T/2*2) = 4 * 4 * 1 = 16 tokens (matches img_len)
    pixel_values = torch.randn(128, 588)
    image_grid_thw = torch.tensor([[1, 8, 8]])
    
    # 5. CASE 1: No Pruning (baseline verification)
    print("\n--- Running Case 1: Real Forward (No Pruning) ---")
    inputs_embeds_unpruned, mask_unpruned = _get_input_embeds(
        model,
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        pixel_values=pixel_values,
        pixel_values_videos=None,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        vision_token_keep_mask=None
    )
    
    assert inputs_embeds_unpruned.shape == (1, seq_len, 64)
    assert mask_unpruned.shape == (1, seq_len)
    print("  ✓ Unpruned embeds shape: ", list(inputs_embeds_unpruned.shape))
    print("  ✓ Case 1 (No Pruning) passed successfully!")

    # 6. CASE 2: 50% Pruning (keep 16, prune 16)
    print("\n--- Running Case 2: Real Forward (50% Pruning) ---")
    keep_mask = torch.ones(img_len, dtype=torch.bool)
    keep_mask[torch.arange(0, img_len, 2)] = False # prune even indices (16 kept, 16 pruned)
    
    inputs_embeds_pruned, mask_pruned = _get_input_embeds(
        model,
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        pixel_values=pixel_values,
        pixel_values_videos=None,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        vision_token_keep_mask=keep_mask
    )
    
    assert inputs_embeds_pruned.shape == (1, seq_len, 64)
    assert mask_pruned.shape == (1, seq_len)
    
    # Verify that the attention_mask is successfully set to 0 for all pruned positions
    pruned_positions = t1 + torch.arange(0, img_len, 2)
    assert torch.all(mask_pruned[0, pruned_positions] == 0)
    print("  ✓ Attention mask set to 0 at pruned positions!")
    
    # Verify that unpruned positions remain 1
    kept_positions = t1 + torch.arange(1, img_len, 2)
    assert torch.all(mask_pruned[0, kept_positions] == 1)
    print("  ✓ Attention mask remains 1 at kept positions!")
    
    print("  ✓ Case 2 (50% Pruning) passed successfully!")

    # 7. Run the actual model forward call using verl wrapping to verify end-to-end integration
    print("\n--- Running Case 3: End-to-End Model Forward with Pruning ---")
    # Wrap model forward method with our custom wrapper
    model.forward = qwen2_vl_forward.__get__(model, type(model))
    
    outputs = model(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        position_ids=position_ids.clone(),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_token_keep_mask=keep_mask
    )
    
    hidden_states = outputs[0]
    assert hidden_states.shape == (1, seq_len, 64)
    print("  ✓ Output hidden_states shape: ", list(hidden_states.shape))
    print("  ✓ End-to-end forward call executed successfully on PyTorch/Transformers layers!")

    print("\n✓ ALL REAL PYTORCH/TRANSFORMERS INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_real_qwen2_5_pruning()
