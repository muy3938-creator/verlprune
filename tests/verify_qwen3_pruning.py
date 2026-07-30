import sys
import os
import importlib.util
from types import ModuleType
from unittest.mock import MagicMock

# 1. Stub out ray, tensordict, and verl dependencies to avoid ModuleNotFoundError
sys.modules['ray'] = MagicMock()
sys.modules['tensordict'] = MagicMock()

# Stub verl.utils submodules imported by qwen3_vl.py
mock_compat = ModuleType("transformers_compat")
mock_compat.unpack_visual_output = lambda x: x
sys.modules['verl.utils.transformers_compat'] = mock_compat

# Dynamically load the real sequence compressor module and bind it to sys.modules
spec_comp = importlib.util.spec_from_file_location("verl.models.vision_token_pruning.sequence_compressor", "verl/models/vision_token_pruning/sequence_compressor.py")
sequence_compressor = importlib.util.module_from_spec(spec_comp)
sys.modules['verl.models.vision_token_pruning.sequence_compressor'] = sequence_compressor
sys.modules['sequence_compressor'] = sequence_compressor
spec_comp.loader.exec_module(sequence_compressor)
prune_visual_embedding_outputs = sequence_compressor.prune_visual_embedding_outputs

# 2. Dynamically load qwen3_vl.py
spec = importlib.util.spec_from_file_location("qwen3_vl", "verl/models/transformers/qwen3_vl.py")
qwen3_vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3_vl)
_get_input_embeds = qwen3_vl._get_input_embeds

import torch
import torch.nn as nn

# 3. Create a real PyTorch module structure representing the Qwen3-VL layers
class RealPyTorchVisionConfig:
    def __init__(self):
        self.in_channels = 3
        self.temporal_patch_size = 2
        self.patch_size = 14

class RealPyTorchConfig:
    def __init__(self):
        self.image_token_id = 151655
        self.vision_config = RealPyTorchVisionConfig()

class RealPyTorchVisionTransformer(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        # Use a real linear projection layer to project patch pixels (588) to model hidden_size (16)
        self.proj = nn.Linear(588, hidden_size)
        self.dtype = torch.float32

    def forward(self, pixel_values, grid_thw=None):
        # pixel_values shape: (128, 588)
        # Perform real linear projection and pool/merge patches to visual tokens:
        # In Qwen3-VL, 128 patches are merged 2x2 spatially and 2x temporally to 16 tokens
        features = self.proj(pixel_values) # (128, 16)
        # Reshape and average pool to simulate spatial-temporal merging (128 -> 16 tokens)
        primary_embeds = features.view(16, 8, 16).mean(dim=1) # (16, 16)
        
        # Simulate DeepStack multi-layered features (e.g. outputs of 4 intermediate layers)
        deepstack_layers = [
            (features * (i + 1)).view(16, 8, 16).mean(dim=1)
            for i in range(4)
        ]
        return primary_embeds, deepstack_layers

class RealPyTorchModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = RealPyTorchConfig()
        self.visual = RealPyTorchVisionTransformer(hidden_size)
        self.embeddings = nn.Embedding(152000, hidden_size)
        
    def get_input_embeddings(self):
        return self.embeddings

def test_qwen3_pruning():
    print("--- Initializing Real PyTorch Qwen3 Mockup ---")
    model = RealPyTorchModel(hidden_size=16)
    print("  ✓ Real PyTorch Model layers instantiated successfully!")
    
    # 4. Generate inputs representing a batch of 1
    # Layout: 4 text tokens + 16 image tokens + 4 text tokens = 24 tokens total
    t1, img_len, t2 = 4, 16, 4
    seq_len = t1 + img_len + t2
    
    input_ids = torch.randint(10, 500, (1, seq_len))
    image_start = t1
    image_end = t1 + img_len
    input_ids[0, image_start:image_end] = model.config.image_token_id
    
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    
    # Real random pixel values (128 patches * 588 features)
    pixel_values = torch.randn(128, 588)
    image_grid_thw = torch.tensor([[1, 8, 8]])
    
    # CASE 1: No pruning (vision_token_keep_mask is None)
    print("\n--- Running Case 1: Real PyTorch Forward (No Pruning) ---")
    out_unpruned = _get_input_embeds(
        model,
        input_ids.clone(),
        attention_mask=attention_mask.clone(),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_token_keep_mask=None
    )
    
    assert out_unpruned["inputs_embeds"].shape == (1, seq_len, 16)
    assert out_unpruned["attention_mask"].shape == (1, seq_len)
    assert out_unpruned["deepstack_visual_embeds"] is not None
    assert len(out_unpruned["deepstack_visual_embeds"]) == 4
    for emb in out_unpruned["deepstack_visual_embeds"]:
        assert emb.shape == (img_len, 16)
    print("  ✓ Unpruned embeds shape: ", list(out_unpruned["inputs_embeds"].shape))
    print("  ✓ Case 1 passed successfully!")
    
    # CASE 2: With 50% pruning (keep 8, prune 8)
    print("\n--- Running Case 2: Real PyTorch Forward (50% Pruning) ---")
    keep_mask = torch.ones(img_len, dtype=torch.bool)
    keep_mask[torch.arange(0, img_len, 2)] = False  # Prune even indices (8 pruned, 8 kept)
    
    # In CASE 2, the inputs representing the student sequence will have pruned tokens set to 0 in attention_mask
    student_attention_mask = attention_mask.clone()
    student_attention_mask[0, image_start + torch.arange(0, img_len, 2)] = 0
    
    student_input_ids = input_ids.clone()
    
    out_pruned = _get_input_embeds(
        model,
        student_input_ids,
        attention_mask=student_attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        vision_token_keep_mask=keep_mask
    )
    
    # Verify shape alignments
    assert out_pruned["inputs_embeds"].shape == (1, seq_len, 16)
    assert out_pruned["attention_mask"].shape == (1, seq_len)
    
    # Attention mask at pruned positions in the returned dict should be 0
    pruned_positions = image_start + torch.arange(0, img_len, 2)
    assert torch.all(out_pruned["attention_mask"][0, pruned_positions] == 0)
    print("  ✓ attention_mask successfully modified at pruned positions in returned output!")
    
    # Verify DeepStack auxiliary features are correctly pruned using the real sequence compressor
    assert len(out_pruned["deepstack_visual_embeds"]) == 4
    for emb in out_pruned["deepstack_visual_embeds"]:
        assert emb.shape == (8, 16)
    print("  ✓ DeepStack visual embeddings successfully pruned (16 -> 8)!")
    
    print("\n✓ ALL REAL PYTORCH QWEN3 PRUNING VERIFICATION TESTS PASSED!")

if __name__ == "__main__":
    test_qwen3_pruning()
