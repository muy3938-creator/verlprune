import sys
import importlib.util
from types import ModuleType
from unittest.mock import MagicMock

# 1. Stub out modules imported by qwen3_vl.py to prevent import failures
mock_modeling = ModuleType("modeling_qwen3_vl")

class DummyClass:
    pass

mock_modeling.Qwen3VLCausalLMOutputWithPast = DummyClass
mock_modeling.Qwen3VLForConditionalGeneration = DummyClass

sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = mock_modeling
sys.modules['verl.utils.transformers_compat'] = MagicMock()

# 2. Dynamically load qwen3_vl.py
spec = importlib.util.spec_from_file_location("qwen3_vl", "verl/models/transformers/qwen3_vl.py")
qwen3_vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3_vl)
get_rope_index = qwen3_vl.get_rope_index

import torch

# Simple mock processor
class MockImageProcessor:
    def __init__(self, merge_size=2):
        self.merge_size = merge_size

class MockProcessor:
    def __init__(self):
        self.image_processor = MockImageProcessor()
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652

# Basic implementation of unpad_input
def mock_unpad_input(input_ids, attention_mask):
    # input_ids: (B, S, 1) or (B, S)
    # attention_mask: (B, S)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    flat_input_ids = input_ids.flatten()
    input_ids_rmpad = flat_input_ids[indices]
    
    # Calculate cu_seqlens
    seqlens = attention_mask.sum(dim=-1)
    cu_seqlens = torch.zeros(seqlens.numel() + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    
    return input_ids_rmpad, indices, cu_seqlens

def mock_index_first_axis(tensor, indices):
    return tensor[indices]

def test_batch_unpad():
    processor = MockProcessor()
    
    # Batch size = 4, each with a different sequence layout
    # Format: (text1_len, image_len, text2_len)
    batch_configs = [
        (4, 240, 21),  # Total unpruned = 265
        (6, 240, 15),  # Total unpruned = 261
        (8, 240, 30),  # Total unpruned = 278
        (2, 240, 10),  # Total unpruned = 252
    ]
    
    max_seq_len = 278
    
    batched_input_ids = []
    batched_attention_mask = []
    batched_position_ids = []
    
    # Generate coordinates and stack with padding
    for i, (t1, img_len, t2) in enumerate(batch_configs):
        unpruned_len = t1 + img_len + t2
        
        # 1. input_ids
        input_ids = torch.arange(unpruned_len)
        image_start = t1
        image_end = t1 + img_len
        input_ids[image_start - 1] = processor.vision_start_token_id
        input_ids[image_start:image_end] = processor.image_token_id
        
        # 2. attention_mask with 50% image tokens pruned
        attention_mask = torch.ones(unpruned_len, dtype=torch.long)
        # Drop even indices
        attention_mask[image_start + torch.arange(0, img_len, 2)] = 0
        
        # 3. get_rope_index wrapper
        vision_token_keep_mask = torch.ones(img_len, dtype=torch.bool)
        vision_token_keep_mask[torch.arange(0, img_len, 2)] = False
        
        image_grid_thw = torch.tensor([[1, 24, 40]])  # h // 2 = 12, w // 2 = 20 -> 240 tokens
        
        pos_ids = get_rope_index(
            processor,
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            vision_token_keep_mask=vision_token_keep_mask
        )
        
        # 4. Right padding to max_seq_len
        pad_len = max_seq_len - unpruned_len
        
        input_ids_padded = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        attention_mask_padded = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
        
        # position_ids padded (standard padding values are 1)
        pos_ids_padded = torch.cat([pos_ids, torch.ones(3, pad_len, dtype=torch.long)], dim=1)
        
        batched_input_ids.append(input_ids_padded)
        batched_attention_mask.append(attention_mask_padded)
        batched_position_ids.append(pos_ids_padded)
        
    batched_input_ids = torch.stack(batched_input_ids)          # (4, 278)
    batched_attention_mask = torch.stack(batched_attention_mask)  # (4, 278)
    batched_position_ids = torch.stack(batched_position_ids)      # (4, 3, 278)
    
    print("Batched Input IDs shape:", batched_input_ids.shape)
    print("Batched Attention Mask shape:", batched_attention_mask.shape)
    print("Batched Position IDs shape:", batched_position_ids.shape)
    
    # Simulate unpad_input (just like dp_actor.py)
    input_ids_rmpad, indices, cu_seqlens = mock_unpad_input(batched_input_ids, batched_attention_mask)
    
    # Transpose position_ids to align with dp_actor.py: (bsz, 3, seq_len) -> (3, bsz, seq_len)
    position_ids = batched_position_ids.transpose(0, 1)
    
    # Flatten position IDs along batch and seq dimensions and index it
    # c b s -> (b s) c
    rearranged_pos = position_ids.permute(1, 2, 0).reshape(-1, 3)
    position_ids_rmpad = mock_index_first_axis(rearranged_pos, indices).transpose(0, 1)
    
    print("\nTotal active tokens across batch (total_nnz):", input_ids_rmpad.shape[0])
    print("Unpadded Position IDs shape:", position_ids_rmpad.shape)
    print("Cumulative sequence lengths (cu_seqlens):", cu_seqlens.tolist())
    
    # Assertions
    assert position_ids_rmpad.shape == (3, input_ids_rmpad.shape[0])
    
    # Let's verify each sample's sliced coordinates inside position_ids_rmpad
    for sample_idx, (t1, img_len, t2) in enumerate(batch_configs):
        start_idx = cu_seqlens[sample_idx].item()
        end_idx = cu_seqlens[sample_idx + 1].item()
        
        sample_pos = position_ids_rmpad[:, start_idx:end_idx]
        print(f"\n--- Checking Sample {sample_idx + 1} (active tokens: {end_idx - start_idx}) ---")
        
        # 1. Text 1 starts at 0
        assert torch.all(sample_pos[:, :t1] == torch.arange(t1))
        print(f"  ✓ Text 1 coordinate check passed (0..{t1-1})")
        
        # 2. Text 2 starts exactly after image bounds (which is offset by unpruned max)
        # Expected offset is t1 + llm_grid_w (which is 20)
        expected_text2_start = t1 + 20
        
        visual_keep_count = img_len // 2
        text2_start_in_sample = t1 + visual_keep_count
        
        print(f"  Sample pos at text2_start_in_sample ({text2_start_in_sample}):", sample_pos[:, text2_start_in_sample])
        print(f"  Expected text2 start:", expected_text2_start)
        assert sample_pos[0, text2_start_in_sample].item() == expected_text2_start
        print(f"  ✓ Text 2 start coordinate check passed (exactly {expected_text2_start})")
        
    print("\n✓ ALL BATCH UNPAD VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_batch_unpad()
