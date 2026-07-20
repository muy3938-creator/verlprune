import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("varlen FlashAttention numerics require CUDA", allow_module_level=True)

flash_attn = pytest.importorskip("flash_attn")

from verl.models.vision_token_pruning.transformers_sampler import _pruned_flash_varlen


def _reference(query, key, value, keep, scale):
    batch_size, query_heads, query_length, head_dim = query.shape
    output = query.new_zeros((batch_size, query_length, query_heads, head_dim))
    query_keep = keep[:, -query_length:]
    for batch_index in range(batch_size):
        compact_query = query[batch_index, :, query_keep[batch_index]].unsqueeze(0)
        compact_key = key[batch_index, :, keep[batch_index]].unsqueeze(0)
        compact_value = value[batch_index, :, keep[batch_index]].unsqueeze(0)
        query_positions = torch.arange(
            compact_key.shape[2] - compact_query.shape[2],
            compact_key.shape[2],
            device=query.device,
        )
        key_positions = torch.arange(compact_key.shape[2], device=query.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        compact_output = F.scaled_dot_product_attention(
            compact_query,
            compact_key,
            compact_value,
            attn_mask=allowed[None, None],
            scale=scale,
            enable_gqa=query_heads != key.shape[1],
        ).squeeze(0).transpose(0, 1)
        output[batch_index, query_keep[batch_index]] = compact_output
    return output


@pytest.mark.parametrize("query_length,key_length", [(8, 8), (1, 9)])
def test_real_varlen_flash_matches_compacted_sdpa_for_prefill_and_decode(
    query_length,
    key_length,
):
    torch.manual_seed(4)
    query = torch.randn(
        2,
        4,
        query_length,
        64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(2, 2, key_length, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    keep = torch.ones(2, key_length, dtype=torch.bool, device="cuda")
    keep[0, 2] = False
    keep[0, 5] = False
    keep[1, 1] = False
    keep[1, 4] = False
    # The final row is the current decode query and must always remain active.
    keep[:, -1] = True

    actual = _pruned_flash_varlen(
        query,
        key,
        value,
        keep,
        attention_mask=None,
        scaling=0.125,
        dropout=0.0,
        flash_attn_varlen_func=flash_attn.flash_attn_varlen_func,
    )
    expected = _reference(query, key, value, keep, 0.125)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
