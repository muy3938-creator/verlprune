# All-Flash Transformers versus Hybrid vLLM H20 artifacts

These JSON files back `docs/flex-vs-transformers-1000-token-benchmark.md`.

- `flash-varlen-matrix-v1024/`: 12 clean batch-1 records for 1,024 visual
  tokens, 50%/10%/5% retention, boundaries 0/15, and both backends.
- `flash-varlen-matrix-v2025/`: the matching 12 records for 2,025 visual
  tokens.
- `flash-varlen-benchmark-clean/`: five-repeat layer-15/10% confirmation
  records for both lengths and both backends.

Environment: NVIDIA H20 96 GB, Qwen2.5-VL-3B-Instruct BF16, PyTorch
2.10.0+cu129, FlashAttention 2.8.3, Transformers 4.57.6, and vLLM 0.18.0.
Model loading is excluded.  Every request generated exactly 32 tokens with KV
caching.  The matrix records report zero device utilization before backend
startup.

The Transformers late path gathers retained Q/K/V and calls
`flash_attn_varlen_func`; its full `DynamicCache` allocation is preserved.
The vLLM path uses native FlashAttention through the boundary and FlexAttention
`score_mod` afterwards without compacting paged KV.
