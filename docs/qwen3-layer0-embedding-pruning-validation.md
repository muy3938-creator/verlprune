# Qwen3 layer-0 embedding pruning validation

## Result

The default fused experiment is viable on the dense
`Qwen3-VL-4B-Instruct` model:

```text
visual embedding L2 top-k before decoder layer 0
-> 64 visual tokens physically compacted to 32
-> all decoder layers write the same reduced visual KV set
-> vLLM decode reuses that reduced cache
-> verl actor replays the exact indices
-> backward and optimizer update complete
```

This is the intentionally simple path. It uses FlashAttention after physical
compaction and does not need FlexAttention, a layerwise attention mask, or
decode-time Top-K. `prune_after_layer=-1` is the internal compatibility value;
the user-facing selection boundary is layer 0.

## Dense-model check

The tested model reports `model_type=qwen3_vl`, 36 decoder layers, and no
expert count (`num_experts=None`). It is a dense Qwen3-VL model, not the
`qwen3_vl_moe` variant. The selection metadata temporarily travels through
vLLM 0.18's routed-expert return buffer, but this is only a compatibility
channel and does not turn the model into an MoE.

Qwen3 fast and slow image processors do not choose the same grid for the
224-pixel smoke image: the production fast processor emits 64 merged visual
tokens, while the slow processor emits 49. The smoke and parity tools now use
the same fast processor as verl. The Qwen3 capture payload also carries the
authoritative original token count, so replay validation no longer depends on
a Qwen2.5-specific grid formula.

## H20 end-to-end run

Environment: one NVIDIA H20, PyTorch 2.10.0+cu129, Transformers 4.57.6,
vLLM 0.18.0, BF16, eager vLLM, and FlashAttention actor training.

Rollout configuration:

```text
selector=embedding_norm
selector_input=vision_embedding
keep_ratio=0.5
prune_after_layer=-1
batch_size=1
decode_tokens=8
```

Observed rollout:

```text
image_grid_thw=[1, 16, 16]
visual_tokens=64->32
generated_tokens=8
rollout_seconds=2.305247
output_tokens_per_second=3.470344
E2E_STAGE_ROLLOUT=PASS
```

The actor then ran three real backward/update steps over the exact captured
selection:

```text
full_sequence_tokens=96
student_sequence_tokens=64
step_losses=[0.0002828694, 0.0002828545, 0.0002828543]
step_grad_norms=[0.0223388672, 0.0223388672, 0.0223388672]
parameter_delta_norm=0.0452426337
cuda_peak_allocated_mib=8813.16
E2E_STAGE_TRAIN_STEP=PASS
```

## Rollout/actor numerical parity

An independent run sampled eight tokens with vLLM at temperature 1.0 and
replayed the same token IDs and exact 32-token visual selection through
`DataParallelPPOActor._forward_micro_batch`.

```text
sampled log-prob absolute difference mean = 3.0615e-4
sampled log-prob absolute difference max  = 2.4331e-3
prefill-token absolute difference         = 2.4331e-3
decode-token absolute difference mean     = 2.3000e-6
probability Pearson correlation           = 0.9999962
importance ratios outside [0.9, 1.1]      = 0 / 8
```

The larger first-token difference is normal vLLM-versus-Transformers backend
drift; after the shared reduced KV cache is established, decode replay is
nearly identical. No token exceeded 0.05 log-prob absolute error.

Raw summaries are stored in
`artifacts/qwen3-layer0-embedding-h20/`.

## Maintenance conclusion

For algorithms that can decide from vision embeddings, this layer-0 path is
the preferred experiment surface. A new selector changes only the strategy
function; it does not require modifying vLLM paged attention, per-layer KV
layout, or FlexAttention score modifiers. FlexAttention remains useful only
for delayed-layer or per-query dynamic masking experiments.
