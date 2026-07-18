# Layerwise visual-token pruning validation

## Scope

The experimental layerwise mode extends the validated random-selection
baseline for Qwen2.5-VL. `prune_after_layer=N` keeps the complete sequence
through decoder layer `N`, then excludes the rollout-selected visual tokens
from attention and KV-cache work beginning with layer `N + 1`. The OPD teacher
still receives the complete sequence and image.

The training path retains a stable hidden-state shape, gathers only retained
Q/K/V rows in later attention layers, and scatters attention outputs back into
that shape. The vLLM 0.18 backend compacts each request's later-layer logical KV
layout while retaining the scheduler's original block tables.

## Numerical reference

`tests/vision_token_pruning/test_layerwise_numerics.py` compares the compacted
training path against unmodified Transformers FlashAttention supplied with the
same ordinary 2-D attention mask. The test uses a four-layer Qwen2.5-VL text
model in BF16 on CUDA and checks boundaries 0, 1, and 2.

For every boundary, the retained final hidden states, scalar loss, and every
parameter gradient were bitwise equal (`rtol=0`, `atol=0`). This covers token
embeddings and all attention, MLP, layer-norm, and final-norm parameters.

## Defect found by real vLLM execution

The initial implementation passed mocked backend tests but failed its first
real prefill with:

```text
RuntimeError: cu_seqlens_q must have dtype torch.int32
```

PyTorch promotes an `int32` tensor to `int64` in `torch.cumsum` unless the
output dtype is explicit. The compact metadata builder now performs
`cumsum(..., dtype=torch.int32)`, and a regression test asserts the exact dtype
required by FlashAttention.

## GPU validation

Validation ran on an NVIDIA H20 with PyTorch 2.10.0, Transformers 4.57.6,
vLLM 0.18.0, FlashAttention, and Qwen2.5-VL-3B-Instruct. The model was cloned
from the CNB internal mirror:

```text
https://cnb.cool/ai-models/Qwen/Qwen2.5-VL-3B-Instruct
```

With `keep_ratio=0.5` and `prune_after_layer=15`:

- a direct vLLM prefill plus 18-token decode completed, retaining 32 of 64
  image tokens and returning the exact retained indices;
- a two-request batched rollout also completed: both requests retained 32 of
  64 image tokens at layer boundary 15, decoded 18 tokens, and returned valid
  per-request selection metadata;
- three direct actor optimization steps completed with finite loss and
  gradients, changing the selected parameter by norm `0.0256164`;
- the full Ray + FSDP + vLLM + AgentLoop + OPD-teacher + rank-8 LoRA training
  path completed 3 of 3 global steps and saved its final checkpoint;
- global-step gradient norms were `0.0321222`, `0.0422009`, and `0.0456496`;
- VOPD losses were `0.00106664`, `0.000523576`, and `0.00167211`;
- policy fallback and empty-target fractions remained zero, while teacher and
  teacher-image-swap fractions remained one;
- peak allocated GPU memory was approximately 23.17 GiB;
- all 252 LoRA-B tensors were nonzero after training (8,257,536 nonzero
  elements, combined norm `0.0527589`), confirming optimizer updates rather
  than only successful forward/backward calls.

The focused suite reported `39 passed, 1 skipped`, and Ruff passed on the
pruning runtime, plugins, tests, and GPU smoke script.

## Current constraints

- Layerwise rollout is implemented for Qwen2.5-VL only.
- Each pruned request contains exactly one image and no video.
- Ulysses sequence parallelism must remain disabled (size 1).
- vLLM chunked prefill, prefix caching, compilation, and CUDA graphs are
  disabled for this backend; eager execution is required.
- The selection metadata still uses vLLM's routed-expert return channel, so
  routing replay cannot be enabled simultaneously.
- On the tested CNB PyTorch 2.10 image, asynchronous FSDP GPU-to-CPU parameter
  offload failed with a CUDA invalid-argument error before any pruning forward.
  Disabling actor/ref parameter offload and optimizer offload avoided that
  environment-specific issue and used about 23.17 GiB peak allocated memory.
