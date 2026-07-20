# Arbitrary-layer two-stage vision-token pruning validation

Last updated: 2026-07-20 (Asia/Shanghai)

## Goal and acceptance criteria

This report is updated as each test group finishes. The target is a two-stage
experiment whose first static visual subset and second per-query decode subset
can use independently configured decoder boundaries, while keeping all vLLM
changes in the out-of-tree plugin.

Numerical correctness means that vLLM rollout and the Transformers actor replay
the exact same recorded token indices and masks. The principal measurements are
greedy-token agreement and sampled-token log-probability error. A successful
algorithm test also requires finite forward values; end-to-end cases require a
finite non-zero gradient and parameter update.

Planned coverage:

- dense Qwen2.5-VL-3B and dense Qwen3-VL-4B;
- the five experiment families: embedding/static baseline, DART, DivPrune,
  GreedyPrune, and VisionPulse/two-stage dynamic routing;
- physical first-stage boundary `-1` and delayed boundaries throughout the
  decoder, with the decode boundary equal to or later than the first boundary;
- first- and second-stage ratios down to 5%;
- batches larger than one where GPU memory permits.

## Implemented boundary semantics

The public configuration is:

```yaml
prefill_keep_ratio: 0.5
prefill_prune_after_layer: 7
prune_after_layer: 15
keep_ratio: 0.5
```

`prefill_prune_after_layer=-1` preserves the previous physical mode: the first
subset is selected from visual embeddings and physically compacted before
decoder layer 0. A non-negative value keeps the complete prompt and paged KV
cache; layers through that boundary see all visual tokens, later layers apply a
static FlexAttention mask, and layers after `prune_after_layer` additionally
apply the VisionPulse per-query mask within the first subset.

The delayed mode intentionally does not claim KV-memory reduction. Supporting
different physical KV lengths at arbitrary decoder layers would require invasive
changes to vLLM's paged-cache allocator, which conflicts with this experiment's
maintenance objective.

## Completed local validation

### Configuration and replay tests — PASS

- legacy physical `-1 -> N` configurations remain valid;
- delayed `L1 -> L2` requires `0 <= L1 <= L2`;
- delayed launch preserves all prompt placeholders and KV entries. It passes a
  `1e-9` metadata-hook sentinel because vLLM skips Qwen's embedding
  postprocessor at exact zero; the custom processor/model still return the
  complete visual sequence;
- the actor keeps the full sequence, applies a static first-stage mask after
  `L1`, and combines it with the query-specific second-stage mask after `L2`;
- second-stage indices remain relative to the first-stage subset;
- dropped first-stage visual keys remain masked after the dynamic boundary;
- dense Qwen3 combined-mask output matches an explicit PyTorch mask reference
  exactly in the deterministic CPU test.

Local suite result:

```text
976 passed, 6 skipped in 3.19s
```

The six skips are CUDA/vLLM integration tests and will be replaced by real GPU
results below. Python bytecode compilation and whitespace checks also passed.

### Large deterministic algorithm matrix — PASS

A new 864-case matrix covers:

- five method families: embedding-norm baseline, DART, DivPrune,
  GreedyPrune, and fixed-budget VisionPulse;
- boundaries `0, 1, 7, 15, 27, 34`;
- retention ratios `5%, 10%, 25%, 50%, 75%, 95%`;
- visual lengths `20, 64, 257`;
- two-stage boundary pairs `-1->0`, `-1->15`, `0->0`, `0->1`, `0->15`,
  `7->15`, `15->15`, `15->27`, and `27->34`;
- every cross-product of the six first-stage and six second-stage ratios.

Every selector returned a deterministic, sorted, unique, in-range exact-K
subset. VisionPulse probabilities and attention mass stayed finite in all
cases. The two-stage matrix additionally verified physical versus delayed mode
classification and boundary ordering.

```text
864 passed in 1.64s
```

### Complete CUDA test suite — PASS

The native-Flex lifecycle suite first passed on H20. After adding the delayed
composition case it reports five passes. The complete CUDA-capable pruning
suite then passed on a later workspace:

```text
995 passed, 1 skipped, 16 warnings in 15.42s
```

This covers Flash-through-boundary fallback, static physical-slot persistence,
per-query dynamic re-selection, delayed static-plus-dynamic slot composition,
and Qwen2.5 forward/backward equality against the explicit Transformers mask at
multiple layer boundaries. During this run a Transformers 4.57.6 compatibility
issue was found and fixed: Qwen2.5 attention now reads MRoPE configuration from
`self.config.rope_scaling` when the attention module no longer copies it to a
`self.rope_scaling` attribute.

## Real H20 validation

Environment: PyTorch 2.10.0+cu129, vLLM 0.18.0, Transformers 4.57.6,
BF16, TP=1, eager FlexAttention.

### Qwen2.5-VL-3B delayed two-stage — PASS

Representative case: first-stage boundary 7, decode boundary 15, 50% then
50%.

```text
visual route             = 64 -> 32 -> 16/query
sampled tokens compared  = 8
log-prob abs diff mean   = 0.00932434
log-prob abs diff max    = 0.04785663
probability Pearson      = 0.99950185
ratio outside [0.9,1.1] = 0 / 8
training steps           = 3
gradient norms           = [0.06689, 0.06299, 0.06299]
parameter delta norm     = 0.0258950
peak actor allocation    = 7474.99 MiB
```

The extreme same-anchor case also passed with batch 2:

```text
prefill boundary = 0
decode boundary  = 0
ratios           = 5% then 5%
visual route     = 64 -> 3 -> 1/query
requests         = 2/2 pass, four decode queries each
```

The existing Qwen2.5 four-method H20 suite remains green for DART, DivPrune,
GreedyPrune, and VisionPulse at 5%, with method-specific Transformers parity
and three actor update steps. Together with the embedding baseline this covers
all five method families on Qwen2.5.

### Dense Qwen3-VL-4B all-method rollout — PASS

| Method | Boundary or pair | Ratio | Actual visual set | Result |
|---|---:|---:|---:|---|
| embedding norm | 0 | 5% | 3/64 | pass |
| DART | 7 | 5% | 3/64 | pass |
| DivPrune | 15 | 10% | 6/64 | pass |
| GreedyPrune | 27 | 10% | 6/64 | pass |
| delayed VisionPulse | 7 -> 15 | 50% -> 50% | 64 -> 32 -> 16/query | pass |
| hybrid Flash/Flex VisionPulse | 15 -> 27 | 10% -> 10% | 64 -> 6 -> 1/query | pass |

No implementation forces the final visual token. For example, the Qwen3
embedding-norm 5% case selected `[7, 18, 42]`; DART and GreedyPrune happened to
select token 63 from their own scores.

Qwen3 `7 -> 15`, 50%-then-50% rollout/actor parity:

```text
sampled tokens compared  = 8
log-prob abs diff mean   = 0.000258662
log-prob abs diff max    = 0.00205786
probability Pearson      = 0.99989291
ratio outside [0.9,1.1] = 0 / 8
```

Both delayed training cases completed three updates:

| Pair | Ratios | Gradient evidence | Parameter delta | Peak allocation |
|---|---:|---:|---:|---:|
| 7 -> 15 | 50% -> 50% | `3.87e-6`, finite/non-zero | 0.00214664 | 8819.95 MiB |
| 15 -> 27, hybrid Flash/Flex | 10% -> 10% | 1.21 / 1.20 / 1.20 | 0.0466822 | 8819.95 MiB |

The static Qwen3 method runs establish model/runtime compatibility and exact-K
transport. The full method-specific log-prob parity matrix was already run on
Qwen2.5; on Qwen3 the selector-independent actor replay path is directly
covered by the VisionPulse parity case and exact PyTorch mask tests.

Raw retained files are in `artifacts/arbitrary-layer-two-stage-h20/`.

### Resource interruption log

The first arbitrary-boundary Qwen2.5 run (`7 -> 15`, `50% -> 50%`) started on
a shared H20 that already used 66 GiB. External occupancy rose to 86 GiB during
model initialization and the CNB workspace was then terminated by the platform.
No `rollout.json` was produced, so this attempt is recorded as **resource
interrupted**, neither pass nor implementation failure. A replacement workspace
was requested immediately.

A second and third workspace landed on the same heavily shared H20 pool. They
also lacked the editable/target installation that registers the vLLM plugin
entry point, so those early exits cannot be attributed solely to memory. Later
workspaces explicitly registered the plugin and produced the passing results
above. Small CUDA numerical tests remained stable throughout.

One real integration defect was found during this process: exact
`video_pruning_rate=0` made vLLM skip Qwen's EVS postprocessing hook, so delayed
mode retained the full prompt but lost selection metadata. The final
implementation uses a `1e-9` hook sentinel while its custom processor and model
still preserve every token. Debug traces then showed 64 metadata rows, 32
first-stage candidates, and successful static/dynamic boundaries.

## Known performance interpretation

Delayed arbitrary-layer pruning is designed for algorithm iteration and exact
replay, not immediate speedup. It leaves the full paged KV cache allocated and
uses eager FlexAttention score modifiers. Wall-clock acceleration would require
a compiled sparse gather/paged kernel after the algorithm is stable.

Observed single-case generation rates are therefore diagnostic only: Qwen2.5
`7 -> 15` produced about 0.24 token/s in debug mode; Qwen3 `7 -> 15` about 0.55
token/s; the deeper hybrid Qwen3 `15 -> 27` case about 0.93 token/s. These runs
had different output lengths and shared-GPU load and must not be treated as a
backend benchmark.
