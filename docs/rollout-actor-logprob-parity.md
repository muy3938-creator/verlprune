# vLLM rollout versus verl actor replay numerical parity

## Question

After vLLM performs arbitrary-layer visual-token pruning, does the verl actor
assign materially different probabilities when it replays the same selection
and sampled response during training?

The relevant quantity is not only whether greedy tokens match. For every token
sampled by vLLM, this experiment compares:

```text
rollout_log_prob = log pi_vLLM(sampled_token | prefix, rollout selection)
actor_log_prob   = log pi_actor(sampled_token | prefix, replayed selection)
```

The token-level importance ratio is `exp(actor_log_prob - rollout_log_prob)`.

## Method

The experiment uses Qwen2.5-VL-3B-Instruct in BF16 on an NVIDIA H20 with
PyTorch 2.10.0+cu129, vLLM 0.18.0, and Transformers 4.57.6.

The rollout stage uses temperature 1.0, no top-p/top-k truncation, fixed response
lengths, and records the sampled token's vLLM log-probability. The actor stage:

1. loads a fresh copy of the same checkpoint;
2. attaches the exact serialized rollout selection to every sample;
3. appends the exact sampled response tokens;
4. invokes `DataParallelPPOActor._forward_micro_batch`;
5. uses the real `use_remove_padding=True` actor path and obtains all response
   log-probabilities from one full-sequence forward.

This is stronger than a custom step-by-step Transformers reference because it
exercises the same selection validation, packing, multimodal input extraction,
position handling, and response-log-prob extraction used by actor training.

The matrix contains 26 cases and 2,464 sampled tokens:

- native unpruned vLLM baseline: batch 4/8, response length 16/32/64, two seeds;
- all-Flex and hybrid Flash/Flex controls at layers 7/15/23 and 5%/10% retention;
- DART, DivPrune, GreedyPrune, and dynamic per-query VisionPulse;
- hybrid batch 1/4/8, response length 16/32/64, and three sampling seeds.

## The unpruned baseline

The baseline is the original Qwen vLLM architecture with its native
FlashAttention backend. It does not install a pruning architecture override,
does not route through the Vision-OPD selector, and does not approximate the
baseline with a near-100% keep ratio.

| Group | Tokens | Mean abs log-prob diff | P99 | Max | K3 estimate | Ratio outside [0.8, 1.2] | Probability correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native unpruned baseline | 640 | 0.03062 | 0.17988 | 0.35276 | 0.001372 | 0.78% | 0.999640 |
| Core all-Flex pruning | 384 | 0.02744 | 0.19013 | 0.79666 | 0.001836 | 1.04% | 0.999629 |
| Core hybrid Flash/Flex | 384 | 0.02790 | 0.15611 | 0.41487 | 0.001281 | 0.26% | 0.999618 |
| Expanded hybrid matrix | 1,440 | 0.02884 | 0.19179 | 0.41487 | 0.001387 | 0.97% | 0.999605 |
| Four paper algorithms, hybrid | 256 | 0.02719 | 0.20585 | 0.24954 | 0.001304 | 1.17% | 0.999634 |

The expanded hybrid mean is 5.84% lower than the unpruned baseline. Its P99 is
6.63% higher, while its K3 estimate is only 1.10% higher. These differences are
small relative to run-to-run and token-tail variation. The hybrid path therefore
does not introduce a systematic rollout/actor mismatch beyond the mismatch that
already exists between native vLLM and the Transformers actor.

The core comparison is favorable to the hybrid implementation. Against the
matched all-Flex matrix, hybrid has a similar mean but lower P99, lower K3, a
smaller maximum outlier, and fewer ratios outside PPO's `[0.8, 1.2]` interval.
This is expected because the actor also uses FlashAttention before the pruning
boundary, while an all-Flex rollout uses a different kernel in every decoder
layer.

## Layer and keep-ratio controls

Each row contains 64 sampled tokens from batch 4 and response length 16.

| Layer | Keep | Rollout prefix backend | Mean abs diff | Max abs diff | K3 | Ratio outside [0.8, 1.2] |
|---:|---:|---|---:|---:|---:|---:|
| 7 | 5% | Flex | 0.03427 | 0.13882 | 0.001548 | 0.00% |
| 7 | 5% | Flash | 0.03220 | 0.22127 | 0.001462 | 0.00% |
| 7 | 10% | Flex | 0.03692 | 0.79666 | 0.005122 | 4.69% |
| 7 | 10% | Flash | 0.02965 | 0.12732 | 0.001218 | 0.00% |
| 15 | 5% | Flex | 0.02326 | 0.12473 | 0.000785 | 0.00% |
| 15 | 5% | Flash | 0.02426 | 0.12461 | 0.000822 | 0.00% |
| 15 | 10% | Flex | 0.02184 | 0.12483 | 0.000849 | 0.00% |
| 15 | 10% | Flash | 0.02699 | 0.19179 | 0.001284 | 0.00% |
| 23 | 5% | Flex | 0.02272 | 0.12428 | 0.000795 | 0.00% |
| 23 | 5% | Flash | 0.02769 | 0.13378 | 0.000965 | 0.00% |
| 23 | 10% | Flex | 0.02563 | 0.41366 | 0.001915 | 1.56% |
| 23 | 10% | Flash | 0.02657 | 0.41487 | 0.001934 | 1.56% |

There is no monotonic error increase with later pruning layers or more aggressive
5% retention. The largest all-Flex outlier occurs at layer 7/10%, while most
neighboring cases remain near 0.12. This indicates kernel-dependent BF16 tail
variation rather than a broken layer boundary or wrong replayed selection.

## Selector controls

All rows use layer 15, 5% retention, batch 4, response length 16, and the hybrid
backend.

| Selector | Mean abs diff | Max abs diff | K3 | Ratio outside [0.8, 1.2] |
|---|---:|---:|---:|---:|
| DART | 0.03652 | 0.24954 | 0.002186 | 3.13% |
| DivPrune | 0.02498 | 0.12477 | 0.000910 | 0.00% |
| GreedyPrune | 0.02133 | 0.14753 | 0.000784 | 0.00% |
| VisionPulse dynamic KV | 0.02595 | 0.24429 | 0.001336 | 1.56% |

All four selectors complete real actor replay. Dynamic VisionPulse therefore
does not expose an indexing or query-row alignment failure; its error remains in
the same range as static algorithms and the native baseline.

## Prefill, decode, and sequence accumulation

For the expanded hybrid matrix:

- first sampled token (prefill result) mean absolute difference: `0.01363`;
- later decode tokens mean absolute difference: `0.02955`;
- mean token importance ratio: `1.00045`;
- sampled direct-KL estimate: `0.000936`;
- K3 estimate: `0.001387`;
- mean absolute sequence log-ratio: `0.16772`;
- maximum absolute sequence log-ratio: `0.53491`.

The mean importance ratio is effectively unbiased. Decode contributes more
numerical drift than the first prefill token, which is consistent with repeated
KV-cache kernel execution. Sequence-level products accumulate token-level noise,
so PPO and rollout-correction code should use token-level clipping/metrics rather
than infer safety from a long-sequence product alone.

## Training implications

The policies are not bitwise identical, including in the native unpruned
baseline. The measured hybrid discrepancy is small enough for experimentation:

- 99.03% of expanded-hybrid token ratios remain inside `[0.8, 1.2]`;
- probability correlation is above `0.9996`;
- mean ratio and sampled KL estimates show no systematic policy shift.

The trainer currently reuses `rollout_log_probs` as `old_log_probs` when the step
is exactly one PPO mini-batch and one PPO epoch. In that mode, this backend gap
does not become the initial PPO denominator. When old log-probabilities are
recomputed, the gap is real and should remain visible through the existing
rollout-correction diagnostics. For multi-epoch or stricter experiments, retain
token-level IS clipping/rejection and alert on P99/max drift instead of requiring
exact equality.

## Conclusion

The modified vLLM rollout and the verl actor replay are numerically consistent at
the level expected from the native vLLM/Transformers baseline. Hybrid
FlashAttention-before-pruning does not materially enlarge the average or K3 gap,
and it improves several tail metrics relative to all-Flex rollout. It is safe to
keep as an experimental backend, with explicit monitoring for rare BF16 outliers.

The reusable test driver is `scripts/rollout_actor_replay_parity.py`. Raw rollout,
actor, per-case comparison, and pooled summary JSON files are stored under
`artifacts/rollout-actor-parity-h20/`.
