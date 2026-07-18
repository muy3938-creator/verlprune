# Next-stage research: decode-time KV pruning

This document records the follow-up task without expanding the staged-final
training architecture.

## Question

After visual-token selection during prefill, can decoder layers after a chosen
boundary attend to only a selected subset (for example 50%) of the prompt KV
cache, while keeping all generated-token KV entries, and obtain a useful
latency/memory improvement at contexts up to approximately 2K tokens?

## Required semantics

1. Select cache entries once per request with a deterministic seed or explicit
   algorithm and preserve the final visual MRoPE anchor.
2. Never randomly resample the cache independently at every decode step.
3. Keep every generated token so autoregressive history remains complete.
4. Record exact retained prompt indices and the decoder-layer boundary in a
   versioned protocol.
5. Apply the identical indices in the Transformers numerical reference and in
   vLLM; fail early if either side cannot replay them.

## Numerical gates

- Compare prefill retained hidden states against unmodified Transformers using
  an ordinary 2-D attention mask.
- For every decode step, compare top-k token IDs and log probabilities against
  the same Transformers masked-attention reference, and report maximum and
  mean absolute error rather than claiming bitwise equality across engines.
- Verify single-request and continuously batched decode, prompt lengths near
  512/1K/2K, multiple image resolutions, and pruning boundaries near the
  beginning/middle/end of the decoder.
- Verify KV physical slots, sequence lengths, block reuse, request completion,
  and absence of stale per-request pruning state.

## Performance gates

Measure TTFT, decode milliseconds/token, output tokens/second, peak GPU memory,
and vLLM startup/compile cost for keep ratios 1.0, 0.75, and 0.5. Compare:

1. Transformers reference;
2. standard compiled vLLM without pruning;
3. layer-0 physical pruning with standard vLLM;
4. layerwise prefill/KV pruning with eager vLLM;
5. the proposed decode-time prompt-KV subset.

Adopt the feature only if the end-to-end decode gain survives the loss of CUDA
graphs/compilation and is material for the actual verl batch/concurrency. Keep
it as a research plugin otherwise.

## Framework risks

- vLLM's paged KV layout and attention metadata are private, version-sensitive
  interfaces in the current implementation.
- Prefix caching, chunked prefill, cascade attention, dual-batch overlap, and
  request block reuse can invalidate a cache map keyed only by physical block.
- Random per-step dropping creates noisy, irreproducible model semantics.
- A second actor-side selector would reintroduce rollout/training divergence;
  exact rollout metadata must remain the source of truth.
- Reduced KV attention does not reduce vision encoding or decoder MLP work, so
  theoretical token reduction is not the same as wall-clock speedup.
