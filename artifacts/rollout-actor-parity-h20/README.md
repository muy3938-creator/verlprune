# H20 rollout-to-actor parity artifacts

This directory contains the 26-case, 2,464-token experiment documented in
`docs/rollout-actor-logprob-parity.md`.

`actor-parity-matrix/summary.json` contains pooled groups and all per-case
metrics. Every case directory contains:

- `rollout.json`: sampled token IDs, vLLM sampled-token log-probabilities, and
  the exact serialized visual-token selection;
- `actor.json`: log-probabilities returned by the real
  `DataParallelPPOActor._forward_micro_batch` remove-padding path;
- `comparison.json`: token, sequence, KL-proxy, importance-ratio, and
  probability-correlation metrics.

The four `baseline-native-*` directories are genuine unpruned vLLM runs using
the model's default architecture and attention backend. Generated PNGs and
verbose engine logs are omitted because the script deterministically recreates
them and they contain no additional numerical measurements.
