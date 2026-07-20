# H20 hybrid Flash/Flex experiment artifacts

These files back `docs/hybrid-flash-flex-pruning.md`.

- `hybrid-parity-10/`: Hybrid vLLM output, Transformers PyTorch-mask reference,
  and their one-prefill/three-decode-step comparison.
- `flex-parity-10/`: matching all-Flex vLLM output for direct kernel-path comparison.
- `hybrid-bench/l*-r*-b*-{flex,flash}/rollout.json`: two warmups plus five measured
  rollout runs across pruning layers, keep ratios, and batch sizes.
- `hybrid-bench/l23-fixed20-*/rollout.json`: the fair fixed-20-output-token layer-23
  performance pair.
- `hybrid-bench/algorithm-dart/`: real boundary-Q/K/V selection rollout and the
  three-step training smoke result.
- `hybrid-bench/algorithm-visionpulse/`: real per-query dynamic visual-KV rollout.
- `parity-l23-{flash,flex}/`: separate layer-23 PyTorch-reference checks.

The generated image and verbose engine logs are omitted because they are reproducible
from the smoke scripts and contain no additional measurements.
