# Dynamic decode H20 artifacts

These are raw outputs from the Qwen2.5-VL-3B-Instruct H20 validation described
in [`docs/dynamic-decode-visionpulse-flex.md`](../../docs/dynamic-decode-visionpulse-flex.md).

- `fixed-05/rollout.json`: fixed 5% per-query selection and generated output;
- `fixed-05/train_step.json`: three real actor optimizer steps replaying that
  routing history;
- `visual-mass-tau04/rollout.json`: VisionPulse dynamic budgets at
  temperature 0.4;
- `logit-parity-8/vllm.json`: vLLM tokens, log-probabilities, and routing;
- `logit-parity-8/transformers.json`: Transformers replay results;
- `logit-parity-8/comparison.json`: the passing 0.08 BF16 comparison.

The files contain paths from the disposable remote workspace for provenance;
they do not contain model weights or private data.
