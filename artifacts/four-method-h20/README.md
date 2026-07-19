# Four-method H20 validation artifacts

These JSON files are the raw Qwen2.5-VL-3B-Instruct H20 results summarized in
[`docs/four-method-flex-pruning.md`](../../docs/four-method-flex-pruning.md).

For each static method (`dart`, `divprune`, and `greedy_prune`):

- `rollout.json` records the real batch-2 vLLM Flex rollout at layer 15 and 5%;
- `train_step.json` records three actor forward/backward/optimizer steps;
- `logit-parity-8/vllm.json` records vLLM top-token distributions and the
  actual selected indices for DART and GreedyPrune;
- `logit-parity-8/transformers.json` records PyTorch-mask replay;
- `logit-parity-8/comparison.json` records the passing 8-step BF16 comparison.

The DivPrune run requested all 151,936 vocabulary log-probabilities per step.
Its compact comparison and Transformers replay are retained, while the 44 MiB
intermediate vLLM dump is intentionally omitted from Git.

`arbitrary-layer-batch4-10/` contains a second real rollout for every method,
including VisionPulse, with batch size 4 and exact 10% retention. The selected
boundaries are layers 7, 27, 34, and 7 respectively.

VisionPulse's 5% parity and training artifacts are kept in
[`artifacts/dynamic-decode-h20`](../dynamic-decode-h20/README.md), because that
implementation preceded the three static selectors.

Paths inside the JSON files refer to disposable CNB workspaces for provenance.
The artifacts contain no model weights or private data.
