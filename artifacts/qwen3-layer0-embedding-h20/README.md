# Raw Qwen3 layer-0 H20 summaries

These files were produced on one NVIDIA H20 with dense
Qwen3-VL-4B-Instruct, PyTorch 2.10.0+cu129, Transformers 4.57.6, and vLLM
0.18.0. See `docs/qwen3-layer0-embedding-pruning-validation.md` for commands,
interpretation, and limitations.

- `rollout.json`: physical 50% visual-token rollout summary.
- `train_step.json`: three actor backward/update steps.
- `parity.json`: vLLM sampled-log-prob versus real verl actor replay metrics.
