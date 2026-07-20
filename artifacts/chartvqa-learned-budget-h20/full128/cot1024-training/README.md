# Expanded-CoT 1024-token training artifacts

- `1.jsonl` through `10.jsonl` are the ten pure-dynamic rollout records.
- `training.log` contains the complete 10-step run and W&B run `xxtxeu1i`.
- `benchmark-matrix.log` contains the dense and untrained top-p 16-row runs.
  Its final traceback records the expected failure when PEFT reads verl's raw
  checkpoint `target_modules` as a character list.
- `../trained-top-p-080-cot1024-16.json` is the successful trained result. It
  was produced after `scripts/export_lora_adapter.py` converted the actor state
  into a standard rank-8 PEFT adapter.

The exported adapter is outside Git at:

```text
/Users/test/Desktop/formalresearch/opd_mllm/learned-token-pruning-research-20260720/chartvqa-trained-adapter-cot1024-top080-10step
```

SHA-256:

```text
af963f41220241e24f67eac42c5ccb39151bbf86a9aefbc06a2d4cfc653ecaa3
```
