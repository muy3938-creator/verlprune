# Final H20 validation artifacts

The files beside this manifest are local copies from the completed CNB H20
run. They are intentionally ignored by Git, especially the 57 MiB LoRA
adapter. The staged branch keeps this manifest and the reproducible commands
instead of committing generated weights.

| File | SHA-256 |
|---|---|
| `train.log` | `d0296ef54a4adb0633b22f9da6ba47f9bd9af1293006f5b658dcf6163e047b0b` |
| `rollouts/1.jsonl` | `fe90594518d7fc817d0253323109dd64ef1371cfe1f7d221d63208afa97902ed` |
| `rollouts/2.jsonl` | `6cd3fcfc3a1920258dd4056a67b39ba7f5bc4d72bc9a7aa2e3cd5f0cad66230a` |
| `rollouts/3.jsonl` | `f3b6208b1bcde08e114c01ae5938e10a0809d3edeaf6735e873ef79be9906b58` |
| `global_step_3_lora_adapter/adapter_config.json` | `dc15e97e607f63766a424027767de86f3e8bc3b3975b7bfee644705db651959b` |
| `global_step_3_lora_adapter/adapter_model.safetensors` | `c821e0be332e0d98120a521fc2b9565c1ab326d9c372f2facc4fc5319ea7eede` |

The saved adapter contains 252 LoRA-A and 252 LoRA-B tensors. All 8,257,536
LoRA-B elements are nonzero; their combined float32 norm is `0.0620626062`.
