#!/usr/bin/env python3
"""Extract a PEFT LoRA adapter from a single-rank Vision-OPD FSDP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path


DEFAULT_TARGET_MODULES = (
    ".*language_model.layers.*"
    "(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def export_adapter(
    checkpoint: Path,
    output_dir: Path,
    base_model: str,
    *,
    alpha: int,
    target_modules: str,
) -> dict[str, object]:
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter = OrderedDict(
        (name.replace(".default.weight", ".weight"), tensor.contiguous())
        for name, tensor in state.items()
        if "lora_" in name
    )
    if not adapter:
        raise ValueError(f"checkpoint contains no LoRA tensors: {checkpoint}")

    first_tensor = next(iter(adapter.values()))
    rank = min(first_tensor.shape)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    config.base_model_name_or_path = base_model
    config.save_pretrained(output_dir)
    adapter_path = output_dir / "adapter_model.safetensors"
    save_file(adapter, adapter_path, metadata={"format": "pt"})
    with adapter_path.open("rb") as adapter_file:
        digest = hashlib.file_digest(adapter_file, "sha256").hexdigest()
    return {
        "adapter_dir": str(output_dir),
        "tensor_count": len(adapter),
        "rank": rank,
        "alpha": alpha,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    args = parser.parse_args()
    result = export_adapter(
        args.checkpoint,
        args.output_dir,
        args.base_model,
        alpha=args.alpha,
        target_modules=args.target_modules,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
