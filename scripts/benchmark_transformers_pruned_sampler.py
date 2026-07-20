#!/usr/bin/env python3
"""Benchmark cache-aware Transformers sampling with layerwise visual pruning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def make_image(resolution: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(image)
    margin = max(8, resolution // 10)
    middle = resolution // 2
    draw.rectangle(
        (margin, margin, middle - margin // 2, resolution - margin),
        fill=(220, 30, 30),
    )
    draw.ellipse(
        (middle + margin // 2, margin, resolution - margin, resolution - margin),
        fill=(30, 80, 220),
    )
    return image


def selector_input(selector: str) -> str:
    return "decode_query" if selector == "vision_pulse" else "decoder_key"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--selector-kwargs", type=json.loads, default={})
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--prune-after-layer", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.transformers_sampler import (
        install_transformers_pruning,
    )

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    image = make_image(args.resolution)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Describe the two colored shapes from left to right in one short sentence.",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt] * args.batch_size,
        images=[image] * args.batch_size,
        padding=True,
        return_tensors="pt",
    )
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_tokens = int(inputs["input_ids"][0].eq(image_token_id).sum().item())
    prompt_tokens = int(inputs["attention_mask"][0].sum().item())

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda().eval()
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=args.keep_ratio,
        selector=args.selector,
        selector_kwargs=args.selector_kwargs,
        prune_after_layer=args.prune_after_layer,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input=selector_input(args.selector),
    )
    state = install_transformers_pruning(
        model,
        input_ids=inputs["input_ids"],
        image_token_id=image_token_id,
        config=config,
    )
    cuda_inputs = {key: value.cuda() for key, value in inputs.items()}
    generate_kwargs = dict(
        **cuda_inputs,
        min_new_tokens=args.decode_tokens,
        max_new_tokens=args.decode_tokens,
        do_sample=False,
        use_cache=True,
    )
    for _ in range(args.warmup_runs):
        with torch.inference_mode():
            model.generate(**generate_kwargs)
    torch.cuda.synchronize()
    latencies = []
    output_ids = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    for _ in range(args.measure_runs):
        if config.uses_dynamic_decode_selection:
            state.dynamic_rows = [
                [tuple() for _ in range(state.prompt_length)]
                for _ in range(args.batch_size)
            ]
            state.current_dynamic_indices = None
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**generate_kwargs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    assert output_ids is not None
    selections = state.selections()
    generated = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
    total_seconds = sum(latencies)
    if config.uses_dynamic_decode_selection:
        keep_counts = [len(row) for row in selections[0].query_kept_visual_indices]
    else:
        keep_counts = [len(selections[0].kept_visual_indices)]
    payload = {
        "backend": "transformers_cache",
        "selector": args.selector,
        "selector_kwargs": args.selector_kwargs,
        "keep_ratio": args.keep_ratio,
        "prune_after_layer": args.prune_after_layer,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "prompt_tokens_per_request": prompt_tokens,
        "visual_tokens_per_request": visual_tokens,
        "generated_tokens_per_request": generated,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "latencies_s": latencies,
        "mean_latency_s": total_seconds / len(latencies),
        "output_tokens_per_s": args.batch_size * generated * len(latencies) / total_seconds,
        "cuda_baseline_allocated_mib": baseline_allocated / 2**20,
        "cuda_baseline_reserved_mib": baseline_reserved / 2**20,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "kept_visual_counts": keep_counts,
        "generated_token_ids": output_ids[0, -generated:].cpu().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
