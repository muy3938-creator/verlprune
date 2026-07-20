#!/usr/bin/env python3
"""Small, reproducible ChartQA benchmark for dense and learned-budget paths.

The benchmark intentionally uses one request at a time so a single GPU can
compare the same samples across dense, random-pruned, top-p-pruned, and
post-training-adapter models. It accepts either a local ChartQA parquet export
or a Hugging Face/datasets repository path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROMPT_TEMPLATE = """You are an expert chart reasoning assistant.
Inspect the chart carefully and solve the question. Reason step by step about
the visual evidence, quantities, comparisons, and any arithmetic needed. Use at
most three short reasoning steps and no more than 60 words before the answer.
End with exactly one final answer inside <answer>...</answer>. Do not put extra
text after the answer.

Question: {question}
"""


def _normalise(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if match:
        text = match.group(1)
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9.\-+/% ]", "", text).strip()


def _numbers(text: str) -> list[float]:
    values = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def chartqa_match(prediction: str, labels: list[str]) -> bool:
    pred = _normalise(prediction)
    targets = [_normalise(label) for label in labels]
    if pred in targets or any(target and target in pred for target in targets):
        return True
    pred_numbers = _numbers(pred)
    for target in targets:
        target_numbers = _numbers(target)
        if len(pred_numbers) == 1 and len(target_numbers) == 1:
            if math.isclose(pred_numbers[0], target_numbers[0], rel_tol=0.05, abs_tol=0.05):
                return True
    return False


def _load_split(source: str, split: str):
    from datasets import load_dataset

    source_path = Path(source)
    if source_path.is_dir():
        pattern = source_path / "data" / f"{split}-*.parquet"
        files = sorted(pattern.parent.glob(pattern.name))
        if not files:
            pattern = source_path / f"{split}.parquet"
            files = [pattern] if pattern.exists() else []
        if not files:
            raise FileNotFoundError(f"no ChartQA parquet files found for {split}: {pattern}")
        return load_dataset("parquet", data_files={split: [str(path) for path in files]}, split=split)
    return load_dataset(source, split=split)


def _image(value):
    from io import BytesIO
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"unsupported ChartQA image value: {type(value).__name__}")


def _build_config(mode: str, keep_ratio: float, top_p: float, layer: int):
    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig

    if mode == "random":
        return VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=keep_ratio,
            prune_after_layer=layer,
            selector="random",
            selector_input="decoder_key",
            layerwise_backend="flex",
            pre_pruning_backend="flash",
        )
    if mode == "top_p":
        return VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=keep_ratio,
            prune_after_layer=layer,
            selector="vision_pulse",
            selector_input="decode_query",
            selector_kwargs={
                "budget_mode": "top_p",
                "top_p": top_p,
                "temperature": 1.0,
                "min_keep_ratio": 0.05,
                "max_keep_ratio": 1.0,
                "capture_capacity": 256,
            },
            layerwise_backend="flex",
            pre_pruning_backend="flash",
        )
    raise ValueError(f"unsupported pruning mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=("baseline", "random", "top_p"), required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--stop-at-answer-tag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dataset = _load_split(args.dataset, args.split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.cuda().eval()

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    generation_kwargs = {}
    if args.stop_at_answer_tag:
        from transformers import StopStringCriteria, StoppingCriteriaList

        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopStringCriteria(processor.tokenizer, ["</answer>"])]
        )
    records = []
    total_correct = 0
    total_seconds = 0.0
    keep_counts = []
    generated_token_counts = []
    max_length_hits = 0
    for index, row in enumerate(dataset):
        image = _image(row["image"])
        question = str(row["query"])
        labels = row.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEMPLATE.format(question=question)}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in inputs.items()}
        state = None
        if args.mode != "baseline":
            from verl.models.vision_token_pruning.transformers_sampler import install_transformers_pruning

            config = _build_config(args.mode, args.keep_ratio, args.top_p, args.layer)
            state = install_transformers_pruning(
                model,
                input_ids=inputs["input_ids"],
                image_token_id=image_token_id,
                config=config,
                post_pruning_backend="flash_varlen",
            )
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                **generation_kwargs,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        generated_tokens = len(generated_ids)
        hit_max_new_tokens = generated_tokens >= args.max_new_tokens
        generated_token_counts.append(generated_tokens)
        max_length_hits += int(hit_max_new_tokens)
        answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        correct = chartqa_match(answer, labels)
        total_correct += int(correct)
        total_seconds += elapsed
        row_result = {
            "index": index,
            "query": question,
            "labels": list(labels),
            "answer": answer,
            "correct": correct,
            "latency_s": elapsed,
            "generated_tokens": generated_tokens,
            "hit_max_new_tokens": hit_max_new_tokens,
        }
        if state is not None:
            selection = state.selections()[0]
            if hasattr(selection, "query_kept_visual_indices"):
                counts = [len(values) for values in selection.query_kept_visual_indices if values]
                row_result["kept_visual_counts"] = counts
                keep_counts.extend(counts)
            else:
                row_result["kept_visual_counts"] = [len(selection.kept_visual_indices)]
                keep_counts.append(len(selection.kept_visual_indices))
        records.append(row_result)
        print(json.dumps({"index": index, "correct": correct, "answer": answer[:160]}, ensure_ascii=False), flush=True)

    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "limit": len(records),
        "mode": args.mode,
        "keep_ratio": args.keep_ratio,
        "top_p": args.top_p,
        "layer": args.layer,
        "accuracy": total_correct / max(1, len(records)),
        "mean_latency_s": total_seconds / max(1, len(records)),
        "mean_kept_visual_tokens": sum(keep_counts) / max(1, len(keep_counts)),
        "mean_generated_tokens": sum(generated_token_counts) / max(1, len(generated_token_counts)),
        "max_generated_tokens": max(generated_token_counts, default=0),
        "max_length_hit_rate": max_length_hits / max(1, len(records)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        run.log(metrics)
        run.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
