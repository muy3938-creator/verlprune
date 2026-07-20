#!/usr/bin/env python3
"""Benchmark unpruned Transformers and vLLM multimodal generation.

Run each backend in a separate process. Both paths use the same synthetic
images, rendered chat prompts, target input length, and fixed decode length.
The output is JSON so results can be compared or included in validation docs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    resolution: int
    batch_size: int


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    resolution: int
    batch_size: int
    input_tokens_per_request: int
    visual_tokens_per_request: int
    generated_tokens_per_request: int
    wall_time_s: float
    time_to_first_token_s: float
    decode_ms_per_token: float
    output_tokens_per_s: float
    gpu_memory_baseline_mib: int
    gpu_memory_peak_mib: int
    gpu_memory_delta_mib: int
    cuda_peak_allocated_mib: float | None
    cuda_peak_reserved_mib: float | None


class GPUMemorySampler:
    """Sample total compute-process GPU memory without backend coupling."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.baseline_mib = self._read_mib()
        self.peak_mib = self.baseline_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read_mib() -> int:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return sum(values)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak_mib = max(self.peak_mib, self._read_mib())
            except (OSError, subprocess.SubprocessError, ValueError):
                continue

    def __enter__(self) -> GPUMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self._read_mib())


def _make_image(resolution: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(image)
    margin = max(8, resolution // 10)
    middle = resolution // 2
    draw.rectangle((margin, margin, middle - margin // 2, resolution - margin), fill=(220, 30, 30))
    draw.ellipse((middle + margin // 2, margin, resolution - margin, resolution - margin), fill=(30, 80, 220))
    return image


def _render_prompt(processor: Any, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_case_inputs(
    processor: Any,
    case: BenchmarkCase,
    target_input_tokens: int,
) -> tuple[str, Any, Any, int, int]:
    image = _make_image(case.resolution)
    unit = "Compare the visible shapes, colors, positions, and relationships carefully. "

    def process(repetitions: int, batch_size: int = 1):
        prompt = _render_prompt(processor, unit * repetitions)
        return prompt, processor(
            text=[prompt] * batch_size,
            images=[image] * batch_size,
            padding=True,
            return_tensors="pt",
        )

    low, high = 0, target_input_tokens
    while low < high:
        middle = (low + high + 1) // 2
        _, candidate = process(middle)
        if int(candidate["attention_mask"].sum()) <= target_input_tokens:
            low = middle
        else:
            high = middle - 1

    prompt, inputs = process(low, case.batch_size)
    input_tokens = int(inputs["attention_mask"][0].sum())
    grid = inputs["image_grid_thw"][0]
    merge_size = int(processor.image_processor.merge_size)
    visual_tokens = int(grid.prod().item() // (merge_size**2))
    return prompt, image, inputs, input_tokens, visual_tokens


def _timing_fields(timestamps: list[float], start_time: float, wall_time_s: float) -> tuple[float, float]:
    if not timestamps:
        return wall_time_s, 0.0
    time_to_first_token_s = timestamps[0] - start_time
    if len(timestamps) == 1:
        return time_to_first_token_s, 0.0
    return time_to_first_token_s, (timestamps[-1] - timestamps[0]) * 1000 / (len(timestamps) - 1)


def run_transformers(args: argparse.Namespace, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria

    class TimestampCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.timestamps: list[float] = []

        def __call__(self, *_: object, **__: object) -> bool:
            self.timestamps.append(time.perf_counter())
            return False

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda().eval()

    results = []
    for case_index, case in enumerate(cases):
        _, _, inputs, input_tokens, visual_tokens = _build_case_inputs(
            processor,
            case,
            args.target_input_tokens,
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        if case_index == 0:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    min_new_tokens=2,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()

        criteria = TimestampCriteria()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with GPUMemorySampler() as memory:
            start_time = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    min_new_tokens=args.decode_tokens,
                    max_new_tokens=args.decode_tokens,
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=[criteria],
                )
            torch.cuda.synchronize()
            wall_time_s = time.perf_counter() - start_time

        generated_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
        ttft_s, decode_ms = _timing_fields(criteria.timestamps, start_time, wall_time_s)
        results.append(
            BenchmarkResult(
                backend="transformers",
                resolution=case.resolution,
                batch_size=case.batch_size,
                input_tokens_per_request=input_tokens,
                visual_tokens_per_request=visual_tokens,
                generated_tokens_per_request=generated_tokens,
                wall_time_s=wall_time_s,
                time_to_first_token_s=ttft_s,
                decode_ms_per_token=decode_ms,
                output_tokens_per_s=case.batch_size * generated_tokens / wall_time_s,
                gpu_memory_baseline_mib=memory.baseline_mib,
                gpu_memory_peak_mib=memory.peak_mib,
                gpu_memory_delta_mib=memory.peak_mib - memory.baseline_mib,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        )
    return results


def run_vllm(args: argparse.Namespace, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.target_input_tokens + args.decode_tokens + 64,
        max_num_seqs=max(case.batch_size for case in cases),
        max_num_batched_tokens=args.target_input_tokens * max(case.batch_size for case in cases),
        kv_cache_memory_bytes=args.vllm_kv_cache_mib * 1024 * 1024,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=False,
        disable_log_stats=False,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    sampling = SamplingParams(
        temperature=0.0,
        min_tokens=args.decode_tokens,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
    )
    results = []
    for case_index, case in enumerate(cases):
        prompt, image, _, expected_input_tokens, visual_tokens = _build_case_inputs(
            processor,
            case,
            args.target_input_tokens,
        )
        requests = [{"prompt": prompt, "multi_modal_data": {"image": image}} for _ in range(case.batch_size)]
        if case_index == 0:
            llm.generate(requests[:1], SamplingParams(temperature=0.0, max_tokens=2))

        with GPUMemorySampler() as memory:
            start_time = time.perf_counter()
            outputs = llm.generate(requests, sampling)
            wall_time_s = time.perf_counter() - start_time

        metrics = [output.metrics for output in outputs]
        ttft_values = [metric.first_token_latency for metric in metrics]
        decode_values = [
            (metric.last_token_ts - metric.first_token_ts)
            * 1000
            / max(1, len(output.outputs[0].token_ids) - 1)
            for metric, output in zip(metrics, outputs, strict=True)
        ]
        generated_tokens = len(outputs[0].outputs[0].token_ids)
        input_tokens = len(outputs[0].prompt_token_ids)
        if abs(input_tokens - expected_input_tokens) > 1:
            raise RuntimeError(
                f"processor/vLLM input length mismatch: {expected_input_tokens} != {input_tokens}"
            )
        results.append(
            BenchmarkResult(
                backend="vllm",
                resolution=case.resolution,
                batch_size=case.batch_size,
                input_tokens_per_request=input_tokens,
                visual_tokens_per_request=visual_tokens,
                generated_tokens_per_request=generated_tokens,
                wall_time_s=wall_time_s,
                time_to_first_token_s=statistics.mean(ttft_values),
                decode_ms_per_token=statistics.mean(decode_values),
                output_tokens_per_s=case.batch_size * generated_tokens / wall_time_s,
                gpu_memory_baseline_mib=memory.baseline_mib,
                gpu_memory_peak_mib=memory.peak_mib,
                gpu_memory_delta_mib=memory.peak_mib - memory.baseline_mib,
                # vLLM runs the engine in another process, so torch's parent
                # allocator counters cannot observe its allocations.
                cuda_peak_allocated_mib=None,
                cuda_peak_reserved_mib=None,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("transformers", "vllm"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[224, 448, 896])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--target-input-tokens", type=int, default=1900)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--vllm-kv-cache-mib", type=int, default=512)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    cases = [
        BenchmarkCase(resolution, batch_size)
        for resolution in args.resolutions
        for batch_size in args.batch_sizes
    ]
    runner = run_transformers if args.backend == "transformers" else run_vllm
    payload = {
        "model": args.model,
        "backend": args.backend,
        "target_input_tokens": args.target_input_tokens,
        "decode_tokens": args.decode_tokens,
        "vllm_kv_cache_mib": args.vllm_kv_cache_mib if args.backend == "vllm" else None,
        "vllm_gpu_memory_utilization": (
            args.vllm_gpu_memory_utilization if args.backend == "vllm" else None
        ),
        "results": [asdict(result) for result in runner(args, cases)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
