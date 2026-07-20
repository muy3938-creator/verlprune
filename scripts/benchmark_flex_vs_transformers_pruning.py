#!/usr/bin/env python3
"""Benchmark layerwise visual-token pruning in vLLM and Transformers.

Each invocation loads exactly one backend and runs one pruning configuration.
Keeping cases process-isolated avoids retaining a vLLM worker or a Transformers
model while the other backend is measured.  The JSON output records TTFT
(prefill), post-first-token decode time, end-to-end latency, and memory.

The comparison is intentionally explicit about the kernels in use:

* ``vllm_flex`` uses native vLLM FlashAttention through the pruning boundary
  and the repository's masked FlexAttention adapter afterwards.
* ``transformers_flash`` is the current cache-aware reference adapter: it uses
  Hugging Face FlashAttention 2 through the pruning boundary and PyTorch SDPA
  with a boolean KV mask afterwards.  This is an adapter limitation, not a
  FlashAttention limitation; the actor/training path demonstrates that packed
  variable-length Q/K/V can keep FlashAttention active after pruning.

Neither late-layer path physically compacts the KV cache.  This script is a
latency benchmark for the current research implementations, not a claim that
the two late-layer kernels perform structured sparse reads.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


# The public config deliberately rejects enabled keep_ratio=1.0.  A value this
# close to one rounds to the full visual-token count for all practical prompt
# sizes while still exercising the exact same adapter and anchor-layer path.
FULL_KEEP_CONFIG_RATIO = 1.0 - 1e-9


class GPUMemorySampler:
    """Track device-wide compute memory while preserving the pre-run baseline."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.baseline_mib = self._read_memory_mib()
        self.peak_mib = self.baseline_mib
        self.pre_run_utilization_percent = self._read_utilization_percent()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _run_nvidia_smi(query: str) -> list[str]:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    @classmethod
    def _read_memory_mib(cls) -> int:
        return sum(int(value) for value in cls._run_nvidia_smi("used_memory"))

    @staticmethod
    def _read_utilization_percent() -> int:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values, default=0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak_mib = max(self.peak_mib, self._read_memory_mib())
            except (OSError, subprocess.SubprocessError, ValueError):
                continue

    def __enter__(self) -> "GPUMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self._read_memory_mib())


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


def build_inputs(processor: Any, resolution: int, batch_size: int):
    image = make_image(resolution)
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
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt] * batch_size,
        images=[image] * batch_size,
        padding=True,
        return_tensors="pt",
    )
    image_token_id = getattr(
        processor,
        "image_token_id",
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    )
    visual_tokens = int(inputs["input_ids"][0].eq(image_token_id).sum().item())
    prompt_tokens = int(inputs["attention_mask"][0].sum().item())
    if visual_tokens <= 0:
        raise RuntimeError("processor produced no visual tokens")
    return prompt, image, inputs, image_token_id, prompt_tokens, visual_tokens


def _config_keep_ratio(requested: float) -> float:
    return FULL_KEEP_CONFIG_RATIO if requested == 1.0 else requested


def _summary(samples: list[dict[str, float]]) -> dict[str, float]:
    fields = (
        "e2e_latency_s",
        "prefill_ttft_s",
        "decode_latency_s",
        "decode_ms_per_token",
        "output_tokens_per_s",
    )
    output: dict[str, float] = {}
    for field in fields:
        values = [sample[field] for sample in samples]
        output[f"median_{field}"] = statistics.median(values)
        output[f"mean_{field}"] = statistics.mean(values)
        output[f"min_{field}"] = min(values)
        output[f"max_{field}"] = max(values)
    return output


def _timing_sample(
    *,
    wall_time_s: float,
    ttft_s: float,
    batch_size: int,
    generated_tokens_per_request: int,
) -> dict[str, float]:
    ttft_s = min(max(ttft_s, 0.0), wall_time_s)
    decode_latency_s = wall_time_s - ttft_s
    decode_steps = max(1, generated_tokens_per_request - 1)
    return {
        "e2e_latency_s": wall_time_s,
        "prefill_ttft_s": ttft_s,
        "decode_latency_s": decode_latency_s,
        "decode_ms_per_token": decode_latency_s * 1000 / decode_steps,
        "output_tokens_per_s": batch_size * generated_tokens_per_request / wall_time_s,
    }


def run_transformers(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.transformers_sampler import install_transformers_pruning

    class FirstTokenTimestamp(StoppingCriteria):
        def __init__(self) -> None:
            self.first_token_time: float | None = None

        def __call__(self, *_: object, **__: object) -> bool:
            if self.first_token_time is None:
                # HF generation consumes the sampled token before evaluating
                # stopping criteria, so this is the first-token completion.
                self.first_token_time = time.perf_counter()
            return False

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    prompt, image, inputs, image_token_id, prompt_tokens, visual_tokens = build_inputs(
        processor,
        args.resolution,
        args.batch_size,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda().eval()
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=_config_keep_ratio(args.keep_ratio),
        selector="uniform",
        prune_after_layer=args.prune_after_layer,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decoder_key",
    )
    state = install_transformers_pruning(
        model,
        input_ids=inputs["input_ids"],
        image_token_id=image_token_id,
        config=config,
    )
    cuda_inputs = {key: value.cuda() for key, value in inputs.items()}

    def generate_once() -> tuple[Any, float, float]:
        # Recompute the nominal anchor selection for each independent request.
        state.static_indices = None
        timestamp = FirstTokenTimestamp()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **cuda_inputs,
                min_new_tokens=args.decode_tokens,
                max_new_tokens=args.decode_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[timestamp],
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        first = timestamp.first_token_time
        return output, elapsed, elapsed if first is None else first - started

    for _ in range(args.warmup_runs):
        generate_once()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    output_ids = None
    with GPUMemorySampler() as memory:
        for _ in range(args.measure_runs):
            output_ids, elapsed, ttft = generate_once()
            generated = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
            samples.append(
                _timing_sample(
                    wall_time_s=elapsed,
                    ttft_s=ttft,
                    batch_size=args.batch_size,
                    generated_tokens_per_request=generated,
                )
            )
    assert output_ids is not None
    generated = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
    selections = state.selections()
    kept_counts = [len(selection.kept_visual_indices) for selection in selections]
    return {
        "backend": "transformers_flash",
        "early_attention_backend": "flash_attention_2",
        "late_attention_backend": "pytorch_sdpa_boolean_mask",
        "physical_kv_compaction": False,
        "prompt": prompt,
        "image_resolution": args.resolution,
        "prompt_tokens_per_request": prompt_tokens,
        "visual_tokens_per_request": visual_tokens,
        "kept_visual_tokens_per_request": kept_counts,
        "generated_tokens_per_request": generated,
        "generated_token_ids_first_request": output_ids[0, -generated:].cpu().tolist(),
        "samples": samples,
        "summary": _summary(samples),
        "gpu_memory_baseline_mib": memory.baseline_mib,
        "gpu_memory_peak_mib": memory.peak_mib,
        "gpu_memory_delta_mib": memory.peak_mib - memory.baseline_mib,
        "pre_run_gpu_utilization_percent": memory.pre_run_utilization_percent,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PLUGINS", "vision_opd_token_pruning")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.transport import decode_vllm_selection_capture

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    prompt, image, _, _, prompt_tokens, visual_tokens = build_inputs(
        processor,
        args.resolution,
        args.batch_size,
    )
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=_config_keep_ratio(args.keep_ratio),
        selector="uniform",
        prune_after_layer=args.prune_after_layer,
        layerwise_backend="flex",
        pre_pruning_backend="flash",
        selector_input="decoder_key",
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=visual_tokens + args.decode_tokens + 256,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(2048, args.batch_size * (prompt_tokens + 64)),
        kv_cache_memory_bytes=args.vllm_kv_cache_mib * 1024 * 1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_return_routed_experts=True,
        disable_log_stats=False,
        video_pruning_rate=1.0 - config.keep_ratio,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        attention_config={"backend": "FLEX_ATTENTION"},
        hf_overrides={
            "architectures": ["VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration"],
            "text_config": {"num_experts_per_tok": 1},
            "vision_token_pruning": config.to_backend_payload(),
        },
    )
    requests = [
        {"prompt": prompt, "multi_modal_data": {"image": image}}
        for _ in range(args.batch_size)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        min_tokens=args.decode_tokens,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
    )
    for _ in range(args.warmup_runs):
        llm.generate(requests, sampling)

    samples = []
    outputs = None
    with GPUMemorySampler() as memory:
        for _ in range(args.measure_runs):
            started = time.perf_counter()
            outputs = llm.generate(requests, sampling)
            elapsed = time.perf_counter() - started
            generated = len(outputs[0].outputs[0].token_ids)
            # The batch prefill phase ends when its slowest request produces a
            # first token.  This makes e2e - TTFT a useful batch decode phase.
            ttft = max(output.metrics.first_token_latency for output in outputs)
            samples.append(
                _timing_sample(
                    wall_time_s=elapsed,
                    ttft_s=ttft,
                    batch_size=args.batch_size,
                    generated_tokens_per_request=generated,
                )
            )
    assert outputs is not None
    generated = len(outputs[0].outputs[0].token_ids)
    kept_counts = []
    for output in outputs:
        result = output.outputs[0]
        selection = decode_vllm_selection_capture(
            result.routed_experts,
            keep_ratio=config.keep_ratio,
            original_visual_token_count=visual_tokens,
            selector="uniform",
            selector_kwargs={},
        )
        kept_counts.append(len(selection.kept_visual_indices))
    return {
        "backend": "vllm_flex",
        "early_attention_backend": "vllm_flash_attention",
        "late_attention_backend": "vllm_flex_attention_score_mod",
        "physical_kv_compaction": False,
        "prompt": prompt,
        "image_resolution": args.resolution,
        "prompt_tokens_per_request": prompt_tokens,
        "visual_tokens_per_request": visual_tokens,
        "kept_visual_tokens_per_request": kept_counts,
        "generated_tokens_per_request": generated,
        "generated_token_ids_first_request": list(outputs[0].outputs[0].token_ids),
        "samples": samples,
        "summary": _summary(samples),
        "gpu_memory_baseline_mib": memory.baseline_mib,
        "gpu_memory_peak_mib": memory.peak_mib,
        "gpu_memory_delta_mib": memory.peak_mib - memory.baseline_mib,
        "pre_run_gpu_utilization_percent": memory.pre_run_utilization_percent,
        # vLLM's worker owns a separate allocator.
        "cuda_peak_allocated_mib": None,
        "cuda_peak_reserved_mib": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("vllm_flex", "transformers_flash"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--prune-after-layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--vllm-kv-cache-mib", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.keep_ratio <= 1.0:
        parser.error("--keep-ratio must be in (0, 1]")
    if args.prune_after_layer < 0:
        parser.error("--prune-after-layer must be non-negative")
    if args.batch_size <= 0 or args.decode_tokens <= 1:
        parser.error("--batch-size must be positive and --decode-tokens must exceed one")
    if args.warmup_runs < 0 or args.measure_runs <= 0:
        parser.error("warmup runs must be non-negative and measure runs must be positive")

    # Capture contention and resident memory before importing either backend or
    # loading model weights.  The post-warmup sampler below cannot distinguish
    # our own just-finished warmup from an unrelated process.
    host_preload_gpu_utilization_percent = GPUMemorySampler._read_utilization_percent()
    host_preload_gpu_memory_mib = GPUMemorySampler._read_memory_mib()
    runner = run_vllm if args.backend == "vllm_flex" else run_transformers
    backend_result = runner(args)
    payload = {
        "model": args.model,
        "requested_keep_ratio": args.keep_ratio,
        "config_keep_ratio": _config_keep_ratio(args.keep_ratio),
        "prune_after_layer": args.prune_after_layer,
        "boundary_semantics": (
            f"layers 0..{args.prune_after_layer} use the early backend; "
            f"layers {args.prune_after_layer + 1}+ use the masked late backend"
        ),
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "host_preload_gpu_utilization_percent": host_preload_gpu_utilization_percent,
        "host_preload_gpu_memory_mib": host_preload_gpu_memory_mib,
        **backend_result,
    }
    payload["model_resident_gpu_memory_delta_mib"] = (
        payload["gpu_memory_baseline_mib"] - host_preload_gpu_memory_mib
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
