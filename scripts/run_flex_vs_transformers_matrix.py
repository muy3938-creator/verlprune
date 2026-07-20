#!/usr/bin/env python3
"""Run and aggregate the process-isolated pruning latency matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def gpu_utilization_percent() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return max(int(line.strip()) for line in output.splitlines() if line.strip())


def wait_for_idle(threshold: int, consecutive: int, timeout_s: int) -> None:
    if threshold >= 100:
        return
    started = time.monotonic()
    accepted = 0
    while accepted < consecutive:
        utilization = gpu_utilization_percent()
        accepted = accepted + 1 if utilization <= threshold else 0
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(
                f"GPU did not stay at or below {threshold}% for {consecutive} samples; "
                f"last utilization was {utilization}%"
            )
        if accepted < consecutive:
            time.sleep(1)


def case_name(backend: str, keep_ratio: float, layer: int, batch_size: int) -> str:
    ratio = f"{keep_ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{backend}-r{ratio}-l{layer}-b{batch_size}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("vllm_flex", "transformers_flash"),
        default=["vllm_flex", "transformers_flash"],
    )
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15, 27])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--idle-utilization-threshold", type=int, default=5)
    parser.add_argument("--idle-consecutive-samples", type=int, default=3)
    parser.add_argument("--idle-timeout-s", type=int, default=21600)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).with_name("benchmark_flex_vs_transformers_pruning.py")
    cases = [
        (backend, keep_ratio, layer, batch_size)
        for batch_size in args.batch_sizes
        for layer in args.layers
        for keep_ratio in args.keep_ratios
        for backend in args.backends
    ]
    for index, (backend, keep_ratio, layer, batch_size) in enumerate(cases, 1):
        name = case_name(backend, keep_ratio, layer, batch_size)
        output = args.output_dir / f"{name}.json"
        log = args.output_dir / f"{name}.log"
        if output.exists() and not args.rerun:
            print(f"[{index}/{len(cases)}] skip {name}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] wait-for-idle {name}", flush=True)
        wait_for_idle(
            args.idle_utilization_threshold,
            args.idle_consecutive_samples,
            args.idle_timeout_s,
        )
        command = [
            sys.executable,
            str(benchmark),
            backend,
            "--model",
            args.model,
            "--output",
            str(output),
            "--keep-ratio",
            str(keep_ratio),
            "--prune-after-layer",
            str(layer),
            "--batch-size",
            str(batch_size),
            "--resolution",
            str(args.resolution),
            "--decode-tokens",
            str(args.decode_tokens),
            "--warmup-runs",
            str(args.warmup_runs),
            "--measure-runs",
            str(args.measure_runs),
        ]
        print(f"[{index}/{len(cases)}] run {name}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    records = []
    for backend, keep_ratio, layer, batch_size in cases:
        output = args.output_dir / f"{case_name(backend, keep_ratio, layer, batch_size)}.json"
        if output.exists():
            records.append(json.loads(output.read_text(encoding="utf-8")))
    aggregate = {
        "model": args.model,
        "resolution": args.resolution,
        "decode_tokens": args.decode_tokens,
        "keep_ratios": args.keep_ratios,
        "layers": args.layers,
        "batch_sizes": args.batch_sizes,
        "records": records,
    }
    aggregate_path = args.output_dir / "matrix.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {aggregate_path} with {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
