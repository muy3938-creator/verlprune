import importlib.util
from pathlib import Path

import pytest

from verl.models.vision_token_pruning.protocol import compute_keep_count


def _script_module(name: str):
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_keep_diagnostic_exercises_adapter_without_dropping_tokens():
    benchmark = _script_module("benchmark_flex_vs_transformers_pruning")

    ratio = benchmark._config_keep_ratio(1.0)
    assert ratio < 1.0
    assert compute_keep_count(1024, ratio) == 1024


def test_timing_sample_splits_prefill_and_decode():
    benchmark = _script_module("benchmark_flex_vs_transformers_pruning")

    sample = benchmark._timing_sample(
        wall_time_s=2.0,
        ttft_s=0.45,
        batch_size=4,
        generated_tokens_per_request=32,
    )

    assert sample["prefill_ttft_s"] == pytest.approx(0.45)
    assert sample["decode_latency_s"] == pytest.approx(1.55)
    assert sample["decode_ms_per_token"] == pytest.approx(50.0)
    assert sample["output_tokens_per_s"] == pytest.approx(64.0)


def test_matrix_case_names_are_stable_and_collision_free():
    matrix = _script_module("run_flex_vs_transformers_matrix")

    assert matrix.case_name("vllm_flex", 0.1, 15, 4) == "vllm_flex-r0p1-l15-b4"
    assert matrix.case_name("transformers_flash", 0.05, 0, 1) == (
        "transformers_flash-r0p05-l0-b1"
    )
