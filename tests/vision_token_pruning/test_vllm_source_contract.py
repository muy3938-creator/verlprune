"""Source-level acceptance test for the pinned vLLM 0.18 plugin APIs."""

import os
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def test_vllm_018_plugin_contracts_are_present():
    root_value = os.environ.get("VLLM_SOURCE_ROOT")
    if not root_value:
        pytest.skip("set VLLM_SOURCE_ROOT to run the vLLM source contract test")
    root = Path(root_value)

    qwen25 = (root / "vllm/model_executor/models/qwen2_5_vl.py").read_text()
    qwen3 = (root / "vllm/model_executor/models/qwen3_vl.py").read_text()
    capturer = (root / "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py").read_text()
    vllm_requirements = (root / "requirements/common.txt").read_text()
    project_requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text()

    assert "SupportsMultiModalPruning" in qwen25
    assert "SupportsMultiModalPruning" in qwen3
    assert "def recompute_mrope_positions(" in qwen25
    assert "def recompute_mrope_positions(" in qwen3
    assert "def capture(self, layer_id: int, topk_ids: torch.Tensor)" in capturer
    assert "def get_instance()" in capturer
    assert "transformers >= 4.56.0, < 5" in vllm_requirements

    transformers_pin = next(
        line.split("==", maxsplit=1)[1]
        for line in project_requirements.splitlines()
        if line.startswith("transformers==")
    )
    assert Version(transformers_pin) in SpecifierSet(">=4.56.0,<5")
