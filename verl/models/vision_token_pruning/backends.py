"""Backend capability profiles for visual-token pruning rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VisionTokenPruningConfig


# vLLM's Qwen EVS postprocessing hook is skipped when video_pruning_rate is
# exactly zero. Delayed pruning needs that hook only to attach selection
# metadata, while its processor and model preserve every visual token.
DELAYED_PREFILL_METADATA_PRUNING_RATE = 1e-9


@dataclass(frozen=True)
class VllmPruningLaunchOptions:
    hf_overrides: dict[str, Any]
    cli_args: dict[str, Any]


@dataclass(frozen=True)
class VllmPruningBackendProfile:
    name: str
    architectures: dict[str, str]
    requires_eager: bool = False
    supports_chunked_prefill: bool = True
    supports_prefix_caching: bool = False

    def architecture_for(self, model_type: str) -> str:
        try:
            return self.architectures[model_type]
        except KeyError as exc:
            raise ValueError(
                f"vLLM {self.name} visual-token pruning does not support model_type={model_type!r}"
            ) from exc


PHYSICAL_BACKEND = VllmPruningBackendProfile(
    name="physical",
    architectures={
        "qwen2_5_vl": "VerlPrunedQwen2_5VLForConditionalGeneration",
        "qwen3_vl": "VerlPrunedQwen3VLForConditionalGeneration",
    },
    # Protocol-v3 metadata is injected by a Python model-forward hook. vLLM's
    # compiled graph replays the tensor graph without re-entering that hook.
    requires_eager=True,
)

LAYERWISE_FLEX_BACKEND = VllmPruningBackendProfile(
    name="layerwise_flex",
    architectures={
        "qwen2_5_vl": "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration",
        "qwen3_vl": "VerlLayerwiseFlexPrunedQwen3VLForConditionalGeneration",
    },
    requires_eager=True,
    supports_chunked_prefill=False,
)

LAYERWISE_COMPACT_FLASH_BACKEND = VllmPruningBackendProfile(
    name="layerwise_compact_flash",
    architectures={
        "qwen2_5_vl": "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration",
    },
    requires_eager=True,
    supports_chunked_prefill=False,
)


def resolve_vllm_backend(config: VisionTokenPruningConfig) -> VllmPruningBackendProfile:
    if not config.uses_layerwise_backend:
        return PHYSICAL_BACKEND
    if config.layerwise_backend == "flex":
        return LAYERWISE_FLEX_BACKEND
    return LAYERWISE_COMPACT_FLASH_BACKEND


def build_vllm_pruning_launch_options(
    config: VisionTokenPruningConfig,
    *,
    model_type: str,
    routing_replay_enabled: bool,
) -> VllmPruningLaunchOptions:
    if not config.enabled:
        return VllmPruningLaunchOptions(hf_overrides={}, cli_args={})
    if routing_replay_enabled:
        raise ValueError("vision token pruning cannot share routed_experts with rollout routing replay")

    backend = resolve_vllm_backend(config)
    physical_keep_ratio = (
        config.prefill_keep_ratio
        if config.uses_physical_prefill_pruning
        else (
            1.0 - DELAYED_PREFILL_METADATA_PRUNING_RATE
            if config.uses_delayed_prefill_pruning
            else config.keep_ratio
        )
    )
    assert physical_keep_ratio is not None
    cli_args: dict[str, Any] = {
        "video_pruning_rate": 1.0 - physical_keep_ratio,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "enable_return_routed_experts": True,
    }
    if backend.requires_eager:
        cli_args["enforce_eager"] = True
    if not backend.supports_chunked_prefill:
        cli_args["enable_chunked_prefill"] = False
    if not backend.supports_prefix_caching:
        cli_args["enable_prefix_caching"] = False
    if backend is LAYERWISE_FLEX_BACKEND:
        cli_args["attention_config"] = {"backend": "FLEX_ATTENTION"}
    capture_capacity = 1
    if config.uses_dynamic_decode_selection:
        capture_capacity = int(config.selector_kwargs.get("capture_capacity", 64))
        if capture_capacity <= 0:
            raise ValueError("dynamic visual selection capture_capacity must be positive")
        if config.uses_two_stage_pruning and capture_capacity < 3:
            raise ValueError("two-stage selection capture_capacity must be at least 3")
    return VllmPruningLaunchOptions(
        hf_overrides={
            "architectures": [backend.architecture_for(model_type)],
            # Dense Qwen does not consume this MoE field. vLLM's temporary
            # capture transport uses it as the per-query metadata width.
            "text_config": {"num_experts_per_tok": capture_capacity},
            "vision_token_pruning": config.to_backend_payload(),
        },
        cli_args=cli_args,
    )
