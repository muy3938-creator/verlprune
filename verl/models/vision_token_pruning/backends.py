"""Backend capability profiles for visual-token pruning rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VisionTokenPruningConfig


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
    cli_args: dict[str, Any] = {
        "video_pruning_rate": 1.0 - config.keep_ratio,
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
    return VllmPruningLaunchOptions(
        hf_overrides={
            "architectures": [backend.architecture_for(model_type)],
            "text_config": {"num_experts_per_tok": 1},
            "vision_token_pruning": config.to_backend_payload(),
        },
        cli_args=cli_args,
    )
