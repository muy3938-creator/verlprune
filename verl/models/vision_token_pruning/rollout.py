"""Framework-neutral policy for wiring visual-token pruning into a rollout server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VisionTokenPruningConfig, coerce_vision_token_pruning_config
from .protocol import decode_rollout_selection

PRUNED_VLLM_ARCHITECTURES = {
    "qwen2_5_vl": "VerlPrunedQwen2_5VLForConditionalGeneration",
    "qwen3_vl": "VerlPrunedQwen3VLForConditionalGeneration",
}

LAYERWISE_PRUNED_VLLM_ARCHITECTURES = {
    "qwen2_5_vl": "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration",
}


@dataclass(frozen=True)
class VllmPruningLaunchOptions:
    """Additional vLLM arguments required by the out-of-tree pruning model."""

    hf_overrides: dict[str, Any]
    cli_args: dict[str, Any]


class VisionTokenPruningRollout:
    """Own rollout launch, request validation, and selection decoding rules."""

    def __init__(
        self,
        config: VisionTokenPruningConfig | dict[str, Any] | None,
        *,
        model_type: str,
        image_token_id: int | None,
    ) -> None:
        self.config = coerce_vision_token_pruning_config(config)
        self.model_type = model_type
        self.image_token_id = image_token_id

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def build_launch_options(self, *, routing_replay_enabled: bool) -> VllmPruningLaunchOptions:
        if not self.enabled:
            return VllmPruningLaunchOptions(hf_overrides={}, cli_args={})
        if routing_replay_enabled:
            raise ValueError("vision token pruning cannot share routed_experts with rollout routing replay")
        architectures = (
            LAYERWISE_PRUNED_VLLM_ARCHITECTURES
            if self.config.uses_layerwise_backend
            else PRUNED_VLLM_ARCHITECTURES
        )
        try:
            architecture = architectures[self.model_type]
        except KeyError as exc:
            implementation = "layerwise" if self.config.uses_layerwise_backend else "layer-0"
            raise ValueError(
                f"vLLM {implementation} visual-token pruning does not support model_type={self.model_type!r}"
            ) from exc

        pruning_config = {
            "keep_ratio": self.config.keep_ratio,
            "selector": self.config.selector,
        }
        if self.config.uses_layerwise_backend:
            pruning_config["prune_after_layer"] = self.config.prune_after_layer

        return VllmPruningLaunchOptions(
            hf_overrides={
                "architectures": [architecture],
                "text_config": {"num_experts_per_tok": 1},
                "vision_token_pruning": pruning_config,
            },
            cli_args={
                "video_pruning_rate": 1.0 - self.config.keep_ratio,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
                "enable_return_routed_experts": True,
                **(
                    {
                        "enable_chunked_prefill": False,
                        "enable_prefix_caching": False,
                        "enforce_eager": True,
                    }
                    if self.config.uses_layerwise_backend
                    else {}
                ),
            },
        )

    def inspect_request(
        self,
        *,
        prompt_ids: list[int],
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> int | None:
        """Validate a request and return its expanded image-token count."""

        if not self.enabled:
            return None
        if image_data is None or len(image_data) != 1 or video_data is not None:
            raise ValueError("vision token pruning requires exactly one image and no video")
        if self.image_token_id is None:
            raise ValueError("vision token pruning requires image_token_id in the model config")
        token_count = prompt_ids.count(self.image_token_id)
        if token_count == 0:
            raise ValueError("vision token pruning found no expanded image tokens in the rollout prompt")
        return token_count

    def decode_selection(self, routed_experts: Any, *, original_token_count: int | None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if original_token_count is None:
            raise RuntimeError("missing original image-token count for a pruned rollout")
        return decode_rollout_selection(
            routed_experts,
            keep_ratio=self.config.keep_ratio,
            original_visual_token_count=original_token_count,
            selector=self.config.selector,
        ).to_wire()
