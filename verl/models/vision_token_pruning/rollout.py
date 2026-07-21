"""Framework-neutral policy for wiring visual-token pruning into a rollout server."""

from __future__ import annotations

from typing import Any

from .backends import VllmPruningLaunchOptions, build_vllm_pruning_launch_options
from .config import VisionTokenPruningConfig, coerce_vision_token_pruning_config
from .curriculum import resolve_keep_ratio
from .transport import (
    decode_vllm_dynamic_selection_capture,
    decode_vllm_selection_capture,
    decode_vllm_two_stage_selection_capture,
)


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
        return build_vllm_pruning_launch_options(
            self.config,
            model_type=self.model_type,
            routing_replay_enabled=routing_replay_enabled,
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

    def keep_ratio_for_step(self, global_step: int | None) -> float:
        return resolve_keep_ratio(
            self.config.keep_ratio_schedule,
            global_step,
            fallback=self.config.keep_ratio,
        )

    def decode_selection(
        self,
        routed_experts: Any,
        *,
        original_token_count: int | None,
        keep_ratio: float | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if original_token_count is None:
            raise RuntimeError("missing original image-token count for a pruned rollout")
        effective_keep_ratio = self.config.keep_ratio if keep_ratio is None else float(keep_ratio)
        if self.config.uses_two_stage_pruning:
            assert self.config.prefill_keep_ratio is not None
            return decode_vllm_two_stage_selection_capture(
                routed_experts,
                prefill_keep_ratio=self.config.prefill_keep_ratio,
                prefill_selector=self.config.prefill_selector,
                prefill_selector_kwargs=self.config.prefill_selector_kwargs,
                decode_keep_ratio=effective_keep_ratio,
                decode_selector=self.config.selector,
                decode_selector_kwargs=self.config.selector_kwargs,
            ).to_wire()
        decoder = (
            decode_vllm_dynamic_selection_capture
            if self.config.uses_dynamic_decode_selection
            else decode_vllm_selection_capture
        )
        ratio_name = (
            "nominal_keep_ratio"
            if self.config.uses_dynamic_decode_selection
            else "keep_ratio"
        )
        return decoder(
            routed_experts,
            **{ratio_name: effective_keep_ratio},
            original_visual_token_count=original_token_count,
            selector=self.config.selector,
            selector_kwargs=self.config.selector_kwargs,
        ).to_wire()
