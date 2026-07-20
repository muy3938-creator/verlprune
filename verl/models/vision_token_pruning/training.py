"""Rollout-to-training replay and teacher isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import VisionTokenPruningConfig, compute_selector_fingerprint
from .embeddings import KEEP_MASK_KEY
from .protocol import (
    DynamicVisionTokenSelection,
    TwoStageVisionTokenSelection,
    VisionTokenSelection,
    selection_from_wire,
)

SELECTION_WIRE_KEY = "vision_token_selection"
_USE_PREPARED_LAYERWISE_MASK = object()


def selection_to_keep_mask(selection: VisionTokenSelection, *, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros(selection.original_visual_token_count, dtype=torch.bool, device=device)
    mask[list(selection.kept_visual_indices)] = True
    return mask


def attach_selection_to_multi_modal_inputs(
    multi_modal_inputs: dict[str, Any],
    selection_wire: dict[str, Any],
) -> dict[str, Any]:
    if KEEP_MASK_KEY in multi_modal_inputs or SELECTION_WIRE_KEY in multi_modal_inputs:
        raise ValueError("visual-token pruning metadata is already attached")
    selection = selection_from_wire(selection_wire)
    attached = {**multi_modal_inputs, SELECTION_WIRE_KEY: selection.to_wire()}
    if isinstance(selection, VisionTokenSelection):
        attached[KEEP_MASK_KEY] = selection_to_keep_mask(selection)
    elif isinstance(selection, TwoStageVisionTokenSelection):
        attached[KEEP_MASK_KEY] = selection_to_keep_mask(selection.prefill)
    return attached


def strip_pruning_metadata(multi_modal_inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove all pruning fields before an unpruned teacher forward."""

    return {
        key: value
        for key, value in multi_modal_inputs.items()
        if key not in {KEEP_MASK_KEY, SELECTION_WIRE_KEY}
    }


def strip_selection_metadata(multi_modal_inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the model-facing mask and remove its serialized rollout record."""

    return {key: value for key, value in multi_modal_inputs.items() if key != SELECTION_WIRE_KEY}


@dataclass(frozen=True)
class PreparedActorPruningInputs:
    """One fully validated actor/teacher forward plan."""

    attention_mask: torch.Tensor
    per_sample_multi_modal_inputs: list[dict[str, Any] | None]
    layerwise_attention_mask: torch.Tensor | None = None
    dynamic_layerwise_attention_mask: torch.Tensor | None = None

    def layerwise_forward_kwargs(
        self,
        config: VisionTokenPruningConfig,
        *,
        attention_mask: torch.Tensor | None | object = _USE_PREPARED_LAYERWISE_MASK,
        dynamic_attention_mask: torch.Tensor | None | object = _USE_PREPARED_LAYERWISE_MASK,
    ) -> dict[str, Any]:
        mask = (
            self.layerwise_attention_mask
            if attention_mask is _USE_PREPARED_LAYERWISE_MASK
            else attention_mask
        )
        dynamic_mask = (
            self.dynamic_layerwise_attention_mask
            if dynamic_attention_mask is _USE_PREPARED_LAYERWISE_MASK
            else dynamic_attention_mask
        )
        if mask is None and dynamic_mask is None:
            return {}
        if not config.uses_layerwise_backend:
            raise ValueError("layerwise forward metadata requires the layerwise backend")
        if dynamic_mask is not None:
            if not isinstance(dynamic_mask, torch.Tensor):
                raise TypeError("dynamic layerwise attention mask must be a tensor or None")
            output = {
                "vision_token_dynamic_attention_mask": dynamic_mask,
                "vision_token_prune_after_layer": config.prune_after_layer,
            }
            if mask is not None:
                if not config.uses_delayed_prefill_pruning:
                    raise ValueError("combined static/dynamic masks require delayed two-stage pruning")
                if not isinstance(mask, torch.Tensor):
                    raise TypeError("layerwise attention mask must be a tensor or None")
                output["vision_token_pruning_mask"] = mask
                output["vision_token_prefill_prune_after_layer"] = config.prefill_prune_after_layer
                output["vision_token_decode_prune_after_layer"] = config.prune_after_layer
            return output
        if not isinstance(mask, torch.Tensor):
            raise TypeError("layerwise attention mask must be a tensor or None")
        return {
            "vision_token_pruning_mask": mask,
            "vision_token_prune_after_layer": config.prune_after_layer,
        }


def prepare_actor_pruning_inputs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    per_sample_multi_modal_inputs: list[dict[str, Any] | None],
    image_token_id: int | None,
    config: VisionTokenPruningConfig,
    apply_pruning: bool,
) -> PreparedActorPruningInputs:
    """Build actor inputs or strip all metadata for the unpruned teacher."""

    if not apply_pruning:
        return PreparedActorPruningInputs(
            attention_mask=attention_mask,
            per_sample_multi_modal_inputs=[
                strip_pruning_metadata(inputs) if inputs is not None else None
                for inputs in per_sample_multi_modal_inputs
            ],
        )
    if not config.enabled:
        raise ValueError("cannot apply visual-token pruning when it is disabled in the actor config")
    if image_token_id is None:
        raise ValueError("vision token pruning requires model.config.image_token_id")

    if config.uses_two_stage_pruning:
        prefill_attention_mask, layerwise_attention_mask, dynamic_attention_mask = replay_two_stage_rollout_selection(
            input_ids,
            attention_mask,
            per_sample_multi_modal_inputs,
            image_token_id=image_token_id,
            config=config,
        )
        return PreparedActorPruningInputs(
            attention_mask=prefill_attention_mask,
            per_sample_multi_modal_inputs=[
                (
                    strip_pruning_metadata(inputs)
                    if config.uses_delayed_prefill_pruning
                    else strip_selection_metadata(inputs)
                )
                if inputs is not None
                else None
                for inputs in per_sample_multi_modal_inputs
            ],
            layerwise_attention_mask=layerwise_attention_mask,
            dynamic_layerwise_attention_mask=dynamic_attention_mask,
        )

    if config.uses_dynamic_decode_selection:
        dynamic_attention_mask = replay_dynamic_rollout_selection(
            input_ids,
            attention_mask,
            per_sample_multi_modal_inputs,
            image_token_id=image_token_id,
            expected_keep_ratio=config.keep_ratio,
            expected_selector=config.selector,
            expected_selector_kwargs=config.selector_kwargs,
        )
        return PreparedActorPruningInputs(
            attention_mask=attention_mask,
            per_sample_multi_modal_inputs=[
                strip_pruning_metadata(inputs) if inputs is not None else None
                for inputs in per_sample_multi_modal_inputs
            ],
            dynamic_layerwise_attention_mask=dynamic_attention_mask,
        )

    replayed_attention_mask = replay_rollout_selection_on_attention_mask(
        input_ids,
        attention_mask,
        per_sample_multi_modal_inputs,
        image_token_id=image_token_id,
        expected_keep_ratio=config.keep_ratio,
        expected_selector=config.selector,
        expected_selector_kwargs=config.selector_kwargs,
    )
    if config.uses_layerwise_backend:
        return PreparedActorPruningInputs(
            attention_mask=attention_mask,
            per_sample_multi_modal_inputs=[
                strip_pruning_metadata(inputs) if inputs is not None else None
                for inputs in per_sample_multi_modal_inputs
            ],
            layerwise_attention_mask=replayed_attention_mask,
        )
    return PreparedActorPruningInputs(
        attention_mask=replayed_attention_mask,
        per_sample_multi_modal_inputs=[
            strip_selection_metadata(inputs) if inputs is not None else None
            for inputs in per_sample_multi_modal_inputs
        ],
    )


def replay_rollout_selection_on_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    per_sample_multi_modal_inputs: list[dict[str, Any] | None],
    *,
    image_token_id: int,
    expected_keep_ratio: float,
    expected_selector: str = "random",
    expected_selector_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Replay the exact rollout-selected image tokens on the actor sequence."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have identical rank-2 shapes")
    if len(per_sample_multi_modal_inputs) != input_ids.shape[0]:
        raise ValueError("multi_modal_inputs length must match batch size")

    expected_fingerprint = compute_selector_fingerprint(expected_selector, expected_selector_kwargs)
    output = attention_mask.clone()
    for sample_index, multi_modal_inputs in enumerate(per_sample_multi_modal_inputs):
        if multi_modal_inputs is None:
            raise ValueError(f"sample {sample_index} is missing multimodal inputs")
        if KEEP_MASK_KEY not in multi_modal_inputs or SELECTION_WIRE_KEY not in multi_modal_inputs:
            raise ValueError(f"sample {sample_index} is missing rollout visual-token selection")

        keep_mask = multi_modal_inputs[KEEP_MASK_KEY]
        selection = VisionTokenSelection.from_wire(multi_modal_inputs[SELECTION_WIRE_KEY])
        if selection.keep_ratio != expected_keep_ratio:
            raise ValueError(
                f"sample {sample_index} rollout keep_ratio {selection.keep_ratio} "
                f"does not match actor keep_ratio {expected_keep_ratio}"
            )
        if selection.selector != expected_selector:
            raise ValueError(
                f"sample {sample_index} rollout selector {selection.selector!r} "
                f"does not match actor selector {expected_selector!r}"
            )
        if selection.selector_fingerprint != expected_fingerprint:
            raise ValueError(
                f"sample {sample_index} rollout strategy configuration does not match the actor config"
            )
        if not torch.equal(keep_mask.cpu(), selection_to_keep_mask(selection)):
            raise ValueError(f"sample {sample_index} keep mask does not match rollout selection")

        image_positions = (
            (input_ids[sample_index] == image_token_id) & output[sample_index].bool()
        ).nonzero(as_tuple=False).flatten()
        if image_positions.numel() != selection.original_visual_token_count:
            raise ValueError(
                f"sample {sample_index} selection covers {selection.original_visual_token_count} visual tokens, "
                f"but input contains {image_positions.numel()} image tokens"
            )
        output[sample_index, image_positions[~keep_mask.to(image_positions.device)]] = 0
    return output


def replay_two_stage_rollout_selection(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    per_sample_multi_modal_inputs: list[dict[str, Any] | None],
    *,
    image_token_id: int,
    config: VisionTokenPruningConfig,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Replay first-stage pruning and decode routing within that subset."""

    if not config.uses_two_stage_pruning or config.prefill_keep_ratio is None:
        raise ValueError("two-stage replay requires a two-stage pruning config")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have identical rank-2 shapes")
    if len(per_sample_multi_modal_inputs) != input_ids.shape[0]:
        raise ValueError("multi_modal_inputs length must match batch size")

    prefill_fingerprint = compute_selector_fingerprint(
        config.prefill_selector,
        config.prefill_selector_kwargs,
    )
    decode_fingerprint = compute_selector_fingerprint(config.selector, config.selector_kwargs)
    stage_one_attention = attention_mask.clone()
    batch_size, sequence_length = input_ids.shape
    dynamic_mask = torch.ones(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=input_ids.device,
    )

    for sample_index, multi_modal_inputs in enumerate(per_sample_multi_modal_inputs):
        if multi_modal_inputs is None or SELECTION_WIRE_KEY not in multi_modal_inputs:
            raise ValueError(f"sample {sample_index} is missing two-stage rollout selection")
        selection = selection_from_wire(multi_modal_inputs[SELECTION_WIRE_KEY])
        if not isinstance(selection, TwoStageVisionTokenSelection):
            raise ValueError(f"sample {sample_index} does not contain a two-stage selection")
        prefill = selection.prefill
        decode = selection.decode
        if prefill.keep_ratio != config.prefill_keep_ratio:
            raise ValueError(f"sample {sample_index} prefill keep ratio does not match actor config")
        if prefill.selector != config.prefill_selector or prefill.selector_fingerprint != prefill_fingerprint:
            raise ValueError(f"sample {sample_index} prefill selector does not match actor config")
        if decode.nominal_keep_ratio != config.keep_ratio:
            raise ValueError(f"sample {sample_index} decode keep ratio does not match actor config")
        if decode.selector != config.selector or decode.selector_fingerprint != decode_fingerprint:
            raise ValueError(f"sample {sample_index} decode selector does not match actor config")
        keep_mask = multi_modal_inputs.get(KEEP_MASK_KEY)
        if keep_mask is None or not torch.equal(keep_mask.cpu(), selection_to_keep_mask(prefill)):
            raise ValueError(f"sample {sample_index} prefill keep mask does not match rollout selection")

        image_positions = (
            (input_ids[sample_index] == image_token_id) & attention_mask[sample_index].bool()
        ).nonzero(as_tuple=False).flatten()
        if len(image_positions) != prefill.original_visual_token_count:
            raise ValueError(
                f"sample {sample_index} prefill selection covers {prefill.original_visual_token_count} "
                f"visual tokens, but input contains {len(image_positions)}"
            )
        keep_on_device = keep_mask.to(image_positions.device)
        stage_one_attention[sample_index, image_positions[~keep_on_device]] = 0
        retained_image_positions = image_positions[keep_on_device]

        # The dynamic stage is relative to the first-stage subset. In delayed
        # mode the dropped entries still physically exist, so every query must
        # continue masking them after the decode boundary as well.
        dynamic_mask[sample_index, :, image_positions[~keep_on_device]] = False

        routing_attention = (
            stage_one_attention[sample_index]
            if config.uses_physical_prefill_pruning
            else attention_mask[sample_index]
        )
        valid_positions = routing_attention.bool().nonzero(as_tuple=False).flatten()
        if len(decode.query_kept_visual_indices) > len(valid_positions):
            raise ValueError(f"sample {sample_index} has more decode routes than compact sequence tokens")
        for relative_query, selected_indices in enumerate(decode.query_kept_visual_indices):
            if not selected_indices:
                continue
            query_position = valid_positions[relative_query]
            dynamic_mask[sample_index, query_position, retained_image_positions] = False
            selected = torch.tensor(
                selected_indices,
                dtype=torch.long,
                device=retained_image_positions.device,
            )
            dynamic_mask[
                sample_index,
                query_position,
                retained_image_positions.index_select(0, selected),
            ] = True
    if config.uses_physical_prefill_pruning:
        return stage_one_attention, None, dynamic_mask
    return attention_mask, stage_one_attention, dynamic_mask


def replay_dynamic_rollout_selection(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    per_sample_multi_modal_inputs: list[dict[str, Any] | None],
    *,
    image_token_id: int,
    expected_keep_ratio: float,
    expected_selector: str,
    expected_selector_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Build a query-by-key mask that exactly replays decode-time routing."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have identical rank-2 shapes")
    if len(per_sample_multi_modal_inputs) != input_ids.shape[0]:
        raise ValueError("multi_modal_inputs length must match batch size")

    expected_fingerprint = compute_selector_fingerprint(expected_selector, expected_selector_kwargs)
    batch_size, sequence_length = input_ids.shape
    output = torch.ones(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=input_ids.device,
    )
    for sample_index, multi_modal_inputs in enumerate(per_sample_multi_modal_inputs):
        if multi_modal_inputs is None or SELECTION_WIRE_KEY not in multi_modal_inputs:
            raise ValueError(f"sample {sample_index} is missing rollout visual-token selection")
        selection = selection_from_wire(multi_modal_inputs[SELECTION_WIRE_KEY])
        if not isinstance(selection, DynamicVisionTokenSelection):
            raise ValueError(f"sample {sample_index} does not contain dynamic decode selection")
        if selection.nominal_keep_ratio != expected_keep_ratio:
            raise ValueError(
                f"sample {sample_index} rollout keep_ratio {selection.nominal_keep_ratio} "
                f"does not match actor keep_ratio {expected_keep_ratio}"
            )
        if selection.selector != expected_selector:
            raise ValueError(
                f"sample {sample_index} rollout selector {selection.selector!r} "
                f"does not match actor selector {expected_selector!r}"
            )
        if selection.selector_fingerprint != expected_fingerprint:
            raise ValueError(
                f"sample {sample_index} rollout strategy configuration does not match the actor config"
            )

        valid_positions = attention_mask[sample_index].bool().nonzero(as_tuple=False).flatten()
        query_rows = selection.query_kept_visual_indices
        if len(query_rows) > len(valid_positions):
            raise ValueError(
                f"sample {sample_index} captured {len(query_rows)} query routes for "
                f"{len(valid_positions)} valid sequence tokens"
            )
        image_positions = (
            (input_ids[sample_index] == image_token_id) & attention_mask[sample_index].bool()
        ).nonzero(as_tuple=False).flatten()
        if len(image_positions) != selection.original_visual_token_count:
            raise ValueError(
                f"sample {sample_index} selection covers {selection.original_visual_token_count} "
                f"visual tokens, but input contains {len(image_positions)} image tokens"
            )

        for relative_query, selected_indices in enumerate(query_rows):
            if not selected_indices:
                continue
            query_position = valid_positions[relative_query]
            output[sample_index, query_position, image_positions] = False
            selected = torch.tensor(
                selected_indices,
                dtype=torch.long,
                device=image_positions.device,
            )
            output[sample_index, query_position, image_positions.index_select(0, selected)] = True
    return output


def pack_dynamic_attention_mask(
    dynamic_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pack padded per-sample query/key masks into one block-diagonal sequence."""

    if dynamic_mask.ndim != 3:
        raise ValueError("dynamic attention mask must have shape [batch, query, key]")
    if dynamic_mask.shape[:2] != attention_mask.shape or dynamic_mask.shape[2] != attention_mask.shape[1]:
        raise ValueError("dynamic attention mask and padding mask shapes are inconsistent")
    parts = []
    for sample_index in range(attention_mask.shape[0]):
        valid = attention_mask[sample_index].bool().nonzero(as_tuple=False).flatten()
        sample = dynamic_mask[sample_index].index_select(0, valid).index_select(1, valid)
        parts.append(sample)
    packed = torch.block_diag(*parts) if parts else dynamic_mask.new_empty((0, 0))
    return packed.unsqueeze(0)
