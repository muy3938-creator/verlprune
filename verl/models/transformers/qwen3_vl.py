# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

from verl.models.vision_token_pruning.embeddings import prune_visual_embedding_outputs
from verl.utils.transformers_compat import unpack_visual_output

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def qwen3_vl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen3-VL attention with an optional layerwise visual-token mask."""

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    layerwise_mask = kwargs.pop("vision_token_pruning_mask", None)
    dynamic_mask = kwargs.pop("vision_token_dynamic_attention_mask", None)
    prune_after_layer = kwargs.pop("vision_token_prune_after_layer", None)
    if layerwise_mask is not None and dynamic_mask is not None:
        raise ValueError("static and dynamic visual-token masks are mutually exclusive")
    apply_pruning = (
        layerwise_mask is not None
        and prune_after_layer is not None
        and self.layer_idx > int(prune_after_layer)
    )
    apply_dynamic = (
        dynamic_mask is not None
        and prune_after_layer is not None
        and self.layer_idx > int(prune_after_layer)
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    retained_indices = None
    if apply_pruning and attention_mask is None:
        if hidden_states.shape[0] != 1 or layerwise_mask.numel() != hidden_states.shape[1]:
            raise ValueError("packed Qwen3-VL layerwise pruning expects one flattened sequence")
        retained_indices = (
            layerwise_mask.reshape(-1)
            .to(device=query_states.device, dtype=torch.bool)
            .nonzero(as_tuple=False)
            .flatten()
        )
        query_states = query_states.index_select(2, retained_indices)
        key_states = key_states.index_select(2, retained_indices)
        value_states = value_states.index_select(2, retained_indices)
    elif apply_pruning:
        raise ValueError("padded Qwen3-VL layerwise replay is not supported; enable remove padding")

    if apply_dynamic:
        sequence_length = hidden_states.shape[1]
        if dynamic_mask.shape != (hidden_states.shape[0], sequence_length, sequence_length):
            raise ValueError("Qwen3-VL dynamic mask must have shape [batch, query, key]")
        allowed = dynamic_mask.to(device=query_states.device, dtype=torch.bool)
        allowed &= torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=query_states.device,
        ).tril()
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=allowed[:, None],
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            enable_gqa=query_states.shape[1] != key_states.shape[1],
        ).transpose(1, 2)
        attn_weights = None
    else:
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    if retained_indices is not None:
        compact_output = attn_output
        attn_output = compact_output.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1], *compact_output.shape[2:])
        )
        attn_output.index_copy_(1, retained_indices, compact_output)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to separate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1
            # (we use timestamps to encode the temporal information for videos)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    vision_token_keep_mask: Optional[torch.BoolTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = unpack_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))

        image_embeds, deepstack_image_embeds = prune_visual_embedding_outputs(
            image_embeds,
            deepstack_image_embeds,
            vision_token_keep_mask,
        )
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = unpack_visual_output(
            model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        )
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = unpack_visual_output(
            model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds = inputs_embeds + 0.0 * emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


@dataclass
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    vision_token_keep_mask: Optional[torch.BoolTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        vision_token_keep_mask,
    )  # avoid lora module having multiple keyword arguments
    kwargs.update(input_kwargs)
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def patch_qwen3_vl_moe_sparse_moe_block_forward():
    """
    Monkey patch to fix a bug in transformers 4.57.3 where Qwen3VLMoeTextSparseMoeBlock.forward
    incorrectly uses torch.zeros_like(hidden_states) instead of torch.zeros_like(router_logits)
    when creating router_weights (line 148 in modeling_qwen3_vl_moe.py).

    This is a minimal fix that only changes the problematic line while keeping the rest of the
    original implementation intact.
    """
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    except ImportError:
        # Model not available, skip patching
        return

    # Store the original forward method for reference
    original_forward = Qwen3VLMoeTextSparseMoeBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # BUG FIX: Original code incorrectly uses hidden_states here, should use router_logits
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

    # Apply the patch
    Qwen3VLMoeTextSparseMoeBlock.forward = patched_forward
    logger.info("Monkey patched Qwen3VLMoeTextSparseMoeBlock.forward to fix router_weights bug")
