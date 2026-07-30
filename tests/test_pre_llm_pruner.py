"""Unit tests for the minimal Pre-LLM vision token pruning modules."""

import pytest
import torch

from verl.models.vision_token_pruning.pre_llm_pruner import (
    select_vision_tokens,
    _resolve_keep_count,
)
from verl.models.vision_token_pruning.sequence_compressor import (
    CompressedSequence,
    compress_sequence,
    indices_to_keep_mask,
    prune_visual_embeddings,
)


# ==========================================================================
# select_vision_tokens
# ==========================================================================

class TestSelectVisionTokens:
    """Tests for the main selection API."""

    def test_random_basic(self):
        embeds = torch.randn(100, 64)
        indices = select_vision_tokens(embeds, keep_ratio=0.5, method="random")
        assert indices.shape == (50,)
        assert torch.equal(indices, indices.sort().values)
        assert indices.unique().shape == indices.shape  # no duplicates

    def test_uniform_basic(self):
        embeds = torch.randn(100, 64)
        indices = select_vision_tokens(embeds, keep_count=25, method="uniform")
        assert indices.shape == (25,)
        assert torch.equal(indices, indices.sort().values)

    def test_text_saliency(self):
        embeds = torch.randn(100, 64)
        text = torch.randn(10, 64)
        indices = select_vision_tokens(
            embeds, keep_ratio=0.3, method="text_saliency", text_embeds=text
        )
        assert indices.shape == (30,)

    def test_greedy_prune(self):
        embeds = torch.randn(100, 64)
        text = torch.randn(10, 64)
        indices = select_vision_tokens(
            embeds,
            keep_ratio=0.4,
            method="greedy_prune",
            text_embeds=text,
            similarity_threshold=0.8,
        )
        assert indices.shape == (40,)
        assert torch.equal(indices, indices.sort().values)

    def test_keep_all(self):
        embeds = torch.randn(10, 64)
        indices = select_vision_tokens(embeds, keep_ratio=1.0, method="random")
        assert indices.shape == (10,)

    def test_keep_count_exceeds_total(self):
        embeds = torch.randn(5, 64)
        indices = select_vision_tokens(embeds, keep_count=100, method="random")
        assert indices.shape == (5,)

    def test_invalid_method_raises(self):
        embeds = torch.randn(10, 64)
        with pytest.raises(ValueError, match="Unknown selection method"):
            select_vision_tokens(embeds, keep_ratio=0.5, method="nonexistent")

    def test_both_ratio_and_count_raises(self):
        embeds = torch.randn(10, 64)
        with pytest.raises(ValueError, match="Specify either"):
            select_vision_tokens(embeds, keep_ratio=0.5, keep_count=5)

    def test_neither_ratio_nor_count_raises(self):
        embeds = torch.randn(10, 64)
        with pytest.raises(ValueError, match="Must specify either"):
            select_vision_tokens(embeds, method="random")

    def test_wrong_ndim_raises(self):
        embeds = torch.randn(10, 4, 64)  # rank-3
        with pytest.raises(ValueError, match="rank-2"):
            select_vision_tokens(embeds, keep_ratio=0.5)

    def test_reproducibility_with_generator(self):
        embeds = torch.randn(100, 64)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        idx1 = select_vision_tokens(embeds, keep_ratio=0.3, method="random", generator=g1)
        idx2 = select_vision_tokens(embeds, keep_ratio=0.3, method="random", generator=g2)
        assert torch.equal(idx1, idx2)


# ==========================================================================
# _resolve_keep_count
# ==========================================================================

class TestResolveKeepCount:
    def test_ratio(self):
        assert _resolve_keep_count(100, keep_ratio=0.5, keep_count=None) == 50

    def test_count(self):
        assert _resolve_keep_count(100, keep_ratio=None, keep_count=30) == 30

    def test_ratio_clamps_to_1(self):
        assert _resolve_keep_count(3, keep_ratio=0.1, keep_count=None) == 1

    def test_count_clamps_to_total(self):
        assert _resolve_keep_count(5, keep_ratio=None, keep_count=999) == 5


# ==========================================================================
# prune_visual_embeddings
# ==========================================================================

class TestPruneVisualEmbeddings:
    def test_none_mask(self):
        e = torch.randn(10, 64)
        assert prune_visual_embeddings(e, None) is e

    def test_basic(self):
        e = torch.randn(10, 64)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[0, 3, 7]] = True
        out = prune_visual_embeddings(e, mask)
        assert out.shape == (3, 64)
        assert torch.equal(out, e[[0, 3, 7]])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            prune_visual_embeddings(torch.randn(5, 64), torch.ones(3, dtype=torch.bool))


# ==========================================================================
# compress_sequence
# ==========================================================================

class TestCompressSequence:
    @staticmethod
    def _make_sample(n_text: int, n_image: int, hidden_dim: int, image_token_id: int):
        text_ids = torch.arange(1, n_text + 1)  # non-image tokens
        image_ids = torch.full((n_image,), image_token_id)
        input_ids = torch.cat([text_ids, image_ids]).unsqueeze(0)  # (1, seq)
        embeds = torch.randn(1, n_text + n_image, hidden_dim)
        attn_mask = torch.ones(1, n_text + n_image, dtype=torch.long)
        pos_ids = torch.arange(n_text + n_image).unsqueeze(0).expand(3, 1, -1)  # (3, 1, seq)
        return input_ids, embeds, attn_mask, pos_ids

    def test_basic_compression(self):
        IMAGE_ID = 151655
        ids, embeds, mask, pos = self._make_sample(5, 10, 64, IMAGE_ID)
        keep = torch.zeros(10, dtype=torch.bool)
        keep[:5] = True  # keep first 5 image tokens

        result = compress_sequence(
            input_ids=ids,
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=pos,
            image_token_id=IMAGE_ID,
            keep_mask=keep,
        )
        assert isinstance(result, CompressedSequence)
        # 5 text + 5 kept image = 10
        assert result.input_ids.shape == (1, 10)
        assert result.inputs_embeds.shape == (1, 10, 64)
        assert result.attention_mask.shape == (1, 10)
        assert result.n_dropped == 5

    def test_no_pruning(self):
        IMAGE_ID = 151655
        ids, embeds, mask, pos = self._make_sample(3, 4, 32, IMAGE_ID)
        keep = torch.ones(4, dtype=torch.bool)
        result = compress_sequence(
            input_ids=ids,
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=pos,
            image_token_id=IMAGE_ID,
            keep_mask=keep,
        )
        assert result.n_dropped == 0
        assert result.input_ids.shape == ids.shape

    def test_wrong_mask_length_raises(self):
        IMAGE_ID = 151655
        ids, embeds, mask, pos = self._make_sample(3, 4, 32, IMAGE_ID)
        with pytest.raises(ValueError, match="does not match"):
            compress_sequence(
                input_ids=ids,
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=pos,
                image_token_id=IMAGE_ID,
                keep_mask=torch.ones(99, dtype=torch.bool),
            )


# ==========================================================================
# training_replay
# ==========================================================================

from verl.models.vision_token_pruning.training_replay import (
    apply_pre_llm_pruning_to_attention_mask,
    attach_keep_indices_to_sample,
    strip_pruning_from_sample,
    validate_replay_alignment,
)


class TestTrainingReplay:
    def test_attach_and_strip(self):
        multi_modal_inputs = {"pixel_values": torch.randn(10, 64)}
        keep_indices = torch.tensor([0, 2, 4])
        mm_out, selection_rec = attach_keep_indices_to_sample(
            multi_modal_inputs, keep_indices, original_visual_token_count=5, keep_ratio=0.6, method="random"
        )
        assert "vision_token_keep_mask" in mm_out
        assert mm_out["vision_token_keep_mask"].shape == (5,)
        assert mm_out["vision_token_keep_mask"].sum() == 3
        assert selection_rec["original_count"] == 5
        assert selection_rec["kept_count"] == 3

        stripped = strip_pruning_from_sample(mm_out)
        assert "vision_token_keep_mask" not in stripped
        assert "pixel_values" in stripped

    def test_validate_replay_alignment(self):
        IMAGE_ID = 151655
        input_ids = torch.tensor([[100, IMAGE_ID, IMAGE_ID, IMAGE_ID, 200]])
        keep_indices = torch.tensor([0, 2])
        mm_inputs, selection_rec = attach_keep_indices_to_sample(
            {}, keep_indices, original_visual_token_count=3
        )

        # Should pass without error
        validate_replay_alignment(input_ids, mm_inputs, IMAGE_ID, sample_index=0, selection_record=selection_rec)

        # Mismatch count should raise ValueError
        wrong_input_ids = torch.tensor([[100, IMAGE_ID, 200]])
        with pytest.raises(ValueError, match="does not match"):
            validate_replay_alignment(wrong_input_ids, mm_inputs, IMAGE_ID, sample_index=0, selection_record=selection_rec)

    def test_apply_pre_llm_pruning_to_attention_mask(self):
        IMAGE_ID = 151655
        input_ids = torch.tensor([[100, IMAGE_ID, IMAGE_ID, IMAGE_ID, 200]])
        attention_mask = torch.ones_like(input_ids)
        keep_indices = torch.tensor([0, 2])  # drop index 1
        mm_inputs, _ = attach_keep_indices_to_sample(
            {}, keep_indices, original_visual_token_count=3
        )

        new_mask = apply_pre_llm_pruning_to_attention_mask(
            input_ids, attention_mask, [mm_inputs], IMAGE_ID
        )
        assert new_mask[0, 0] == 1  # text
        assert new_mask[0, 1] == 1  # kept image token 0
        assert new_mask[0, 2] == 0  # dropped image token 1
        assert new_mask[0, 3] == 1  # kept image token 2
        assert new_mask[0, 4] == 1  # text


# ==========================================================================
# vllm_transport (Method A Side-Channel Registry)
# ==========================================================================

from verl.models.vision_token_pruning.vllm_transport import (
    clear_request_selection_registry,
    decode_vllm_selection_capture,
    pop_request_selection,
    register_request_selection,
)


class TestSideChannelTransport:
    def test_side_channel_registry_flow(self):
        clear_request_selection_registry()

        req_id = "request-uuid-12345"
        keep_indices = torch.tensor([0, 3, 7, 12])
        register_request_selection(req_id, keep_indices)

        # Pop from registry
        retrieved = pop_request_selection(req_id)
        assert retrieved is not None
        assert torch.equal(retrieved, keep_indices)

        # Second pop should return None (already consumed)
        assert pop_request_selection(req_id) is None

    def test_decode_vllm_selection_capture_with_request_id(self):
        clear_request_selection_registry()

        req_id = "request-uuid-99999"
        keep_indices = torch.tensor([1, 4, 9])
        register_request_selection(req_id, keep_indices)

        # decode via request_id string
        retrieved = decode_vllm_selection_capture(req_id)
        assert retrieved is not None
        assert torch.equal(retrieved, keep_indices)



