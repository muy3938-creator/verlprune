import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.transport import (  # noqa: E402
    decode_embedding_selection_metadata,
    decode_vllm_dynamic_selection_capture,
    decode_vllm_selection_capture,
    encode_embedding_selection_metadata,
)


def test_embedding_metadata_codec_round_trip_across_radix_boundary():
    indices = torch.tensor([0, 1, 255, 256, 1023], dtype=torch.long)
    metadata = encode_embedding_selection_metadata(indices, dtype=torch.float32)
    annotated = torch.cat([torch.zeros(len(indices), 3), metadata], dim=1)

    assert decode_embedding_selection_metadata(annotated).tolist() == [1, 2, 256, 257, 1024]


def test_embedding_metadata_codec_rejects_unrepresentable_index():
    with pytest.raises(ValueError, match="fewer than 65536"):
        encode_embedding_selection_metadata(torch.tensor([65535]), dtype=torch.float32)


def test_dynamic_capture_preserves_per_query_rows_and_variable_budgets():
    selection = decode_vllm_dynamic_selection_capture(
        [
            [[0, 0, 0]],
            [[1, 4, 0]],
            [[2, 3, 4]],
        ],
        nominal_keep_ratio=0.05,
        original_visual_token_count=4,
        selector="vision_pulse",
    )

    assert selection.query_kept_visual_indices == ((), (0, 3), (1, 2, 3))


def test_static_binary_capture_uses_only_zero_one_values_and_round_trips_indices():
    def binary_row(index: int | None, original_count: int = 4) -> list[list[int]]:
        row = [[0] for _ in range(36)]
        if index is not None:
            encoded = index + 1
            for bit in range(16):
                row[bit][0] = (encoded >> bit) & 1
                row[16 + bit][0] = (original_count >> bit) & 1
            row[-1][0] = 1
        return row

    selection = decode_vllm_selection_capture(
        [
            binary_row(0),
            binary_row(None),
            binary_row(2),
        ],
        keep_ratio=0.5,
        # The binary Qwen3 payload is authoritative; this deliberately models
        # a processor-side estimate that differs from vLLM's image expansion.
        original_visual_token_count=3,
        selector="uniform",
    )

    assert selection.original_visual_token_count == 4
    assert selection.kept_visual_indices == (0, 2)
