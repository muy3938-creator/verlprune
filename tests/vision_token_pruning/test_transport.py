import pytest

torch = pytest.importorskip("torch")

from verl.models.vision_token_pruning.transport import (  # noqa: E402
    decode_embedding_selection_metadata,
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
