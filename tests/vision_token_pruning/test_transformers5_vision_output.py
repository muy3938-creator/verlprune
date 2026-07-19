from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from verl.models.transformers.qwen2_vl import _unwrap_vision_embeddings  # noqa: E402


def test_transformers5_pooler_output_is_unwrapped():
    embeddings = torch.randn(4, 8)

    assert _unwrap_vision_embeddings(SimpleNamespace(pooler_output=embeddings)) is embeddings
    assert _unwrap_vision_embeddings(embeddings) is embeddings


def test_invalid_vision_output_fails_closed():
    with pytest.raises(TypeError, match="pooler_output"):
        _unwrap_vision_embeddings(SimpleNamespace(last_hidden_state=torch.randn(4, 8)))
