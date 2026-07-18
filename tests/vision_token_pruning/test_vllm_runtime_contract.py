import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from verl.vllm_plugins.vision_token_pruning import (  # noqa: E402
    VerlRandomPrunedQwen2_5VLForConditionalGeneration,
    VerlRandomPrunedQwen3VLForConditionalGeneration,
    _RandomPruningMixin,
)


def test_oot_models_expose_vllm_018_constructor_and_pruning_interface():
    for model_class in (
        VerlRandomPrunedQwen2_5VLForConditionalGeneration,
        VerlRandomPrunedQwen3VLForConditionalGeneration,
    ):
        parameters = inspect.signature(model_class.__init__).parameters
        assert "vllm_config" in parameters
        assert "prefix" in parameters
        assert model_class.supports_multimodal_pruning is True


@pytest.mark.parametrize("qwen3", [False, True])
def test_real_vllm_mrope_helper_physically_gathers_selected_embeddings(qwen3):
    dummy = SimpleNamespace(
        visual=SimpleNamespace(spatial_merge_size=2),
        _keep_ratio=0.5,
        _selection_seed=1234,
        _selection_counter=0,
        _append_zero_position_axis=qwen3,
    )
    embeddings = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    image_grid_thw = torch.tensor([[1, 4, 4]])

    (annotated,) = _RandomPruningMixin._prune_and_annotate_images(
        dummy,
        (embeddings,),
        image_grid_thw,
    )
    encoded = annotated[:, -2:].long()
    kept_indices = encoded[:, 0] + encoded[:, 1] * 256 - 1

    assert annotated.shape[0] == 2
    assert kept_indices[-1].item() == 3
    assert torch.equal(annotated[:, :8], embeddings[kept_indices])
