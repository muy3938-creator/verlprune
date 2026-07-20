import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from verl.models.vision_token_pruning.config import VisionTokenPruningConfig  # noqa: E402
from verl.models.vision_token_pruning.strategy import VisionTokenSelectionEngine  # noqa: E402
from verl.models.vision_token_pruning.transport import (  # noqa: E402
    decode_embedding_selection_metadata,
)
from verl.vllm_plugins.vision_token_pruning import (  # noqa: E402
    VerlPrunedQwen2_5VLForConditionalGeneration,
    VerlPrunedQwen3VLForConditionalGeneration,
    VerlRandomPrunedQwen2_5VLForConditionalGeneration,
    VerlRandomPrunedQwen3VLForConditionalGeneration,
    _RandomPruningMixin,
)
from verl.vllm_plugins.layerwise_flex_vision_token_pruning import (  # noqa: E402
    _LayerwiseFlexPruningMixin,
)


def test_oot_models_expose_vllm_018_constructor_and_pruning_interface():
    for model_class in (
        VerlPrunedQwen2_5VLForConditionalGeneration,
        VerlPrunedQwen3VLForConditionalGeneration,
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
        _selection_engine=VisionTokenSelectionEngine(
            VisionTokenPruningConfig(enabled=True, keep_ratio=0.5),
            seed=1234,
        ),
        _append_zero_position_axis=qwen3,
        _binary_selection_capture=qwen3,
    )
    embeddings = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    image_grid_thw = torch.tensor([[1, 4, 4]])

    (annotated,) = _RandomPruningMixin._prune_and_annotate_images(
        dummy,
        (embeddings,),
        image_grid_thw,
    )
    encoded = annotated[:, -4:-2].long() if qwen3 else annotated[:, -2:].long()
    kept_indices = encoded[:, 0] + encoded[:, 1] * 256 - 1

    assert annotated.shape[0] == 2
    assert kept_indices[-1].item() == 3
    assert torch.equal(annotated[:, :8], embeddings[kept_indices])


def test_two_stage_layerwise_helper_physically_compacts_before_flex_decode_routing():
    prefill_config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        selector="embedding_norm",
    )
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=0.5,
        selector="vision_pulse",
        selector_input="decode_query",
        prune_after_layer=3,
        prefill_keep_ratio=0.5,
    )
    dummy = SimpleNamespace(
        visual=SimpleNamespace(spatial_merge_size=2),
        _pruning_config=config,
        _prefill_selection_engine=VisionTokenSelectionEngine(prefill_config, seed=0),
        _selection_engine=VisionTokenSelectionEngine(config, seed=0),
        _append_zero_position_axis=False,
    )
    embeddings = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    (annotated,) = _LayerwiseFlexPruningMixin._annotate_image_selection(
        dummy,
        (embeddings,),
        torch.tensor([[1, 4, 4]]),
    )
    encoded_indices = decode_embedding_selection_metadata(annotated[:, -4:-2]) - 1
    encoded_counts = decode_embedding_selection_metadata(annotated[:, -2:])

    assert annotated.shape[0] == 2
    assert encoded_indices.tolist() == [2, 3]
    assert encoded_counts.tolist() == [4, 4]
    assert torch.equal(annotated[:, :8], embeddings[encoded_indices])
