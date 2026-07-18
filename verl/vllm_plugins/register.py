def register_vision_token_pruning_models() -> None:
    """Register layer-0 random-pruning Qwen-VL models."""

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "VerlRandomPrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlRandomPrunedQwen2_5VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlRandomPrunedQwen3VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlRandomPrunedQwen3VLForConditionalGeneration",
    )
