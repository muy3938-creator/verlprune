def register_vision_token_pruning_models() -> None:
    """Register selector-driven Qwen-VL out-of-tree models."""

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "VerlPrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlPrunedQwen2_5VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlPrunedQwen3VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlPrunedQwen3VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.layerwise_flex_vision_token_pruning:"
        "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlLayerwiseFlexPrunedQwen3VLForConditionalGeneration",
        "verl.vllm_plugins.layerwise_flex_vision_token_pruning:"
        "VerlLayerwiseFlexPrunedQwen3VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.layerwise_vision_token_pruning:"
        "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration",
    )

    # Keep old architecture names loadable for existing configs and checkpoints.
    ModelRegistry.register_model(
        "VerlRandomPrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlRandomPrunedQwen2_5VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlRandomPrunedQwen3VLForConditionalGeneration",
        "verl.vllm_plugins.vision_token_pruning:VerlRandomPrunedQwen3VLForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "VerlLayerwiseRandomPrunedQwen2_5VLForConditionalGeneration",
        "verl.vllm_plugins.layerwise_vision_token_pruning:"
        "VerlLayerwiseRandomPrunedQwen2_5VLForConditionalGeneration",
    )
