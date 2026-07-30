"""Register out-of-tree vLLM pruned model classes."""


def register_vision_token_pruning_models() -> None:
    """Register Pre-LLM pruned Qwen-VL out-of-tree models with vLLM."""
    try:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "VerlPrunedQwen2_5VLForConditionalGeneration",
            "verl.vllm_plugins.vllm_pre_llm_pruning:VerlPrunedQwen2_5VLForConditionalGeneration",
        )
    except ImportError:
        pass
