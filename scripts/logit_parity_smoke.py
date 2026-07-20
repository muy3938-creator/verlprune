#!/usr/bin/env python3
"""Compare layerwise Transformers and vLLM Flex prefill/decode logits.

The stages run in separate processes so vLLM releases its GPU allocation
before the Transformers reference loads.  Step zero checks the final prefill
logit; later steps check decode logits that reuse the prompt KV selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _make_image(path: Path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 64, 96, 136), fill=(220, 30, 30))
    draw.ellipse((132, 64, 204, 136), fill=(30, 80, 220))
    image.save(path)
    return image


def _prompt(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Describe the two colored shapes from left to right in one short sentence.",
                },
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm(
    model_path: str,
    output_dir: Path,
    *,
    keep_ratio: float,
    prune_after_layer: int,
    max_tokens: int,
    all_logprobs: bool,
    dynamic: bool,
    selector: str,
    selector_input: str,
    selector_kwargs: dict,
    pre_pruning_backend: str,
    gpu_memory_utilization: float,
) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PLUGINS", "vision_opd_token_pruning")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "shapes.png"
    image = _make_image(image_path)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    prompt = _prompt(processor)
    config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=keep_ratio,
        prune_after_layer=prune_after_layer,
        layerwise_backend="flex",
        pre_pruning_backend=pre_pruning_backend,
        selector="vision_pulse" if dynamic else selector,
        selector_kwargs=selector_kwargs,
        selector_input="decode_query" if dynamic else selector_input,
    )
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    grid = processed["image_grid_thw"][0].tolist()
    merge_size = int(processor.image_processor.merge_size)
    original_visual_tokens = int(grid[0] * grid[1] * grid[2] // (merge_size**2))
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_return_routed_experts=True,
        video_pruning_rate=1.0 - keep_ratio,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        attention_config={"backend": "FLEX_ATTENTION"},
        hf_overrides={
            "architectures": ["VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration"],
            "text_config": {
                "num_experts_per_tok": int(selector_kwargs.get("capture_capacity", 64))
                if dynamic
                else 1
            },
            "vision_token_pruning": config.to_backend_payload(),
        },
    )
    if all_logprobs:
        llm_kwargs["max_logprobs"] = -1
    llm = LLM(**llm_kwargs)
    output = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=-1 if all_logprobs else 20,
        ),
    )[0].outputs[0]
    if output.logprobs is None:
        raise RuntimeError("vLLM did not return requested decode log-probabilities")
    steps = []
    for token_id, token_logprobs in zip(output.token_ids, output.logprobs, strict=True):
        steps.append(
            {
                "token_id": int(token_id),
                "sampled_logprob": float(token_logprobs[token_id].logprob),
                "top_logprobs": {
                    str(candidate_id): float(logprob.logprob)
                    for candidate_id, logprob in token_logprobs.items()
                },
            }
        )
    if dynamic:
        from verl.models.vision_token_pruning.transport import (
            decode_vllm_dynamic_selection_capture,
        )

        selection_wire = decode_vllm_dynamic_selection_capture(
            output.routed_experts,
            nominal_keep_ratio=keep_ratio,
            original_visual_token_count=original_visual_tokens,
            selector="vision_pulse",
            selector_kwargs=selector_kwargs,
        ).to_wire()
    else:
        from verl.models.vision_token_pruning.transport import decode_vllm_selection_capture

        selection_wire = decode_vllm_selection_capture(
            output.routed_experts,
            keep_ratio=keep_ratio,
            original_visual_token_count=original_visual_tokens,
            selector=selector,
            selector_kwargs=selector_kwargs,
        ).to_wire()
    payload = {
        "model_path": model_path,
        "image_path": str(image_path),
        "prompt": prompt,
        "keep_ratio": keep_ratio,
        "prune_after_layer": prune_after_layer,
        "generated_text": output.text,
        "all_logprobs": all_logprobs,
        "dynamic": dynamic,
        "selector": "vision_pulse" if dynamic else selector,
        "selector_input": "decode_query" if dynamic else selector_input,
        "selector_kwargs": selector_kwargs,
        "pre_pruning_backend": pre_pruning_backend,
        "selection": selection_wire,
        "steps": steps,
    }
    (output_dir / "vllm.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"generated_token_ids={[step['token_id'] for step in steps]}")
    print(f"generated_text={output.text!r}")
    print("LOGIT_PARITY_VLLM=PASS")


def run_transformers(output_dir: Path) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.qwen2_vl import get_rope_index
    from verl.models.vision_token_pruning.protocol import VisionTokenSelection
    from verl.models.vision_token_pruning.training import (
        attach_selection_to_multi_modal_inputs,
        replay_dynamic_rollout_selection,
    )

    reference = json.loads((output_dir / "vllm.json").read_text(encoding="utf-8"))
    model_path = reference["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    image = Image.open(reference["image_path"]).convert("RGB")
    prompt_inputs = processor(
        text=[reference["prompt"]],
        images=[image],
        return_tensors="pt",
    )
    input_ids = prompt_inputs["input_ids"].cuda()
    pixel_values = prompt_inputs["pixel_values"].cuda()
    image_grid_thw = prompt_inputs["image_grid_thw"].cuda()
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    dynamic = bool(reference.get("dynamic", False))
    selected = None
    full_dynamic_mask = None
    if dynamic:
        full_replay_ids = torch.cat(
            (
                input_ids,
                input_ids.new_tensor([[step["token_id"] for step in reference["steps"]]]),
            ),
            dim=1,
        ).cpu()
        full_replay_attention = torch.ones_like(full_replay_ids)
        attached = attach_selection_to_multi_modal_inputs({}, reference["selection"])
        full_dynamic_mask = replay_dynamic_rollout_selection(
            full_replay_ids,
            full_replay_attention,
            [attached],
            image_token_id=image_token_id,
            expected_keep_ratio=float(reference["keep_ratio"]),
            expected_selector="vision_pulse",
            expected_selector_kwargs=reference["selector_kwargs"],
        ).cuda()
    else:
        selected = torch.tensor(
            VisionTokenSelection.from_wire(reference["selection"]).kept_visual_indices,
            dtype=torch.long,
            device="cuda",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda()
    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True)
    model.eval()
    model.config.use_cache = False

    steps = []
    with torch.no_grad():
        for vllm_step in reference["steps"]:
            sequence_length = input_ids.shape[1]
            pruning_kwargs = {}
            if dynamic:
                pruning_kwargs["vision_token_dynamic_attention_mask"] = full_dynamic_mask[
                    :, :sequence_length, :sequence_length
                ]
            else:
                keep_mask = torch.ones((1, sequence_length), dtype=torch.bool, device="cuda")
                keep_mask[0, image_positions] = False
                keep_mask[0, image_positions[selected]] = True
                pruning_kwargs["vision_token_pruning_mask"] = keep_mask
            full_attention_mask = torch.ones_like(input_ids)
            vision_position_ids = get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=full_attention_mask[0],
            )
            text_position_ids = torch.arange(
                sequence_length,
                dtype=torch.long,
                device="cuda",
            ).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0).unsqueeze(1)
            output = model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                vision_token_prune_after_layer=int(reference["prune_after_layer"]),
                **pruning_kwargs,
            )
            logprobs = output.logits[0, -1].float().log_softmax(dim=-1)
            argmax_token_id = int(logprobs.argmax())
            forced_token_id = int(vllm_step["token_id"])
            top_values, top_indices = logprobs.topk(20)
            vllm_logprobs = vllm_step["top_logprobs"]
            compared_token_ids = torch.tensor(
                [int(token_id) for token_id in vllm_logprobs],
                dtype=torch.long,
                device="cuda",
            )
            vllm_values = torch.tensor(
                [float(value) for value in vllm_logprobs.values()],
                dtype=torch.float32,
                device="cuda",
            )
            all_differences = (
                logprobs.index_select(0, compared_token_ids) - vllm_values
            ).abs()
            steps.append(
                {
                    "forced_token_id": forced_token_id,
                    "argmax_token_id": argmax_token_id,
                    "sampled_logprob": float(logprobs[forced_token_id]),
                    "top_logprobs": {
                        str(int(token_id)): float(logprob)
                        for token_id, logprob in zip(top_indices, top_values, strict=True)
                    },
                    "compared_logprob_count": len(compared_token_ids),
                    "all_logprob_abs_diff_mean": float(all_differences.mean()),
                    "all_logprob_abs_diff_p99": float(
                        torch.quantile(all_differences, 0.99)
                    ),
                    "all_logprob_abs_diff_max": float(all_differences.max()),
                }
            )
            input_ids = torch.cat(
                (input_ids, input_ids.new_tensor([[forced_token_id]])),
                dim=1,
            )

    payload = {
        "selected_visual_indices": selected.tolist() if selected is not None else None,
        "steps": steps,
    }
    (output_dir / "transformers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"selected_visual_indices={selected.tolist() if selected is not None else 'dynamic'}")
    print(f"argmax_token_ids={[step['argmax_token_id'] for step in steps]}")
    print("LOGIT_PARITY_TRANSFORMERS=PASS")


def compare(output_dir: Path, *, sampled_tolerance: float) -> None:
    vllm_result = json.loads((output_dir / "vllm.json").read_text(encoding="utf-8"))
    transformers_result = json.loads(
        (output_dir / "transformers.json").read_text(encoding="utf-8")
    )
    comparisons = []
    for step_index, (vllm_step, transformers_step) in enumerate(
        zip(vllm_result["steps"], transformers_result["steps"], strict=True)
    ):
        shared_top_tokens = set(vllm_step["top_logprobs"]).intersection(
            transformers_step["top_logprobs"]
        )
        top_differences = [
            abs(
                vllm_step["top_logprobs"][token]
                - transformers_step["top_logprobs"][token]
            )
            for token in shared_top_tokens
        ]
        comparisons.append(
            {
                "step": step_index,
                "phase": "prefill" if step_index == 0 else "decode",
                "token_id": vllm_step["token_id"],
                "transformers_argmax_token_id": transformers_step["argmax_token_id"],
                "sampled_logprob_abs_diff": abs(
                    vllm_step["sampled_logprob"] - transformers_step["sampled_logprob"]
                ),
                "shared_top20_tokens": len(shared_top_tokens),
                "max_shared_top20_logprob_abs_diff": max(top_differences, default=0.0),
                "compared_logprob_count": transformers_step["compared_logprob_count"],
                "all_logprob_abs_diff_mean": transformers_step["all_logprob_abs_diff_mean"],
                "all_logprob_abs_diff_p99": transformers_step["all_logprob_abs_diff_p99"],
                "all_logprob_abs_diff_max": transformers_step["all_logprob_abs_diff_max"],
            }
        )
    payload = {
        "all_greedy_tokens_match": all(
            item["token_id"] == item["transformers_argmax_token_id"]
            for item in comparisons
        ),
        "max_sampled_logprob_abs_diff": max(
            item["sampled_logprob_abs_diff"] for item in comparisons
        ),
        "sampled_tolerance": sampled_tolerance,
        "comparisons": comparisons,
    }
    payload["within_tolerance"] = (
        payload["all_greedy_tokens_match"]
        and payload["max_sampled_logprob_abs_diff"] <= sampled_tolerance
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["within_tolerance"]:
        raise RuntimeError("vLLM Flex and Transformers logits did not meet parity tolerance")
    print("LOGIT_PARITY_COMPARE=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("vllm", "transformers", "compare"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/flex_logit_parity"))
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--prune-after-layer", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--sampled-tolerance", type=float, default=0.15)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--all-logprobs", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--selector", default="uniform")
    parser.add_argument(
        "--selector-input",
        choices=("vision_embedding", "decoder_key"),
        default="vision_embedding",
    )
    parser.add_argument("--selector-kwargs", type=json.loads, default={})
    parser.add_argument("--pre-pruning-backend", choices=("flex", "flash"), default="flex")
    args = parser.parse_args()
    if args.stage == "vllm":
        run_vllm(
            args.model,
            args.output_dir,
            keep_ratio=args.keep_ratio,
            prune_after_layer=args.prune_after_layer,
            max_tokens=args.max_tokens,
            all_logprobs=args.all_logprobs,
            dynamic=args.dynamic,
            selector=args.selector,
            selector_input=args.selector_input,
            selector_kwargs=args.selector_kwargs,
            pre_pruning_backend=args.pre_pruning_backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    elif args.stage == "transformers":
        run_transformers(args.output_dir)
    else:
        compare(args.output_dir, sampled_tolerance=args.sampled_tolerance)


if __name__ == "__main__":
    main()
