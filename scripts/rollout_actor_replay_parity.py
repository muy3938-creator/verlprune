#!/usr/bin/env python3
"""Measure vLLM rollout versus verl actor replay log-probability drift.

The rollout and actor stages run in separate processes so each backend can use
the full GPU. The actor stage deliberately calls DataParallelPPOActor's real
remove-padding micro-batch path instead of a bespoke step-by-step reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from unittest.mock import patch


def _make_image(path: Path, variant: int):
    from PIL import Image, ImageDraw

    palettes = [
        ((220, 30, 30), (30, 80, 220)),
        ((30, 170, 70), (180, 40, 190)),
        ((235, 145, 25), (25, 160, 175)),
        ((90, 45, 190), (210, 185, 20)),
    ]
    left, right = palettes[variant % len(palettes)]
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    inset = (variant % 3) * 3
    draw.rectangle((24 + inset, 64, 96 + inset, 136), fill=left)
    draw.ellipse((132 - inset, 64, 204 - inset, 136), fill=right)
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


def run_rollout(
    model_path: str,
    output_dir: Path,
    *,
    keep_ratio: float,
    prune_after_layer: int,
    batch_size: int,
    response_length: int,
    selector: str,
    selector_input: str,
    selector_kwargs: dict,
    pre_pruning_backend: str,
    temperature: float,
    seed: int,
    gpu_memory_utilization: float,
    baseline_unpruned: bool,
    plugin_no_prune: bool,
    prefill_keep_ratio: float | None,
    prefill_selector: str,
    prefill_selector_kwargs: dict,
) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PLUGINS", "vision_opd_token_pruning")

    from transformers import AutoConfig, AutoProcessor
    from vllm import LLM, SamplingParams

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.transport import (
        decode_vllm_dynamic_selection_capture,
        decode_vllm_selection_capture,
        decode_vllm_two_stage_selection_capture,
    )

    if temperature <= 0:
        raise ValueError("actor parity requires a positive rollout temperature")
    if baseline_unpruned and plugin_no_prune:
        raise ValueError("native baseline and plugin no-prune modes are mutually exclusive")
    if plugin_no_prune and selector_input == "decode_query":
        raise ValueError("plugin no-prune diagnostics require a static selector")
    effective_keep_ratio = 1.0 - 1e-9 if plugin_no_prune else keep_ratio
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = AutoConfig.from_pretrained(model_path).model_type
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = _prompt(processor)
    image_paths = []
    images = []
    for index in range(batch_size):
        image_path = output_dir / f"shapes-{index}.png"
        image_paths.append(str(image_path))
        images.append(_make_image(image_path, index))

    pruning_config = (
        VisionTokenPruningConfig()
        if baseline_unpruned
        else VisionTokenPruningConfig(
            enabled=True,
            keep_ratio=effective_keep_ratio,
            prune_after_layer=prune_after_layer,
            layerwise_backend="flex",
            pre_pruning_backend=pre_pruning_backend,
            selector=selector,
            selector_kwargs=selector_kwargs,
            selector_input=selector_input,
            prefill_keep_ratio=prefill_keep_ratio,
            prefill_selector=prefill_selector,
            prefill_selector_kwargs=prefill_selector_kwargs,
        )
    )
    processed = processor(text=[prompt], images=[images[0]], return_tensors="pt")
    grid = processed["image_grid_thw"][0].tolist()
    image_token_id = getattr(
        processor,
        "image_token_id",
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    )
    original_visual_tokens = int(processed["input_ids"].eq(image_token_id).sum())
    layerwise = prune_after_layer >= 0

    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=batch_size,
        max_num_batched_tokens=(16384 if model_type == "qwen3_vl" else max(512, batch_size * 160)),
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=False,
    )
    if baseline_unpruned:
        llm_kwargs["enable_chunked_prefill"] = False
    else:
        physical_architectures = {
            "qwen2_5_vl": "VerlPrunedQwen2_5VLForConditionalGeneration",
            "qwen3_vl": "VerlPrunedQwen3VLForConditionalGeneration",
        }
        layerwise_architectures = {
            "qwen2_5_vl": "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration",
            "qwen3_vl": "VerlLayerwiseFlexPrunedQwen3VLForConditionalGeneration",
        }
        llm_kwargs.update(
            enable_return_routed_experts=True,
            video_pruning_rate=1.0 - (prefill_keep_ratio or effective_keep_ratio),
            enable_chunked_prefill=not layerwise,
            hf_overrides={
                "architectures": [
                    (layerwise_architectures if layerwise else physical_architectures)[model_type]
                ],
                "text_config": {
                    "num_experts_per_tok": int(selector_kwargs.get("capture_capacity", 64))
                    if selector_input == "decode_query"
                    else 1
                },
                "vision_token_pruning": pruning_config.to_backend_payload(),
            },
        )
        if layerwise:
            llm_kwargs["attention_config"] = {"backend": "FLEX_ATTENTION"}
    llm = LLM(
        **llm_kwargs,
    )
    requests = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
    outputs = llm.generate(
        requests,
        SamplingParams(
            max_tokens=response_length,
            min_tokens=response_length,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            seed=seed,
            logprobs=20,
        ),
    )

    samples = []
    for index, request_output in enumerate(outputs):
        generated = request_output.outputs[0]
        if generated.logprobs is None:
            raise RuntimeError("vLLM did not return rollout log probabilities")
        token_ids = [int(token_id) for token_id in generated.token_ids]
        rollout_log_probs = [
            float(token_log_probs[token_id].logprob)
            for token_id, token_log_probs in zip(
                generated.token_ids,
                generated.logprobs,
                strict=True,
            )
        ]
        if len(token_ids) != response_length:
            raise RuntimeError(f"request {index} generated {len(token_ids)} tokens; expected {response_length}")
        if baseline_unpruned:
            selection_wire = None
        elif prefill_keep_ratio is not None:
            selection = decode_vllm_two_stage_selection_capture(
                generated.routed_experts,
                prefill_keep_ratio=prefill_keep_ratio,
                prefill_selector=prefill_selector,
                prefill_selector_kwargs=prefill_selector_kwargs,
                decode_keep_ratio=effective_keep_ratio,
                decode_selector=selector,
                decode_selector_kwargs=selector_kwargs,
            )
            selection_wire = selection.to_wire()
        elif selector_input == "decode_query":
            selection = decode_vllm_dynamic_selection_capture(
                generated.routed_experts,
                nominal_keep_ratio=effective_keep_ratio,
                original_visual_token_count=original_visual_tokens,
                selector=selector,
                selector_kwargs=selector_kwargs,
            )
            selection_wire = selection.to_wire()
        else:
            selection = decode_vllm_selection_capture(
                generated.routed_experts,
                keep_ratio=effective_keep_ratio,
                original_visual_token_count=original_visual_tokens,
                selector=selector,
                selector_kwargs=selector_kwargs,
            )
            selection_wire = selection.to_wire()
            if plugin_no_prune and len(selection.kept_visual_indices) != original_visual_tokens:
                raise RuntimeError(
                    "plugin no-prune diagnostic unexpectedly removed visual tokens: "
                    f"kept {len(selection.kept_visual_indices)} of {original_visual_tokens}"
                )
        samples.append(
            {
                "sample_index": index,
                "image_path": image_paths[index],
                "token_ids": token_ids,
                "rollout_log_probs": rollout_log_probs,
                "generated_text": generated.text,
                "selection": selection_wire,
            }
        )

    payload = {
        "model_path": model_path,
        "prompt": prompt,
        "image_grid_thw": grid,
        "original_visual_tokens": original_visual_tokens,
        "pruning_enabled": not baseline_unpruned,
        "plugin_no_prune": plugin_no_prune,
        "keep_ratio": effective_keep_ratio,
        "requested_keep_ratio": keep_ratio,
        "prune_after_layer": prune_after_layer,
        "batch_size": batch_size,
        "response_length": response_length,
        "selector": selector,
        "selector_input": selector_input,
        "selector_kwargs": selector_kwargs,
        "pre_pruning_backend": pre_pruning_backend,
        "prefill_keep_ratio": prefill_keep_ratio,
        "prefill_selector": prefill_selector,
        "prefill_selector_kwargs": prefill_selector_kwargs,
        "temperature": temperature,
        "seed": seed,
        "samples": samples,
    }
    (output_dir / "rollout.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ROLLOUT_ACTOR_PARITY_ROLLOUT=PASS batch={batch_size} response_length={response_length}")


def _actor_config(pruning_config):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "vision_token_pruning": pruning_config,
            "use_remove_padding": True,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1,
            "use_dynamic_bsz": False,
            "use_prefix_grouper": False,
            "entropy_from_logits_with_chunking": False,
            "use_torch_compile": False,
            "fsdp_config": {"dtype": "bfloat16"},
            "calculate_sum_pi_squared": False,
            "sum_pi_squared_checkpointing": False,
            "entropy_checkpointing": False,
            "policy_loss": {"loss_mode": "vanilla"},
        }
    )


def _build_actor_batch(reference: dict, processor, device):
    import torch
    from PIL import Image

    if processor.config.model_type == "qwen3_vl":
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index
    from verl.models.vision_token_pruning.training import attach_selection_to_multi_modal_inputs

    input_ids = []
    attention_masks = []
    position_ids = []
    responses = []
    multi_modal_inputs = []
    for sample in reference["samples"]:
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt_inputs = processor(
            text=[reference["prompt"]],
            images=[image],
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"][0]
        response = torch.tensor(sample["token_ids"], dtype=torch.long)
        full_ids = torch.cat((prompt_ids, response), dim=0)
        full_attention = torch.ones_like(full_ids)
        vision_positions = get_rope_index(
            processor,
            input_ids=full_ids,
            image_grid_thw=prompt_inputs["image_grid_thw"],
            attention_mask=full_attention,
        )
        text_positions = torch.arange(len(full_ids), dtype=torch.long).unsqueeze(0)
        full_positions = torch.cat((text_positions, vision_positions), dim=0)
        model_inputs = {
            "pixel_values": prompt_inputs["pixel_values"].to(device),
            "image_grid_thw": prompt_inputs["image_grid_thw"].to(device),
        }
        selection = sample.get("selection")
        multi_modal_inputs.append(
            attach_selection_to_multi_modal_inputs(model_inputs, selection) if selection is not None else model_inputs
        )
        input_ids.append(full_ids)
        attention_masks.append(full_attention)
        position_ids.append(full_positions)
        responses.append(response)

    lengths = {len(value) for value in input_ids}
    if len(lengths) != 1:
        raise ValueError("actor parity currently requires equal fixed sequence lengths")
    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "position_ids": torch.stack(position_ids).to(device),
        "responses": torch.stack(responses).to(device),
        "multi_modal_inputs": multi_modal_inputs,
    }


def run_actor(case_dirs: list[Path]) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    if not case_dirs:
        raise ValueError("no rollout cases found")
    references = [json.loads((case_dir / "rollout.json").read_text(encoding="utf-8")) for case_dir in case_dirs]
    model_paths = {reference["model_path"] for reference in references}
    if len(model_paths) != 1:
        raise ValueError("all actor cases must use the same model")
    model_path = next(iter(model_paths))
    processor = AutoProcessor.from_pretrained(model_path)
    processor.config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda()
    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True)
    model.eval()
    model.config.use_cache = False

    for case_dir, reference in zip(case_dirs, references, strict=True):
        pruning_config = (
            VisionTokenPruningConfig()
            if not reference.get("pruning_enabled", True)
            else VisionTokenPruningConfig(
                enabled=True,
                keep_ratio=float(reference["keep_ratio"]),
                prune_after_layer=int(reference["prune_after_layer"]),
                layerwise_backend="flex",
                pre_pruning_backend=str(reference["pre_pruning_backend"]),
                selector=str(reference["selector"]),
                selector_input=str(reference["selector_input"]),
                selector_kwargs=dict(reference["selector_kwargs"]),
                prefill_keep_ratio=reference.get("prefill_keep_ratio"),
                prefill_selector=str(reference.get("prefill_selector", "embedding_norm")),
                prefill_selector_kwargs=dict(reference.get("prefill_selector_kwargs", {})),
            )
        )
        with patch("torch.distributed.get_rank", return_value=0):
            actor = DataParallelPPOActor(
                _actor_config(pruning_config),
                model,
                actor_optimizer=object(),
            )
        micro_batch = _build_actor_batch(reference, processor, torch.device("cuda"))
        with torch.no_grad():
            outputs = actor._forward_micro_batch(
                micro_batch,
                temperature=float(reference["temperature"]),
                calculate_entropy=False,
            )
        actor_log_probs = outputs["log_probs"].float().cpu()
        expected_shape = (
            int(reference["batch_size"]),
            int(reference["response_length"]),
        )
        if tuple(actor_log_probs.shape) != expected_shape:
            raise RuntimeError(f"actor returned shape {tuple(actor_log_probs.shape)}; expected {expected_shape}")
        payload = {
            "actor_path": "DataParallelPPOActor._forward_micro_batch",
            "use_remove_padding": True,
            "attention_backend": model.config._attn_implementation,
            "actor_log_probs": actor_log_probs.tolist(),
        }
        (case_dir / "actor.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summarize_case(case_dir)
        print(f"ROLLOUT_ACTOR_PARITY_ACTOR=PASS case={case_dir.name}")


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def parity_metrics(
    rollout_log_probs: list[list[float]],
    actor_log_probs: list[list[float]],
) -> dict:
    if len(rollout_log_probs) != len(actor_log_probs) or not rollout_log_probs:
        raise ValueError("rollout and actor batches must be non-empty and have equal size")
    response_lengths = [len(row) for row in rollout_log_probs]
    if not response_lengths or min(response_lengths) == 0:
        raise ValueError("response length must be positive")
    signed = []
    per_position = [[] for _ in range(max(response_lengths))]
    sequence_log_ratios = []
    rollout_probabilities = []
    actor_probabilities = []
    for rollout_row, actor_row in zip(rollout_log_probs, actor_log_probs, strict=True):
        if len(rollout_row) != len(actor_row):
            raise ValueError("each rollout and actor parity row must have the same response length")
        sequence_delta = 0.0
        for position, (rollout_value, actor_value) in enumerate(zip(rollout_row, actor_row, strict=True)):
            delta = actor_value - rollout_value
            signed.append(delta)
            per_position[position].append(abs(delta))
            sequence_delta += delta
            rollout_probabilities.append(math.exp(rollout_value))
            actor_probabilities.append(math.exp(actor_value))
        sequence_log_ratios.append(sequence_delta)

    absolute = sorted(abs(value) for value in signed)
    probability_differences = [
        abs(actor - rollout) for actor, rollout in zip(actor_probabilities, rollout_probabilities, strict=True)
    ]
    importance_ratios = [math.exp(value) for value in signed]
    mean_signed = sum(signed) / len(signed)
    rollout_probability_mean = sum(rollout_probabilities) / len(rollout_probabilities)
    actor_probability_mean = sum(actor_probabilities) / len(actor_probabilities)
    centered_rollout = [value - rollout_probability_mean for value in rollout_probabilities]
    centered_actor = [value - actor_probability_mean for value in actor_probabilities]
    denominator = math.sqrt(
        sum(value * value for value in centered_rollout) * sum(value * value for value in centered_actor)
    )
    correlation = (
        sum(left * right for left, right in zip(centered_rollout, centered_actor, strict=True)) / denominator
        if denominator
        else 1.0
    )
    return {
        "token_count": len(signed),
        "response_length": response_lengths[0] if len(set(response_lengths)) == 1 else None,
        "response_length_min": min(response_lengths),
        "response_length_max": max(response_lengths),
        "sampled_logprob_signed_diff_mean": mean_signed,
        "sampled_logprob_abs_diff_mean": sum(absolute) / len(absolute),
        "sampled_logprob_abs_diff_p50": _quantile(absolute, 0.50),
        "sampled_logprob_abs_diff_p95": _quantile(absolute, 0.95),
        "sampled_logprob_abs_diff_p99": _quantile(absolute, 0.99),
        "sampled_logprob_abs_diff_max": absolute[-1],
        "prefill_token_abs_diff_mean": sum(per_position[0]) / len(per_position[0]),
        "decode_token_abs_diff_mean": (
            sum(sum(values) for values in per_position[1:]) / sum(len(values) for values in per_position[1:])
            if max(response_lengths) > 1
            else 0.0
        ),
        "probability_abs_diff_mean": sum(probability_differences) / len(probability_differences),
        "probability_abs_diff_max": max(probability_differences),
        "probability_pearson": correlation,
        "direct_kl_sample_estimate": -mean_signed,
        "k3_kl_sample_estimate": sum(math.exp(value) - value - 1.0 for value in signed) / len(signed),
        "token_importance_ratio_mean": sum(importance_ratios) / len(importance_ratios),
        "token_importance_ratio_min": min(importance_ratios),
        "token_importance_ratio_max": max(importance_ratios),
        "fraction_ratio_outside_0_9_1_1": sum(value < 0.9 or value > 1.1 for value in importance_ratios)
        / len(importance_ratios),
        "fraction_ratio_outside_0_8_1_2": sum(value < 0.8 or value > 1.2 for value in importance_ratios)
        / len(importance_ratios),
        "sequence_log_ratio_abs_mean": sum(abs(value) for value in sequence_log_ratios) / len(sequence_log_ratios),
        "sequence_log_ratio_abs_max": max(abs(value) for value in sequence_log_ratios),
        "fraction_abs_diff_gt_0_05": sum(value > 0.05 for value in absolute) / len(absolute),
        "fraction_abs_diff_gt_0_10": sum(value > 0.10 for value in absolute) / len(absolute),
        "per_position_abs_diff_mean": [sum(values) / len(values) for values in per_position],
    }


def summarize_case(case_dir: Path) -> dict:
    rollout = json.loads((case_dir / "rollout.json").read_text(encoding="utf-8"))
    actor = json.loads((case_dir / "actor.json").read_text(encoding="utf-8"))
    rollout_log_probs = [sample["rollout_log_probs"] for sample in rollout["samples"]]
    metrics = parity_metrics(rollout_log_probs, actor["actor_log_probs"])
    payload = {
        "case": {
            **{
                key: rollout[key]
                for key in (
                    "keep_ratio",
                    "prune_after_layer",
                    "batch_size",
                    "response_length",
                    "selector",
                    "selector_input",
                    "selector_kwargs",
                    "pre_pruning_backend",
                    "temperature",
                )
            },
            "prefill_keep_ratio": rollout.get("prefill_keep_ratio"),
            "prefill_selector": rollout.get("prefill_selector"),
            "pruning_enabled": rollout.get("pruning_enabled", True),
            "plugin_no_prune": rollout.get("plugin_no_prune", False),
        },
        "metrics": metrics,
    }
    (case_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def discover_case_dirs(cases_root: Path) -> list[Path]:
    return sorted(path.parent for path in cases_root.glob("*/rollout.json"))


def _absolute_difference_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def compare_plugin_no_prune_backend_pairs(records: list[tuple]) -> dict:
    pairs = {}
    for name, rollout, actor, _ in records:
        if not rollout.get("plugin_no_prune", False):
            continue
        key = (
            int(rollout["prune_after_layer"]),
            int(rollout["batch_size"]),
            int(rollout["response_length"]),
            int(rollout["seed"]),
            str(rollout["selector"]),
        )
        pairs.setdefault(key, {})[str(rollout["pre_pruning_backend"])] = (
            name,
            rollout,
            actor,
        )

    exact_rollout_differences = []
    common_prefix_rollout_differences = []
    exact_actor_differences = []
    exact_sequence_count = 0
    sequence_count = 0
    matching_token_positions = 0
    token_position_count = 0
    pair_payload = {}
    for key, backends in sorted(pairs.items()):
        if set(backends) != {"flex", "flash"}:
            continue
        flex_name, flex_rollout, flex_actor = backends["flex"]
        flash_name, flash_rollout, flash_actor = backends["flash"]
        pair_exact = 0
        pair_sequences = 0
        pair_matching = 0
        pair_positions = 0
        for flex_sample, flash_sample, flex_log_probs, flash_log_probs in zip(
            flex_rollout["samples"],
            flash_rollout["samples"],
            flex_actor["actor_log_probs"],
            flash_actor["actor_log_probs"],
            strict=True,
        ):
            flex_tokens = flex_sample["token_ids"]
            flash_tokens = flash_sample["token_ids"]
            pair_sequences += 1
            pair_positions += len(flex_tokens)
            pair_matching += sum(
                flex_token == flash_token
                for flex_token, flash_token in zip(
                    flex_tokens,
                    flash_tokens,
                    strict=True,
                )
            )
            common_prefix_length = 0
            for flex_token, flash_token in zip(flex_tokens, flash_tokens, strict=True):
                if flex_token != flash_token:
                    break
                common_prefix_length += 1
            common_prefix_rollout_differences.extend(
                abs(flex_value - flash_value)
                for flex_value, flash_value in zip(
                    flex_sample["rollout_log_probs"][:common_prefix_length],
                    flash_sample["rollout_log_probs"][:common_prefix_length],
                    strict=True,
                )
            )
            if flex_tokens == flash_tokens:
                pair_exact += 1
                exact_rollout_differences.extend(
                    abs(flex_value - flash_value)
                    for flex_value, flash_value in zip(
                        flex_sample["rollout_log_probs"],
                        flash_sample["rollout_log_probs"],
                        strict=True,
                    )
                )
                exact_actor_differences.extend(
                    abs(flex_value - flash_value)
                    for flex_value, flash_value in zip(
                        flex_log_probs,
                        flash_log_probs,
                        strict=True,
                    )
                )
        exact_sequence_count += pair_exact
        sequence_count += pair_sequences
        matching_token_positions += pair_matching
        token_position_count += pair_positions
        pair_payload["|".join(str(value) for value in key)] = {
            "flex_case": flex_name,
            "hybrid_case": flash_name,
            "sequence_count": pair_sequences,
            "exact_sequence_count": pair_exact,
            "matching_token_positions": pair_matching,
            "token_position_count": pair_positions,
        }

    return {
        "pair_count": len(pair_payload),
        "sequence_count": sequence_count,
        "exact_sequence_count": exact_sequence_count,
        "exact_sequence_rate": exact_sequence_count / sequence_count if sequence_count else None,
        "matching_token_positions": matching_token_positions,
        "token_position_count": token_position_count,
        "token_position_match_rate": matching_token_positions / token_position_count if token_position_count else None,
        "same_prefix_rollout_logprob_abs_diff": _absolute_difference_stats(common_prefix_rollout_differences),
        "exact_sequence_rollout_logprob_abs_diff": _absolute_difference_stats(exact_rollout_differences),
        "exact_sequence_actor_logprob_abs_diff": _absolute_difference_stats(exact_actor_differences),
        "pairs": pair_payload,
    }


def aggregate_cases(case_dirs: list[Path]) -> dict:
    records = []
    for case_dir in case_dirs:
        rollout = json.loads((case_dir / "rollout.json").read_text(encoding="utf-8"))
        actor = json.loads((case_dir / "actor.json").read_text(encoding="utf-8"))
        comparison = summarize_case(case_dir)
        records.append((case_dir.name, rollout, actor, comparison))

    groups = {
        "baseline_unpruned": lambda name, rollout: not rollout.get("pruning_enabled", True),
        "core_all_flex": lambda name, rollout: (
            name.startswith("uniform-")
            and rollout.get("pruning_enabled", True)
            and not rollout.get("plugin_no_prune", False)
            and rollout["pre_pruning_backend"] == "flex"
        ),
        "core_hybrid": lambda name, rollout: (
            name.startswith("uniform-")
            and rollout.get("pruning_enabled", True)
            and not rollout.get("plugin_no_prune", False)
            and rollout["pre_pruning_backend"] == "flash"
        ),
        "expanded_hybrid": lambda name, rollout: (
            rollout.get("pruning_enabled", True) and rollout["pre_pruning_backend"] == "flash"
        ),
        "paper_algorithms_hybrid": lambda name, rollout: name.startswith("algorithm-"),
        "plugin_no_prune_all_flex": lambda name, rollout: (
            rollout.get("plugin_no_prune", False) and rollout["pre_pruning_backend"] == "flex"
        ),
        "plugin_no_prune_hybrid": lambda name, rollout: (
            rollout.get("plugin_no_prune", False) and rollout["pre_pruning_backend"] == "flash"
        ),
        "all_pruned": lambda name, rollout: rollout.get("pruning_enabled", True),
        "all_cases": lambda name, rollout: True,
    }
    group_payload = {}
    for group_name, predicate in groups.items():
        selected = [record for record in records if predicate(record[0], record[1])]
        if not selected:
            group_payload[group_name] = {
                "case_count": 0,
                "case_names": [],
                "metrics": None,
            }
            continue
        rollout_rows = [sample["rollout_log_probs"] for _, rollout, _, _ in selected for sample in rollout["samples"]]
        actor_rows = [row for _, _, actor, _ in selected for row in actor["actor_log_probs"]]
        group_payload[group_name] = {
            "case_count": len(selected),
            "case_names": [record[0] for record in selected],
            "metrics": parity_metrics(rollout_rows, actor_rows),
        }
    return {
        "case_count": len(records),
        "groups": group_payload,
        "plugin_no_prune_backend_pairs": compare_plugin_no_prune_backend_pairs(records),
        "cases": {name: comparison for name, _, _, comparison in records},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rollout", "actor", "compare", "aggregate"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rollout_actor_parity"))
    parser.add_argument("--cases-root", type=Path)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--prune-after-layer", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--response-length", type=int, default=16)
    parser.add_argument("--selector", default="uniform")
    parser.add_argument(
        "--selector-input",
        choices=("vision_embedding", "decoder_key", "decode_query"),
        default="vision_embedding",
    )
    parser.add_argument("--selector-kwargs", type=json.loads, default={})
    parser.add_argument("--prefill-keep-ratio", type=float)
    parser.add_argument("--prefill-selector", default="embedding_norm")
    parser.add_argument("--prefill-selector-kwargs", type=json.loads, default={})
    parser.add_argument("--pre-pruning-backend", choices=("flex", "flash"), default="flash")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.28)
    parser.add_argument("--baseline-unpruned", action="store_true")
    parser.add_argument("--plugin-no-prune", action="store_true")
    args = parser.parse_args()

    if args.stage == "rollout":
        run_rollout(
            args.model,
            args.output_dir,
            keep_ratio=args.keep_ratio,
            prune_after_layer=args.prune_after_layer,
            batch_size=args.batch_size,
            response_length=args.response_length,
            selector=args.selector,
            selector_input=args.selector_input,
            selector_kwargs=args.selector_kwargs,
            pre_pruning_backend=args.pre_pruning_backend,
            temperature=args.temperature,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            baseline_unpruned=args.baseline_unpruned,
            plugin_no_prune=args.plugin_no_prune,
            prefill_keep_ratio=args.prefill_keep_ratio,
            prefill_selector=args.prefill_selector,
            prefill_selector_kwargs=args.prefill_selector_kwargs,
        )
        return

    case_dirs = discover_case_dirs(args.cases_root) if args.cases_root else [args.output_dir]
    if args.stage == "actor":
        run_actor(case_dirs)
    elif args.stage == "compare":
        for case_dir in case_dirs:
            payload = summarize_case(case_dir)
            print(case_dir.name, json.dumps(payload["metrics"], sort_keys=True))
    else:
        if args.cases_root is None:
            raise ValueError("aggregate stage requires --cases-root")
        payload = aggregate_cases(case_dirs)
        (args.cases_root / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"ROLLOUT_ACTOR_PARITY_AGGREGATE=PASS cases={payload['case_count']}")


if __name__ == "__main__":
    main()
