#!/usr/bin/env python3
"""Single-GPU end-to-end smoke test for Vision-OPD token pruning.

Run the two stages in separate processes so the vLLM engine releases its GPU
memory before the actor/teacher training step starts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def _make_image(path: Path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 64, 96, 136), fill=(220, 30, 30))
    draw.ellipse((132, 64, 204, 136), fill=(30, 80, 220))
    image.save(path)
    return image


def run_rollout(
    model_path: str,
    output_dir: Path,
    keep_ratio: float,
    selector: str,
    selector_kwargs: dict,
    prune_after_layer: int,
    batch_size: int,
    layerwise_backend: str,
    pre_pruning_backend: str,
    selector_input: str,
    gpu_memory_utilization: float,
    warmup_runs: int,
    measure_runs: int,
    fixed_output_tokens: bool,
) -> None:
    if warmup_runs < 0 or measure_runs < 1:
        raise ValueError("warmup_runs must be >= 0 and measure_runs must be >= 1")
    # Importing and constructing vLLM before processor tensor work keeps its
    # worker process independent from a parent CUDA context.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_PLUGINS", "vision_opd_token_pruning")
    from vllm import LLM, SamplingParams

    from verl.models.vision_token_pruning.config import VisionTokenPruningConfig
    from verl.models.vision_token_pruning.transport import (
        decode_vllm_dynamic_selection_capture,
        decode_vllm_selection_capture,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "shapes.png"
    image = _make_image(image_path)

    layerwise = prune_after_layer >= 0
    architecture = "VerlPrunedQwen2_5VLForConditionalGeneration"
    if layerwise:
        architecture = (
            "VerlLayerwiseFlexPrunedQwen2_5VLForConditionalGeneration"
            if layerwise_backend == "flex"
            else "VerlLayerwisePrunedQwen2_5VLForConditionalGeneration"
        )
    pruning_config = VisionTokenPruningConfig(
        enabled=True,
        keep_ratio=keep_ratio,
        selector=selector,
        selector_kwargs=selector_kwargs,
        prune_after_layer=prune_after_layer,
        layerwise_backend=layerwise_backend,
        pre_pruning_backend=pre_pruning_backend,
        selector_input=selector_input,
    ).to_backend_payload()

    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=batch_size,
        # The prompt used here is about 100 tokens after visual expansion.
        # Scale the scheduler budget so batch-size stress tests are genuine
        # concurrent prefills rather than silently split by a 512-token cap.
        max_num_batched_tokens=max(512, batch_size * 128),
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_return_routed_experts=True,
        video_pruning_rate=1.0 - keep_ratio,
        enable_chunked_prefill=not layerwise,
        enable_prefix_caching=False,
        hf_overrides={
            "architectures": [architecture],
            "text_config": {
                "num_experts_per_tok": int(selector_kwargs.get("capture_capacity", 64))
                if selector_input == "decode_query"
                else 1
            },
            "vision_token_pruning": pruning_config,
        },
    )
    if layerwise and layerwise_backend == "flex":
        llm_kwargs["attention_config"] = {"backend": "FLEX_ATTENTION"}
    llm = LLM(**llm_kwargs)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
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
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[prompt], images=[image], return_tensors="pt")
    grid = processed["image_grid_thw"][0].tolist()
    merge_size = int(processor.image_processor.merge_size)
    original_visual_tokens = int(grid[0] * grid[1] * grid[2] // (merge_size**2))

    requests = [
        {"prompt": prompt, "multi_modal_data": {"image": image}}
        for _ in range(batch_size)
    ]
    sampling_params = SamplingParams(
        max_tokens=20,
        min_tokens=20 if fixed_output_tokens else 0,
        temperature=0.0,
    )
    for _ in range(warmup_runs):
        llm.generate(requests, sampling_params)

    measured_results = []
    rollout_latencies = []
    for _ in range(measure_runs):
        rollout_start = time.perf_counter()
        results = llm.generate(requests, sampling_params)
        rollout_latencies.append(time.perf_counter() - rollout_start)
        measured_results.append(results)
    results = measured_results[-1]
    rollout_seconds = sum(rollout_latencies)
    measured_output_tokens = sum(
        len(item.outputs[0].token_ids)
        for run_results in measured_results
        for item in run_results
    )
    decoded_results = []
    for request_index, request_output in enumerate(results):
        result = request_output.outputs[0]
        if selector_input == "decode_query":
            selection = decode_vllm_dynamic_selection_capture(
                result.routed_experts,
                nominal_keep_ratio=keep_ratio,
                original_visual_token_count=original_visual_tokens,
                selector=selector,
                selector_kwargs=selector_kwargs,
            )
            active_counts = [
                len(indices)
                for indices in selection.query_kept_visual_indices
                if indices
            ]
            selection_summary = f"dynamic_counts={active_counts}"
        else:
            selection = decode_vllm_selection_capture(
                result.routed_experts,
                keep_ratio=keep_ratio,
                original_visual_token_count=original_visual_tokens,
                selector=selector,
                selector_kwargs=selector_kwargs,
            )
            selection_summary = f"visual_tokens={original_visual_tokens}->{len(selection.kept_visual_indices)}"
        decoded_results.append((result, selection))
        print(
            f"request={request_index} generated_tokens={len(result.token_ids)} "
            f"{selection_summary}"
        )
    result, selection = decoded_results[0]
    record = {
        "model_path": model_path,
        "image_path": str(image_path),
        "prompt": prompt,
        "generated_token_ids": list(result.token_ids),
        "generated_text": result.text,
        "selection": selection.to_wire(),
        "image_grid_thw": grid,
        "prune_after_layer": prune_after_layer,
        "selector": selector,
        "selector_kwargs": selector_kwargs,
        "rollout_batch_size": batch_size,
        "layerwise_backend": layerwise_backend,
        "pre_pruning_backend": pre_pruning_backend,
        "selector_input": selector_input,
        "rollout_seconds": rollout_seconds,
        "rollout_latencies": rollout_latencies,
        "median_rollout_seconds": statistics.median(rollout_latencies),
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "fixed_output_tokens": fixed_output_tokens,
        "output_tokens_per_second": measured_output_tokens / rollout_seconds,
    }
    (output_dir / "rollout.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"rollout_text={result.text!r}")
    print(f"image_grid_thw={grid}")
    print(f"generated_tokens={len(result.token_ids)}")
    if selector_input == "decode_query":
        active_rows = [indices for indices in selection.query_kept_visual_indices if indices]
        print(f"dynamic_query_count={len(active_rows)}")
        print(f"dynamic_keep_counts={[len(indices) for indices in active_rows]}")
        print(f"first_dynamic_indices={list(active_rows[0]) if active_rows else []}")
    else:
        print(f"visual_tokens={original_visual_tokens}->{len(selection.kept_visual_indices)}")
    print(f"prune_after_layer={prune_after_layer}")
    print(f"selector={selector}")
    print(f"selector_kwargs={selector_kwargs}")
    print(f"layerwise_backend={layerwise_backend}")
    print(f"pre_pruning_backend={pre_pruning_backend}")
    print(f"selector_input={selector_input}")
    print(f"rollout_seconds={rollout_seconds:.6f}")
    print(f"rollout_latencies={rollout_latencies}")
    print(f"median_rollout_seconds={record['median_rollout_seconds']:.6f}")
    print(f"output_tokens_per_second={record['output_tokens_per_second']:.6f}")
    if selector_input != "decode_query":
        print(f"first_kept_indices={list(selection.kept_visual_indices[:8])}")
        print(f"last_kept_index={selection.kept_visual_indices[-1]}")
    print("E2E_STAGE_ROLLOUT=PASS")


def _distillation_loss(student_logits, teacher_logits, response_ids):
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F

    from verl.trainer.ppo.core_algos import compute_self_distillation_loss

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    topk = min(100, student_log_probs.shape[-1])
    student_topk_log_probs, topk_indices = student_log_probs.topk(topk, dim=-1)
    teacher_topk_log_probs = teacher_log_probs.gather(-1, topk_indices)
    student_token_log_probs = student_log_probs.gather(-1, response_ids[..., None]).squeeze(-1)
    teacher_token_log_probs = teacher_log_probs.gather(-1, response_ids[..., None]).squeeze(-1)
    response_mask = torch.ones_like(student_token_log_probs)
    config = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=topk,
        distillation_add_tail=True,
        alpha=0.5,
        is_clip=2.0,
    )
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_token_log_probs,
        teacher_log_probs=teacher_token_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        old_log_probs=student_token_log_probs.detach(),
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        self_distillation_mask=torch.ones(student_logits.shape[0], device=student_logits.device),
    )
    return loss, metrics


def run_train_step(output_dir: Path, learning_rate: float, steps: int) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.qwen2_vl import get_rope_index
    from verl.models.vision_token_pruning.protocol import (
        DynamicVisionTokenSelection,
        VisionTokenSelection,
        selection_from_wire,
    )
    from verl.models.vision_token_pruning.runtime import (
        KEEP_MASK_KEY,
        attach_selection_to_multi_modal_inputs,
        replay_rollout_selection_on_attention_mask,
    )
    from verl.models.vision_token_pruning.training import replay_dynamic_rollout_selection

    record = json.loads((output_dir / "rollout.json").read_text(encoding="utf-8"))
    prune_after_layer = int(record.get("prune_after_layer", -1))
    layerwise = prune_after_layer >= 0
    model_path = record["model_path"]
    image = Image.open(record["image_path"]).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    prompt_inputs = processor(text=[record["prompt"]], images=[image], return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"]
    response_ids = torch.tensor([record["generated_token_ids"]], dtype=prompt_ids.dtype)
    if response_ids.numel() == 0:
        raise RuntimeError("rollout produced no response tokens")
    full_input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    full_attention_mask = torch.ones_like(full_input_ids)

    selection = selection_from_wire(record["selection"])
    multimodal_inputs = attach_selection_to_multi_modal_inputs({}, selection.to_wire())
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    dynamic = isinstance(selection, DynamicVisionTokenSelection)
    if dynamic:
        dynamic_attention_mask = replay_dynamic_rollout_selection(
            full_input_ids,
            full_attention_mask,
            [multimodal_inputs],
            image_token_id=image_token_id,
            expected_keep_ratio=selection.nominal_keep_ratio,
            expected_selector=selection.selector,
            expected_selector_kwargs=record.get("selector_kwargs", {}),
        )
        student_attention_mask = full_attention_mask
    else:
        assert isinstance(selection, VisionTokenSelection)
        dynamic_attention_mask = None
        student_attention_mask = replay_rollout_selection_on_attention_mask(
            full_input_ids,
            full_attention_mask,
            [multimodal_inputs],
            image_token_id=image_token_id,
            expected_keep_ratio=selection.keep_ratio,
            expected_selector=selection.selector,
            expected_selector_kwargs=record.get("selector_kwargs", {}),
        )
    compact_keep = student_attention_mask[0].bool()
    student_input_ids = full_input_ids if layerwise else full_input_ids[:, compact_keep]

    vision_position_ids = get_rope_index(
        processor,
        input_ids=full_input_ids[0],
        image_grid_thw=prompt_inputs["image_grid_thw"],
        attention_mask=full_attention_mask[0],
    )
    text_position_ids = torch.arange(full_input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    full_position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
    teacher_position_ids = full_position_ids.unsqueeze(1)
    student_position_ids = (
        teacher_position_ids if layerwise else full_position_ids[:, compact_keep].unsqueeze(1)
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda()
    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=True)
    model.train()
    model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_module = model.model.language_model.layers[-1].self_attn.o_proj
    for parameter in train_module.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    full_input_ids = full_input_ids.cuda()
    student_input_ids = student_input_ids.cuda()
    response_ids = response_ids.cuda()
    pixel_values = prompt_inputs["pixel_values"].cuda()
    image_grid_thw = prompt_inputs["image_grid_thw"].cuda()
    teacher_position_ids = teacher_position_ids.cuda()
    student_position_ids = student_position_ids.cuda()
    keep_mask = multimodal_inputs.get(KEEP_MASK_KEY)
    if keep_mask is not None:
        keep_mask = keep_mask.cuda()
    layerwise_attention_mask = student_attention_mask.cuda()
    if dynamic_attention_mask is not None:
        dynamic_attention_mask = dynamic_attention_mask.cuda()

    response_length = response_ids.shape[1]
    teacher_prompt_length = prompt_ids.shape[1]
    student_prompt_length = (
        teacher_prompt_length
        if layerwise
        else int(student_attention_mask[:, :teacher_prompt_length].sum().item())
    )
    with torch.no_grad():
        teacher_output = model(
            input_ids=full_input_ids,
            attention_mask=None,
            position_ids=teacher_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        teacher_logits = teacher_output.logits[
            :, teacher_prompt_length - 1 : teacher_prompt_length - 1 + response_length
        ].detach()

    parameter_before = trainable[0].detach().float().clone()
    step_losses = []
    step_raw_jsd = []
    step_grad_norms = []
    metrics_before = None
    if dynamic:
        student_kwargs = {
            "vision_token_dynamic_attention_mask": dynamic_attention_mask,
            "vision_token_prune_after_layer": prune_after_layer,
        }
    elif layerwise:
        student_kwargs = {
            "vision_token_pruning_mask": layerwise_attention_mask,
            "vision_token_prune_after_layer": prune_after_layer,
        }
    else:
        student_kwargs = {"vision_token_keep_mask": keep_mask}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        student_output = model(
            input_ids=student_input_ids,
            attention_mask=None,
            position_ids=student_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            **student_kwargs,
        )
        student_logits = student_output.logits[
            :, student_prompt_length - 1 : student_prompt_length - 1 + response_length
        ]
        loss, metrics = _distillation_loss(student_logits, teacher_logits, response_ids)
        if metrics_before is None:
            metrics_before = metrics
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        step_losses.append(float(loss.detach()))
        step_raw_jsd.append(metrics["self_distillation/raw_jsd_token_mean"])
        step_grad_norms.append(float(grad_norm))
    torch.cuda.synchronize()
    parameter_delta = (trainable[0].detach().float() - parameter_before).norm().item()

    with torch.no_grad():
        updated_output = model(
            input_ids=student_input_ids,
            attention_mask=None,
            position_ids=student_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            **student_kwargs,
        )
        updated_logits = updated_output.logits[
            :, student_prompt_length - 1 : student_prompt_length - 1 + response_length
        ]
        loss_after, metrics_after = _distillation_loss(updated_logits, teacher_logits, response_ids)

    summary = {
        "generated_text": record["generated_text"],
        "full_sequence_tokens": int(full_input_ids.shape[1]),
        "student_sequence_tokens": int(student_input_ids.shape[1]),
        "full_visual_tokens": selection.original_visual_token_count,
        "student_visual_tokens": (
            [len(indices) for indices in selection.query_kept_visual_indices if indices]
            if dynamic
            else len(selection.kept_visual_indices)
        ),
        "response_tokens": response_length,
        "prune_after_layer": prune_after_layer,
        "training_steps": steps,
        "step_losses": step_losses,
        "step_raw_jsd": step_raw_jsd,
        "step_grad_norms": step_grad_norms,
        "loss_before": step_losses[0],
        "loss_after": float(loss_after.detach()),
        "raw_jsd_before": metrics_before["self_distillation/raw_jsd_token_mean"],
        "raw_jsd_after": metrics_after["self_distillation/raw_jsd_token_mean"],
        "grad_norm": step_grad_norms[-1],
        "parameter_delta_norm": parameter_delta,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "attention_backend": model.config._attn_implementation,
    }
    (output_dir / "train_step.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for key, value in summary.items():
        print(f"{key}={value}")
    if not all(
        torch.isfinite(torch.tensor(value))
        for value in (*summary["step_losses"], *summary["step_grad_norms"], summary["loss_after"])
    ):
        raise RuntimeError("training step produced a non-finite metric")
    if parameter_delta <= 0:
        raise RuntimeError("optimizer did not update the selected actor parameter")
    print("E2E_STAGE_TRAIN_STEP=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rollout", "train"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/opd_e2e"))
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--selector", default="random")
    parser.add_argument("--selector-kwargs", type=json.loads, default={})
    parser.add_argument("--prune-after-layer", type=int, default=-1)
    parser.add_argument("--layerwise-backend", choices=("flex", "compact_flash"), default="flex")
    parser.add_argument("--pre-pruning-backend", choices=("flex", "flash"), default="flex")
    parser.add_argument(
        "--selector-input",
        choices=("vision_embedding", "decoder_key", "decode_query"),
        default="vision_embedding",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    parser.add_argument("--fixed-output-tokens", action="store_true")
    args = parser.parse_args()
    if args.stage == "rollout":
        run_rollout(
            args.model,
            args.output_dir,
            args.keep_ratio,
            args.selector,
            args.selector_kwargs,
            args.prune_after_layer,
            args.batch_size,
            args.layerwise_backend,
            args.pre_pruning_backend,
            args.selector_input,
            args.gpu_memory_utilization,
            args.warmup_runs,
            args.measure_runs,
            args.fixed_output_tokens,
        )
    else:
        run_train_step(args.output_dir, args.learning_rate, args.steps)


if __name__ == "__main__":
    main()
