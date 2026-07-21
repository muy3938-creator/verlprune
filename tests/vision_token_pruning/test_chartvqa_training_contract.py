from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_converter_module():
    script = ROOT / "scripts/prepare_chartvqa_opd.py"
    spec = importlib.util.spec_from_file_location("prepare_chartvqa_opd", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return _TinyDataset(self.rows[index] for index in indices)


def test_chartvqa_converter_writes_identical_full_teacher_and_student_inputs(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    Image = pytest.importorskip("PIL.Image")
    converter = _load_converter_module()
    original = Image.new("RGB", (37, 23), color=(12, 34, 56))
    source = tmp_path / "chartvqa"
    (source / "data").mkdir(parents=True)
    (source / "data/train-00000-of-00001.parquet").touch()
    monkeypatch.setattr(
        converter,
        "load_dataset",
        lambda *args, **kwargs: _TinyDataset(
            [{"image": original, "query": "What is the largest bar?", "label": ["A"]}]
        ),
    )

    output = converter.convert_split(str(source), "train", tmp_path / "output", 1)
    row = datasets.Dataset.from_parquet(str(output))[0]

    assert row["prompt"] == row["teacher_prompt"]
    assert row["images"] == row["teacher_images"]
    assert "bbox_images" not in row
    assert row["prompt"][0]["content"].count("<image>") == 1
    assert "What is the largest bar?" in row["prompt"][0]["content"]
    image_path = Path(row["images"][0]["path"])
    with Image.open(image_path) as saved:
        assert saved.size == original.size
        assert saved.convert("RGB").getpixel((0, 0)) == original.getpixel((0, 0))


def test_chartvqa_converter_uses_validation_output_name(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    Image = pytest.importorskip("PIL.Image")
    converter = _load_converter_module()
    source = tmp_path / "chartvqa"
    (source / "data").mkdir(parents=True)
    (source / "data/val-00000-of-00001.parquet").touch()
    monkeypatch.setattr(
        converter,
        "load_dataset",
        lambda *args, **kwargs: _TinyDataset(
            [{"image": Image.new("RGB", (2, 2)), "query": "Q", "label": ["A"]}]
        ),
    )

    output = converter.convert_split(
        str(source),
        "val",
        tmp_path / "output",
        1,
        output_split="validation",
    )

    assert output.name == "validation.parquet"
    assert output.is_file()


def test_chartvqa_launcher_pins_same_input_and_fixed_teacher_contract():
    launcher = (ROOT / "scripts/run_chartvqa_opd_training.sh").read_text()
    lora_config = (ROOT / "verl/trainer/config/chartvqa_opd_lora.yaml").read_text()
    full_config = (ROOT / "verl/trainer/config/chartvqa_opd_full_training.yaml").read_text()

    for expected in (
        "CONFIG_NAME=\"${TRAINING_CONFIG}\"",
        "teacher_model_path=\"${TEACHER_MODEL_PATH}\"",
        "teacher_image_key=teacher_images",
        "teacher_prompt_mode=null",
        "teacher_update_rate=0.0",
        "max_reprompt_len=\"${MAX_PROMPT_LENGTH}\"",
    ):
        assert expected in launcher
    assert 'USE_WANDB="${USE_WANDB:-true}"' in launcher
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.10}"' in launcher
    assert "trainer.logger=['console']" in launcher
    assert "ray stop --force" in launcher
    assert "trap cleanup_runtime EXIT INT TERM" in launcher
    assert "./gg keepalive process" in launcher
    for config in (lora_config, full_config):
        assert "teacher_model_source: fixed" in config
        assert "teacher_model_path: ${actor_rollout_ref.model.path}" in config
        assert "teacher_image_key: teacher_images" in config
        assert "teacher_prompt_mode: null" in config
        assert "teacher_update_rate: 0.0" in config
        assert "keep_ratio: 0.10" in config
        assert "selector: random" in config
        assert "selector_input: vision_embedding" in config
        assert "prune_after_layer: 0" in config
        assert "lora_adapter_path: null" in config
        assert "use_lora:" in config
    assert "use_lora: true" in lora_config
    assert "lora_rank: 8" in lora_config
    assert "lr: 1.0e-5" in lora_config
    assert "use_lora: false" in full_config
    assert "lora_rank: 0" in full_config
    assert "lr: 1.0e-6" in full_config
    assert "answer_hint" not in launcher


def test_fixed_teacher_update_path_never_copies_student_parameters():
    torch = pytest.importorskip("torch")
    try:
        from verl.workers.actor.dp_actor import DataParallelPPOActor
    except ImportError as exc:
        pytest.skip(f"full verl runtime is unavailable: {exc}")

    actor = object.__new__(DataParallelPPOActor)
    actor.config = SimpleNamespace(
        policy_loss={"loss_mode": "vopd"},
        self_distillation=SimpleNamespace(
            teacher_model_source="fixed",
            teacher_regularization="ema",
            teacher_update_rate=1.0,
        ),
    )
    actor.actor_module = torch.nn.Linear(3, 2)
    actor.teacher_module = torch.nn.Linear(3, 2)
    with torch.no_grad():
        actor.actor_module.weight.fill_(7.0)
        actor.teacher_module.weight.fill_(2.0)
    before = actor.teacher_module.weight.detach().clone()

    actor._update_teacher()

    assert torch.equal(actor.teacher_module.weight, before)
    assert not torch.equal(actor.teacher_module.weight, actor.actor_module.weight)


def test_fixed_teacher_loader_explicitly_freezes_the_module():
    worker_source = (ROOT / "verl/workers/fsdp_workers.py").read_text()
    fixed_teacher_branch = worker_source[worker_source.index('if role == "teacher":') :]

    assert "actor_module.requires_grad_(False)" in fixed_teacher_branch
    assert "actor_module.eval()" in fixed_teacher_branch


def test_qwen25_transformers5_processor_binds_vision_position_helper():
    source = (ROOT / "verl/utils/tokenizer.py").read_text()
    qwen25_block = source[source.index('case "Qwen2_5_VLProcessor":') : source.index('case "Qwen3VLProcessor":')]

    assert "get_vision_position_ids" in qwen25_block
    assert "processor.get_vision_position_ids = types.MethodType" in qwen25_block


def test_teacher_rope_path_supports_transformers5_mm_token_types():
    source = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    teacher_rope_block = source[source.index('"mm_token_type_ids" in rope_index_signature.parameters') :]

    assert 'rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids' in teacher_rope_block
    assert 'rope_index_kwargs["input_ids"] = prompt_input_ids.unsqueeze(0)' in teacher_rope_block


def test_qwen25_vl_flops_counter_reads_text_config():
    torch = pytest.importorskip("torch")
    try:
        from verl.utils.flops_counter import _estimate_qwen2_flops
    except ImportError as exc:
        pytest.skip(f"full verl runtime is unavailable: {exc}")

    # Transformers 5.x Qwen2.5-VL exposes language-model fields only through
    # text_config.  The counter should remain callable after a real actor step.
    text_config = SimpleNamespace(
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
        num_key_value_heads=2,
        num_attention_heads=4,
        intermediate_size=16,
        head_dim=2,
    )
    config = SimpleNamespace(text_config=text_config)
    result = _estimate_qwen2_flops(config, 4, [4], 1.0)
    assert torch.isfinite(torch.tensor(result))
    assert result > 0
