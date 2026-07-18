#!/usr/bin/env python3
"""Create a tiny local multimodal dataset for end-to-end Vision-OPD smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from PIL import Image, ImageDraw


def _write_images(output_dir: Path, resolution: int) -> tuple[Path, Path]:
    image_dir = output_dir / "images"
    teacher_dir = output_dir / "teacher_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(image)
    margin = resolution // 10
    center = resolution // 2
    draw.rectangle((margin, margin, center - margin // 2, resolution - margin), fill=(220, 30, 30))
    draw.ellipse((center + margin // 2, margin, resolution - margin, resolution - margin), fill=(30, 80, 220))
    student_path = image_dir / "shapes.png"
    image.save(student_path)

    teacher = image.copy()
    teacher_draw = ImageDraw.Draw(teacher)
    teacher_draw.rectangle(
        (margin // 2, margin // 2, center, resolution - margin // 2),
        outline=(255, 0, 0),
        width=max(2, resolution // 56),
    )
    teacher_path = teacher_dir / "shapes_bbox.png"
    teacher.save(teacher_path)
    return student_path, teacher_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/opd_data"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    student_path, teacher_path = _write_images(output_dir, args.resolution)
    prompt = "<image>\nDescribe the colored shapes from left to right in one short sentence."
    answer = "The left shape is a red rectangle and the right shape is a blue circle."
    records = [
        {
            "data_source": "vision_opd_smoke",
            "prompt": [{"role": "user", "content": prompt}],
            "images": [{"path": str(student_path)}],
            "bbox_images": [{"path": str(teacher_path)}],
            "ability": "visual_question_answering",
            "reward_model": {"style": "none", "ground_truth": answer},
            "extra_info": {
                "answer": answer,
                "question": prompt.replace("<image>\n", ""),
                "source_extra_info": {"synthetic": True},
            },
        }
        for _ in range(args.samples)
    ]
    output_path = output_dir / "train.parquet"
    Dataset.from_list(records).to_parquet(output_path)
    print(f"wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
