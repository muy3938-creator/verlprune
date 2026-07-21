#!/usr/bin/env python3
"""Convert ChartQA rows to Vision-OPD parquet records.

ChartQA has one complete chart image, a question, and list-valued labels.
Vision-OPD keeps two explicit image columns so the student and fixed teacher
can be audited independently: both columns point to the same complete image.
The prompt and teacher_prompt columns are also deliberately identical.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from PIL import Image


PROMPT = """<image>
You are an expert chart reasoning assistant. Analyze the chart carefully,
reason step by step about the visual evidence and any arithmetic needed, then
give the final answer in one concise sentence. Use up to five concise,
evidence-grounded reasoning steps when useful. Put the final answer inside
<answer>...</answer>. Do not invent values that are not supported by the chart.

Question: {question}"""


def decode_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"unsupported image type: {type(value).__name__}")


def convert_split(
    dataset_source: str,
    split: str,
    output_dir: Path,
    limit: int,
    output_split: Optional[str] = None,
) -> Path:
    source_path = Path(dataset_source)
    if source_path.is_dir():
        candidates = sorted((source_path / "data").glob(f"{split}-*.parquet"))
        if not candidates:
            candidates = sorted(source_path.glob(f"{split}-*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"no parquet files found for split={split} under {source_path}")
        data = load_dataset("parquet", data_files={"data": [str(path) for path in candidates]}, split="data")
    else:
        data = load_dataset(dataset_source, split=split)
    if limit > 0:
        data = data.select(range(min(limit, len(data))))
    output_split = output_split or split
    image_dir = output_dir / "images" / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(data):
        image_path = image_dir / f"{index:06d}.png"
        decode_image(item["image"]).save(image_path)
        labels = item.get("label") or []
        if isinstance(labels, str):
            labels = [labels]
        question = str(item.get("query", "")).strip()
        answer = str(labels[0]) if labels else ""
        prompt = [{"role": "user", "content": PROMPT.format(question=question)}]
        rows.append(
            {
                "data_source": "chartvqa",
                # Student and teacher receive the same instruction text.  The
                # duplicated column makes this invariant explicit in parquet.
                "prompt": prompt,
                "teacher_prompt": prompt,
                "images": [{"path": str(image_path)}],
                # This is the complete, uncropped chart, not a bbox/teacher
                # variant.  It intentionally points to the same file as images.
                "teacher_images": [{"path": str(image_path)}],
                "ability": "chart_question_answering",
                "reward_model": {"style": "none", "ground_truth": answer},
                "extra_info": {
                    "question": question,
                    "answer": answer,
                    "labels": list(labels),
                    "chartvqa_index": index,
                },
            }
        )
    output_path = output_dir / f"{output_split}.parquet"
    Dataset.from_list(rows).to_parquet(output_path)
    print(f"wrote {len(rows)} {split} rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/root/data/chartvqa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_split, source_split, limit in (
        ("train", "train", args.train_limit),
        ("validation", "val", args.val_limit),
        ("test", "test", args.test_limit),
    ):
        convert_split(args.dataset, source_split, args.output_dir, limit, output_split=output_split)


if __name__ == "__main__":
    main()
