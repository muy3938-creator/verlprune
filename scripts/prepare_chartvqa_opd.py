#!/usr/bin/env python3
"""Convert cnb.cool/muyugooo/chartvqa rows to Vision-OPD parquet records.

ChartQA has one image column and list-valued labels. Vision-OPD expects image
and teacher-image paths plus a chat-style prompt, so this converter writes a
small local image cache and uses the original chart as the teacher image.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

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


def convert_split(dataset_source: str, split: str, output_dir: Path, limit: int) -> Path:
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
    image_dir = output_dir / "images" / split
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
        rows.append(
            {
                "data_source": "chartvqa",
                "prompt": [{"role": "user", "content": PROMPT.format(question=question)}],
                "images": [{"path": str(image_path)}],
                "bbox_images": [{"path": str(image_path)}],
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
    output_path = output_dir / f"{split}.parquet"
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
    for split, limit in (("train", args.train_limit), ("validation", args.val_limit), ("test", args.test_limit)):
        source_split = "val" if split == "validation" else split
        convert_split(args.dataset, source_split, args.output_dir, limit)


if __name__ == "__main__":
    main()
