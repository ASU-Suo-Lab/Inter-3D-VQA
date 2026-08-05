#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path


ROLE_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}

TEXT_ONLY_SYSTEM = """You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.

You are given text metadata and object references from a synchronized intersection scene. You do not have access to images or point clouds.

Available inputs and grounding rules:
- Use only the text metadata, object references, task instructions, and numeric values explicitly present in the prompt.
- Do not claim to inspect images or infer visual details that are not present in the text.
- Object references may include global planar coordinates and image-view reference coordinates as text-only identifiers.
- Use approach names north, south, east, west, and center area consistently.
- Use lane names left-turn lane, through lane, and right-turn lane.
- Use crosswalk, entry zone, exit zone, and waiting zone consistently.

Answering rules:
- Answer only the queried task.
- Keep the answer concise and grounded in the provided text.
- Do not add extra explanation unless the task itself asks for evidence or summary.
- Keep numbers and units when the task is about counts, distances, speeds, or trajectories.
- If the task is yes/no, keep negative answers brief.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a text-only Qwen3 baseline dataset from the v5 Intersection VQA LlamaFactory dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("LlamaFactory/data/intersection_vqa"),
        help="Source multimodal v5 dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("LlamaFactory/data/intersection_vqa_text_v5"),
        help="Output text-only dataset directory.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        default="intersection_vqa_text_v5",
        help="Prefix for output files and dataset_info entries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def strip_image_prompt(content: str) -> str:
    content = re.sub(r"^(?:<image>)+\s*\n?", "", content)
    content = re.sub(
        r"Image-to-direction mapping:\n(?:- image \d+ = [^\n]+\n)+(?:\n)?",
        "",
        content,
        count=1,
    )
    return content.lstrip()


def convert_row(row: dict) -> dict:
    output = {
        "system": TEXT_ONLY_SYSTEM,
        "messages": [],
    }
    for message in row.get("messages", []):
        converted = dict(message)
        if converted.get("role") == "user" and isinstance(converted.get("content"), str):
            converted["content"] = strip_image_prompt(converted["content"])
        output["messages"].append(converted)

    return output


def convert_jsonl(source_path: Path, output_path: Path) -> int:
    count = 0
    with source_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            count += 1

    return count


def write_dataset_info(output_dir: Path, prefix: str) -> None:
    def entry(file_name: str) -> dict:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "system": "system",
            },
            "tags": ROLE_TAGS,
        }

    write_json(
        output_dir / "dataset_info.json",
        {
            f"{prefix}_train": entry(f"{prefix}_train.jsonl"),
            f"{prefix}_val": entry(f"{prefix}_val.jsonl"),
        },
    )


def write_split_summary(source_dir: Path, output_dir: Path, prefix: str, counts: dict[str, int]) -> None:
    summary_path = source_dir / "split_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    if not isinstance(summary, dict):
        summary = {}

    summary["source_dataset_dir"] = str(source_dir)
    summary["dataset_prefix"] = prefix
    summary["prompt_style"] = "text_only"
    summary["dataset_files"] = {
        "train": f"{prefix}_train.jsonl",
        "val": f"{prefix}_val.jsonl",
        "eval_sidecar": f"{prefix}_eval_sidecar.jsonl",
    }
    summary["text_only_conversion"] = {
        "source_prefix": "intersection_vqa",
        "removed_columns": ["images", "point_clouds"],
        "removed_user_prompt_blocks": ["image_placeholders", "image_to_direction_mapping"],
        "sample_counts": counts,
    }
    write_json(output_dir / "split_summary.json", summary)


def write_readme(output_dir: Path, prefix: str) -> None:
    (output_dir / "README.md").write_text(
        "# Qwen3 Text-Only Intersection VQA v5 Dataset\n\n"
        "This dataset is derived from `LlamaFactory/data/intersection_vqa` for the Qwen3 4B text baseline.\n"
        "It removes image placeholders, image path columns, and point-cloud path columns while preserving the v5 train/val split.\n\n"
        "Dataset names:\n"
        f"- `{prefix}_train`\n"
        f"- `{prefix}_val`\n\n"
        "Multi-GPU training:\n"
        "```bash\n"
        "cd /home/suolab/LLM/LlamaFactory\n"
        "CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 NPROC_PER_NODE=4 \\\n"
        "  bash examples/train_lora/intersection_qwen3_4b_lora_sft.sh\n"
        "```\n\n"
        "Multi-GPU predict + eval:\n"
        "```bash\n"
        "cd /home/suolab/LLM/LlamaFactory\n"
        "CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 NPROC_PER_NODE=4 \\\n"
        "  bash examples/train_lora/intersection_qwen3_4b_lora_predict_eval.sh\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    prefix = args.dataset_name_prefix

    if not source_dir.is_dir():
        raise SystemExit(f"source dataset directory not found: {source_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = convert_jsonl(source_dir / "intersection_vqa_train.jsonl", output_dir / f"{prefix}_train.jsonl")
    val_count = convert_jsonl(source_dir / "intersection_vqa_val.jsonl", output_dir / f"{prefix}_val.jsonl")

    source_sidecar = source_dir / "intersection_vqa_eval_sidecar.jsonl"
    output_sidecar = output_dir / f"{prefix}_eval_sidecar.jsonl"
    if not source_sidecar.is_file():
        raise SystemExit(f"source sidecar not found: {source_sidecar}")
    shutil.copyfile(source_sidecar, output_sidecar)
    sidecar_count = sum(1 for _ in output_sidecar.open("r", encoding="utf-8"))

    write_dataset_info(output_dir, prefix)
    write_split_summary(source_dir, output_dir, prefix, {"train": train_count, "val": val_count, "eval_sidecar": sidecar_count})
    write_readme(output_dir, prefix)

    print(f"Wrote text-only dataset to {output_dir}")
    print(f"train={train_count} val={val_count} eval_sidecar={sidecar_count}")


if __name__ == "__main__":
    main()
