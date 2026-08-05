#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import evaluate_intersection_vqa as v5_eval


SUPPORTED_TEMPLATE_VERSIONS = {"v5", "v6"}
UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_V5_SIDECAR = Path("LlamaFactory/data/intersection_vqa/intersection_vqa_eval_sidecar.jsonl")
DEFAULT_OUTPUT_DIRS = {
    "v5": Path("LlamaFactory/eval/intersection_vqa"),
    "v6": Path("LlamaFactory/eval/intersection_vqa_v6"),
}
EVALUATOR_BY_VERSION = {
    "v5": UTILS_DIR / "evaluate_intersection_vqa.py",
    "v6": UTILS_DIR / "evaluate_intersection_vqa_v6.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically dispatch Intersection VQA evaluation to the V5 or V6 evaluator based on the sidecar template version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sidecar-jsonl", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=v5_eval.default_device())
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for debugging.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def first_nonempty_jsonl_row(path: Path) -> dict:
    require(path.is_file(), f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Sidecar is empty: {path}")


def detect_template_version(sidecar_path: Path) -> str:
    row = first_nonempty_jsonl_row(sidecar_path)
    for key in ("template_version", "dataset_version", "version"):
        value = row.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in SUPPORTED_TEMPLATE_VERSIONS:
                return normalized
    path_str = str(sidecar_path.resolve())
    if "intersection_vqa_v6" in path_str:
        return "v6"
    if "intersection_vqa" in path_str:
        return "v5"
    raise ValueError(
        f"Unable to determine template version from sidecar: {sidecar_path}. "
        "Regenerate the sidecar with template_version or use an official dataset directory."
    )


def build_command(args: argparse.Namespace, template_version: str, sidecar_path: Path) -> list[str]:
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[template_version]
    command = [
        sys.executable,
        str(EVALUATOR_BY_VERSION[template_version]),
        "--predictions",
        str(args.predictions),
        "--sidecar-jsonl",
        str(sidecar_path),
        "--output-dir",
        str(output_dir),
        "--split",
        args.split,
        "--bertscore-model",
        args.bertscore_model,
        "--sim-model",
        args.sim_model,
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
    ]
    if args.skip_semantic_metrics:
        command.append("--skip-semantic-metrics")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> None:
    args = parse_args()
    sidecar_path = args.sidecar_jsonl or DEFAULT_V5_SIDECAR
    template_version = detect_template_version(sidecar_path)
    command = build_command(args, template_version, sidecar_path)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
