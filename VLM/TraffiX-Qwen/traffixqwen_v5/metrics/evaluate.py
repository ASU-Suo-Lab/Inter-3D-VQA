#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from traffixqwen_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_EVAL_DIR, DEFAULT_PREPARED_DIR, resolve_dataset_version_paths
from traffixqwen_v5.data.common import load_json


DEFAULT_SIDECAR = DEFAULT_PREPARED_DIR / "sidecar_val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TraffiX-Qwen Intersection V5 predictions.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_reference_eval(args: argparse.Namespace) -> None:
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    evaluator = Path(resolved["evaluator"]).resolve()
    #ensure(evaluator.is_file(), f"Reference evaluator not found: {evaluator}")
    evaluator_parent = str(evaluator.parent)
    added_path = False
    if evaluator_parent not in sys.path:
        sys.path.insert(0, evaluator_parent)
        added_path = True
    try:
        namespace = runpy.run_path(str(evaluator))
    finally:
        if added_path:
            sys.path.pop(0)
            
    argv = [
        str(evaluator),
        "--sidecar-jsonl",
        str(Path(args.sidecar_jsonl).resolve()),
        "--predictions",
        str(Path(args.predictions).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--split",
        args.split,
        "--bertscore-model",
        args.bertscore_model,
        "--sim-model",
        args.sim_model,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.skip_semantic_metrics:
        argv.append("--skip-semantic-metrics")
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])

    old_argv = sys.argv
    try:
        sys.argv = argv
        namespace["main"]()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    manifest_path = Path(resolved["prepared_dir"]) / "split_manifest.json"
    require(manifest_path.is_file(), f"Prepared split manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    require(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )
    if args.sidecar_jsonl is None:
        args.sidecar_jsonl = str(Path(resolved["prepared_dir"]) / "sidecar_val.jsonl")
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else Path(resolved["work_dir"]) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    run_reference_eval(args)


if __name__ == "__main__":
    main()
