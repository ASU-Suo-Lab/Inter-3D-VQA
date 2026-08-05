from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from senna_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_PREPARED_DIR, DEFAULT_WORK_DIR, resolve_dataset_version_paths, worktree_paths
from senna_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Senna V5 predictions with the shared reference evaluator.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    evaluator = Path(resolved["evaluator"]).resolve()
    predictions = Path(args.predictions).resolve() if args.predictions is not None else worktree_paths(Path(resolved["work_dir"]))["merged_predictions"]
    sidecar = Path(args.sidecar_jsonl).resolve() if args.sidecar_jsonl is not None else Path(resolved["prepared_dir"]) / "sidecar_val.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else worktree_paths(Path(resolved["work_dir"]))["metrics"]
    ensure(evaluator.is_file(), f"Evaluator not found: {evaluator}")
    ensure(predictions.is_file(), f"Predictions file not found: {predictions}")
    ensure(sidecar.is_file(), f"Sidecar file not found: {sidecar}")
    manifest_path = Path(resolved["prepared_dir"]) / "split_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )

    command = [
        sys.executable,
        str(evaluator),
        "--predictions",
        str(predictions),
        "--sidecar-jsonl",
        str(sidecar),
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
        args.device,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.skip_semantic_metrics:
        command.append("--skip-semantic-metrics")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
