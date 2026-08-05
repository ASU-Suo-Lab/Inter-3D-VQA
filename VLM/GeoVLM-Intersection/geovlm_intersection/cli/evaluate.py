from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from geovlm_intersection.config.common import DEFAULT_DATASET_VERSION, ensure_worktree_layout, resolve_dataset_version_paths
from geovlm_intersection.utils import dump_json, ensure, load_json, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeoVLM predictions with the reference intersection evaluator.")
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
        evaluator=args.evaluator,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(isinstance(manifest, dict), f"split_manifest.json must be an object: {prepared_dir / 'split_manifest.json'}")
    ensure(
        manifest.get("dataset_version") == resolved["dataset_version"],
        f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    predictions = Path(args.predictions).resolve() if args.predictions else worktree["predictions_merged"]
    sidecar_jsonl = Path(args.sidecar_jsonl).resolve() if args.sidecar_jsonl else prepared_dir / "sidecar_val.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else worktree["metrics"]
    evaluator = Path(resolved["evaluator"]).resolve()
    ensure(predictions.is_file(), f"Missing predictions file: {predictions}")
    ensure(sidecar_jsonl.is_file(), f"Missing sidecar file: {sidecar_jsonl}")
    ensure(evaluator.is_file(), f"Missing evaluator: {evaluator}")
    prediction_rows = load_jsonl(predictions)
    sidecar_rows = load_jsonl(sidecar_jsonl)
    prediction_ids = {str(row["question_id"]) for row in prediction_rows}
    sidecar_ids = {str(row["question_id"]) for row in sidecar_rows}
    ensure(
        prediction_ids == sidecar_ids,
        f"Prediction coverage mismatch: predictions={len(prediction_ids)} sidecar={len(sidecar_ids)} intersection={len(prediction_ids & sidecar_ids)}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(evaluator),
        "--predictions",
        str(predictions),
        "--sidecar-jsonl",
        str(sidecar_jsonl),
        "--output-dir",
        str(output_dir),
        "--split",
        args.split,
    ]
    if args.skip_semantic_metrics:
        command.append("--skip-semantic-metrics")
    subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]), check=True)
    dump_json(
        worktree["metrics_run_json"],
        {
            "dataset_version": resolved["dataset_version"],
            "prepared_dir": str(prepared_dir),
            "predictions": str(predictions),
            "sidecar_jsonl": str(sidecar_jsonl),
            "evaluator": str(evaluator),
            "split": args.split,
            "partial": False,
            "requested_count": len(sidecar_ids),
            "evaluated_count": len(prediction_ids),
            "prediction_count": len(prediction_ids),
            "prediction_coverage": 1.0,
            "skip_semantic_metrics": args.skip_semantic_metrics,
        },
    )


if __name__ == "__main__":
    main()
