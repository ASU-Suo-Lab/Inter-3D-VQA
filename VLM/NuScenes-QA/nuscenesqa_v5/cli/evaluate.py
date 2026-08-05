from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nuscenesqa_v5.config.common import DEFAULT_DATASET_VERSION, ensure_worktree_layout, resolve_dataset_version_paths
from nuscenesqa_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NuScenes-QA predictions with the reference intersection evaluator.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    manifest_path = prepared_dir / "split_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        ensure(manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared split_manifest dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    predictions = Path(args.predictions).resolve() if args.predictions else worktree["predictions"] / "merged_predictions.jsonl"
    sidecar_jsonl = Path(args.sidecar_jsonl).resolve() if args.sidecar_jsonl else prepared_dir / "sidecar_val.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else worktree["metrics"]
    evaluator = Path(resolved["evaluator"]).resolve()
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


if __name__ == "__main__":
    main()
