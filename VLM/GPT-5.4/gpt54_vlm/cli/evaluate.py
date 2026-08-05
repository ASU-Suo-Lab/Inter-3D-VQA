from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gpt54_vlm.config.common import DEFAULT_DATASET_VERSION, ensure_worktree_layout, resolve_dataset_version_paths
from gpt54_vlm.utils.io import dump_json, dump_jsonl, ensure, load_json, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.1 intersection predictions.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def load_prediction_map(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        if isinstance(payload, dict):
            out: dict[str, str] = {}
            for key, value in payload.items():
                ensure(isinstance(value, str), f"Prediction value for question_id={key} must be a string.")
                out[str(key)] = value
            return out
        ensure(isinstance(payload, list), f"Unsupported prediction JSON payload: {type(payload)!r}")
        rows = payload
    else:
        rows = load_jsonl(path)
    out: dict[str, str] = {}
    for row in rows:
        ensure(isinstance(row, dict) and "question_id" in row, f"Unsupported prediction row: {row}")
        prediction = row.get("prediction", row.get("predict"))
        ensure(isinstance(prediction, str), f"Missing prediction text for question_id={row['question_id']}")
        out[str(row["question_id"])] = prediction
    return out


def load_sidecar_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = load_jsonl(path)
    out: list[dict[str, object]] = []
    for row in rows:
        ensure(isinstance(row, dict), f"Unsupported sidecar row: {row}")
        row_split = row.get("split")
        if row_split is not None and str(row_split) != split:
            continue
        out.append(row)
    return out


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
    ensure(isinstance(manifest, dict), "split_manifest.json must be a JSON object.")
    ensure(manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    predictions = Path(args.predictions).resolve() if args.predictions else worktree["predictions_merged"]
    sidecar_jsonl = Path(args.sidecar_jsonl).resolve() if args.sidecar_jsonl else prepared_dir / "sidecar_val.jsonl"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else worktree["metrics"]
    evaluator = Path(resolved["evaluator"]).resolve()
    ensure(predictions.is_file(), f"Missing predictions file: {predictions}")
    ensure(sidecar_jsonl.is_file(), f"Missing sidecar file: {sidecar_jsonl}")
    ensure(evaluator.is_file(), f"Missing evaluator: {evaluator}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_map = load_prediction_map(predictions)
    sidecar_rows = load_sidecar_rows(sidecar_jsonl, args.split)
    ensure(sidecar_rows, f"No sidecar samples found for split={args.split} in {sidecar_jsonl}")
    matched_sidecar_rows = [row for row in sidecar_rows if str(row["question_id"]) in prediction_map]
    ensure(matched_sidecar_rows, f"No overlapping predictions found for split={args.split} in {predictions}")
    requested_count = len(sidecar_rows)
    evaluated_count = len(matched_sidecar_rows)
    if args.allow_partial:
        partial = evaluated_count != requested_count
        eval_sidecar_jsonl = worktree["partial_sidecar_jsonl"] if partial else sidecar_jsonl
        if partial:
            dump_jsonl(eval_sidecar_jsonl, matched_sidecar_rows)
        else:
            worktree["partial_sidecar_jsonl"].unlink(missing_ok=True)
    else:
        ensure(
            evaluated_count == requested_count,
            f"Prediction coverage incomplete for split={args.split}: {evaluated_count}/{requested_count}. Re-run with --allow-partial to evaluate the completed subset.",
        )
        partial = False
        eval_sidecar_jsonl = sidecar_jsonl
    command = [
        sys.executable,
        str(evaluator),
        "--predictions",
        str(predictions),
        "--sidecar-jsonl",
        str(eval_sidecar_jsonl),
        "--output-dir",
        str(output_dir),
        "--split",
        args.split,
    ]
    if args.skip_semantic_metrics:
        command.append("--skip-semantic-metrics")
    subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]), check=True)
    forward_status = None
    if worktree["forward_run_json"].is_file():
        forward_run = load_json(worktree["forward_run_json"])
        if isinstance(forward_run, dict):
            forward_status = forward_run.get("status")
    dump_json(
        worktree["metrics_run_json"],
        {
            "dataset_version": resolved["dataset_version"],
            "predictions": str(predictions),
            "sidecar_jsonl": str(sidecar_jsonl),
            "evaluated_sidecar_jsonl": str(eval_sidecar_jsonl),
            "evaluator": str(evaluator),
            "split": args.split,
            "partial": partial,
            "requested_count": requested_count,
            "evaluated_count": evaluated_count,
            "prediction_count": len(prediction_map),
            "prediction_coverage": evaluated_count / requested_count,
            "forward_status": forward_status,
            "skip_semantic_metrics": args.skip_semantic_metrics,
        },
    )


if __name__ == "__main__":
    main()
