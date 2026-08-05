from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bevLLM")

from bevllm_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_METRIC_DIR, DEFAULT_PREDS_JSONL, resolve_dataset_version_paths
from bevllm_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BEV-LLM Intersection V5 predictions.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
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


def iter_prediction_records(pred_dir: Path) -> Iterable[Dict]:
    ensure(pred_dir.is_dir(), f"Prediction directory not found: {pred_dir}")
    json_paths = sorted(list(pred_dir.glob("*.json")) + list(pred_dir.glob("*.jsonl")))
    ensure(json_paths, f"No prediction files found in {pred_dir}")
    for path in json_paths:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix == ".jsonl":
                for line in file:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
                continue
            payload = json.load(file)
        if isinstance(payload, list):
            for row in payload:
                yield row
        elif isinstance(payload, dict):
            yield payload
        else:
            raise TypeError(f"Unsupported prediction payload in {path}: {type(payload).__name__}")


def merge_predictions(pred_dir: Path) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in iter_prediction_records(pred_dir):
        question_id = row.get("question_id")
        prediction = row.get("prediction")
        if question_id is None or prediction is None:
            continue
        merged[str(question_id)] = {"question_id": str(question_id), "prediction": str(prediction)}
    ensure(merged, f"No question_id/prediction pairs found in {pred_dir}")
    return [merged[question_id] for question_id in sorted(merged)]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_reference_eval(args: argparse.Namespace, predictions_jsonl: Path) -> None:
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    evaluator = Path(resolved["evaluator"]).resolve()
    ensure(evaluator.is_file(), f"Reference evaluator not found: {evaluator}")
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
        str(predictions_jsonl.resolve()),
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
    if args.sidecar_jsonl is None:
        args.sidecar_jsonl = str(Path(resolved["prepared_dir"]) / "sidecar_val.jsonl")
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else Path(resolved["work_dir"]) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(resolved["prepared_dir"]) / "split_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )

    predictions_path = Path(args.predictions).resolve() if args.predictions is not None else Path(resolved["work_dir"]) / "predictions" / Path(DEFAULT_PREDS_JSONL).name
    if predictions_path.is_file():
        predictions_jsonl = predictions_path
    else:
        pred_dir = Path(args.pred_dir or predictions_path).resolve()
        predictions_jsonl = output_dir / "merged_predictions.jsonl"
        dump_jsonl(predictions_jsonl, merge_predictions(pred_dir))

    run_reference_eval(args, predictions_jsonl)


if __name__ == "__main__":
    main()
