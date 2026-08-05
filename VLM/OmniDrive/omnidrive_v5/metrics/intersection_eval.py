import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from omnidrive_v5.utils.paths import DEFAULT_DATASET_VERSION, LLM_ROOT, layout, resolve_dataset_version_paths


REFERENCE_EVAL = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OmniDrive SunLakes V5/V6 predictions with the patched reference VQA evaluator."
    )
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--pred-dir", default=None, help="Directory with per-frame prediction JSON files.")
    parser.add_argument("--predictions-jsonl", default=None, help="Optional pre-merged prediction JSONL.")
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


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def iter_prediction_records(pred_dir: Path) -> Iterable[Dict]:
    ensure(pred_dir.is_dir(), f"Prediction directory not found: {pred_dir}")
    json_paths = sorted(pred_dir.glob("*.json"))
    ensure(json_paths, f"No prediction JSON files found in {pred_dir}")
    for path in json_paths:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            for row in payload:
                yield row
        elif isinstance(payload, dict):
            yield payload
        else:
            raise TypeError(f"Unsupported prediction payload in {path}: {type(payload).__name__}")


def merge_predictions(pred_dir: Path) -> List[Dict[str, str]]:
    merged: Dict[str, str] = {}
    for row in iter_prediction_records(pred_dir):
        question_id = row.get("question_id")
        prediction = row.get("prediction")
        if question_id is None or prediction is None:
            continue
        merged[str(question_id)] = str(prediction)
    ensure(merged, f"No question_id/prediction pairs found in {pred_dir}")
    return [{"question_id": question_id, "prediction": merged[question_id]} for question_id in sorted(merged)]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_optional_label_parser(namespace: Dict, field_name: str):
    normalize_parse_text = namespace["normalize_parse_text"]

    def parser(text: str) -> Dict[str, object]:
        normalized = normalize_parse_text(text)
        value = None if normalized in {"", "none", "n/a", "na"} else normalized
        return {field_name: value}

    return parser


def patch_reference_templates(namespace: Dict, dataset_version: str) -> None:
    if dataset_version != "v5":
        return
    template_cls = namespace["TemplateEvalSpec"]
    specs = namespace["V5_TEMPLATE_SPECS"]
    specs.update(
        {
            "1_2_2_visibility": template_cls(
                parser=make_optional_label_parser(namespace, "visibility"),
                discrete_fields=lambda _gt: ["visibility"],
            ),
            "1_3_1_weather": template_cls(
                parser=make_optional_label_parser(namespace, "weather"),
                discrete_fields=lambda _gt: ["weather"],
            ),
            "1_3_2_vehicle_signal_state": template_cls(
                parser=make_optional_label_parser(namespace, "signal_state"),
                discrete_fields=lambda _gt: ["signal_state"],
            ),
        }
    )


def run_reference_eval(args: argparse.Namespace, predictions_jsonl: Path, evaluator: Path) -> None:
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
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    sidecar_jsonl = Path(args.sidecar_jsonl).resolve() if args.sidecar_jsonl else (prepared_dir / "intersection_vqa_eval_sidecar.jsonl")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (layout(work_dir)["metrics"])
    output_dir.mkdir(parents=True, exist_ok=True)
    args.sidecar_jsonl = str(sidecar_jsonl)
    args.output_dir = str(output_dir)

    if args.predictions_jsonl:
        predictions_jsonl = Path(args.predictions_jsonl).resolve()
        ensure(predictions_jsonl.is_file(), f"Predictions JSONL not found: {predictions_jsonl}")
    else:
        ensure(args.pred_dir is not None, "Either --pred-dir or --predictions-jsonl must be provided.")
        predictions_jsonl = output_dir / "merged_predictions.jsonl"
        dump_jsonl(predictions_jsonl, merge_predictions(Path(args.pred_dir).resolve()))

    run_reference_eval(args, predictions_jsonl, Path(resolved["evaluator"]).resolve())


if __name__ == "__main__":
    main()
