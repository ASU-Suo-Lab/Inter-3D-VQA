from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from geovlm_intersection.models.semantic_decoder import (
    generate_decoder_outputs,
    build_frozen_semantic_decoder_runtime,
)
from geovlm_intersection.config.common import (
    DEFAULT_DATASET_VERSION,
    ensure_worktree_layout,
    load_validated_features_manifest,
    resolve_dataset_version_paths,
)
from geovlm_intersection.data import GeoVLMFeatureDataset, build_info_index, collate_feature_batch
from geovlm_intersection.pipeline_core import load_checkpoint
from geovlm_intersection.rendering import resolve_final_prediction
from geovlm_intersection.utils import dump_json, dump_jsonl, ensure, load_json


def _summarize_analysis(rows: list[dict[str, object]]) -> dict[str, object]:
    template_prediction_counts: Counter[str] = Counter()
    prediction_source_counts: Counter[str] = Counter()
    decoder_empty_count_by_template: Counter[str] = Counter()
    final_prediction_distributions: dict[str, Counter[str]] = {}
    decoder_prediction_distributions: dict[str, Counter[str]] = {}
    field_distributions: dict[str, dict[str, Counter[str]]] = {}
    selection_score_sums: Counter[str] = Counter()
    selection_score_counts: Counter[str] = Counter()

    for row in rows:
        subtemplate = str(row.get("subtemplate"))
        template_prediction_counts[subtemplate] += 1
        prediction_source = str(row.get("prediction_source") or "")
        field_counters = field_distributions.setdefault(subtemplate, {})
        if prediction_source:
            prediction_source_counts[prediction_source] += 1
            field_counters.setdefault("prediction_source", Counter())[prediction_source] += 1
        prediction = str(row.get("prediction") or "").strip()
        if prediction:
            final_prediction_distributions.setdefault(subtemplate, Counter())[prediction] += 1
        decoder_raw_output = str(row.get("decoder_raw_output") or "").strip()
        if decoder_raw_output:
            decoder_prediction_distributions.setdefault(subtemplate, Counter())[decoder_raw_output] += 1
        elif row.get("decoder_error") is not None:
            decoder_empty_count_by_template[subtemplate] += 1
        decoded = row.get("decoded_prediction")
        if not isinstance(decoded, dict):
            continue
        for field in [
            "object_type",
            "side",
            "camera_name",
            "motion_state",
            "risk_reason",
            "action_state",
            "binary_answer",
        ]:
            value = decoded.get(field)
            if value is None:
                continue
            field_counters.setdefault(field, Counter())[str(value)] += 1
        selected_score = decoded.get("selected_object_score")
        if isinstance(selected_score, (int, float)):
            selection_score_sums[subtemplate] += float(selected_score)
            selection_score_counts[subtemplate] += 1

    return {
        "prediction_count": len(rows),
        "template_prediction_counts": dict(template_prediction_counts),
        "prediction_source_counts": dict(prediction_source_counts),
        "decoder_empty_count_by_template": dict(decoder_empty_count_by_template),
        "final_prediction_distributions": {
            subtemplate: dict(counter.most_common())
            for subtemplate, counter in sorted(final_prediction_distributions.items())
        },
        "decoder_prediction_distributions": {
            subtemplate: dict(counter.most_common())
            for subtemplate, counter in sorted(decoder_prediction_distributions.items())
        },
        "field_distributions": {
            subtemplate: {
                field: dict(counter.most_common())
                for field, counter in sorted(field_map.items())
            }
            for subtemplate, field_map in sorted(field_distributions.items())
        },
        "mean_selected_object_score": {
            subtemplate: selection_score_sums[subtemplate] / selection_score_counts[subtemplate]
            for subtemplate in selection_score_sums
            if selection_score_counts[subtemplate] > 0
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoVLM forward on extracted v5 features.")
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "val_eval"], default="val_eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _select_single_output(outputs: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            selected[key] = value[index : index + 1]
        else:
            selected[key] = value
    return selected


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(isinstance(manifest, dict), f"split_manifest.json must be an object: {prepared_dir / 'split_manifest.json'}")
    ensure(
        manifest.get("dataset_version") == resolved["dataset_version"],
        f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    load_validated_features_manifest(
        worktree,
        dataset_version=resolved["dataset_version"],
        required_splits=(args.split,),
    )
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else worktree["best_checkpoint"]
    ensure(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device=args.device)
    semantic_decoder_runtime = build_frozen_semantic_decoder_runtime(device=args.device)
    ensure(
        checkpoint.get("dataset_version") == resolved["dataset_version"],
        f"Checkpoint dataset_version={checkpoint.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )
    model.eval()

    info_index = build_info_index()
    dataset = GeoVLMFeatureDataset(
        prepared_dir=prepared_dir,
        work_dir=worktree["work_dir"],
        split=args.split,
        info_index=info_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_feature_batch,
    )

    rows: list[dict[str, object]] = []
    failed_count = 0
    with torch.inference_mode():
        progress = tqdm(loader, desc="Forward", unit="batch")
        for batch in progress:
            outputs = model(
                image_tokens=batch["image_tokens"].to(args.device),
                bev_tokens=batch["bev_tokens"].to(args.device),
                object_tokens=batch["object_tokens"].to(args.device),
                raw_object_tokens=batch["raw_object_tokens"].to(args.device),
                question_tokens=batch["question_tokens"].to(args.device),
                subtemplate_ids=batch["subtemplate_index"].to(args.device),
            )
            decoder_outputs = generate_decoder_outputs(
                runtime=semantic_decoder_runtime,
                semantic_prefix_tokens=outputs["semantic_prefix_tokens"],
                prompt_texts=batch["decoder_prompt_text"],
                subtemplates=batch["subtemplate"],
            )
            for index, question_id in enumerate(batch["question_id"]):
                subtemplate = str(batch["subtemplate"][index])
                sample_outputs = _select_single_output(outputs, index)
                decoder_raw_output = decoder_outputs[index]
                prediction_error: str | None = None
                final_result = resolve_final_prediction(
                    subtemplate=subtemplate,
                    outputs=sample_outputs,
                    decoder_raw_output=decoder_raw_output,
                )
                if not final_result.prediction:
                    failed_count += 1
                    if final_result.decoder_error and final_result.structured_error:
                        prediction_error = f"{final_result.decoder_error} | {final_result.structured_error}"
                    else:
                        prediction_error = final_result.decoder_error or final_result.structured_error or "Unknown prediction failure."
                rows.append(
                    {
                        "question_id": question_id,
                        "subtemplate": subtemplate,
                        "prediction": final_result.prediction,
                        "answer": batch["prepared_record"][index]["answer"],
                        "decoded_prediction": final_result.decoded_payload,
                        "prediction_source": final_result.prediction_source,
                        "decoder_raw_output": final_result.decoder_raw_output,
                        "structured_overrides": final_result.structured_overrides,
                        "decoder_error": final_result.decoder_error,
                        "structured_error": final_result.structured_error,
                        "prediction_error": prediction_error,
                    }
                )
            progress.set_postfix(
                done=len(rows),
                errors=failed_count,
                total=len(dataset),
                question_id=str(batch["question_id"][-1]),
            )

    dump_jsonl(worktree["predictions_rank0"], rows)
    dump_jsonl(worktree["predictions_merged"], rows)
    analysis_dir = worktree["work_dir"] / "analysis"
    dump_json(analysis_dir / "prediction_analysis.json", _summarize_analysis(rows))
    dump_json(
        worktree["forward_run_json"],
        {
            "dataset_version": resolved["dataset_version"],
            "prepared_dir": str(prepared_dir),
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "requested_count": len(dataset),
            "success_count": len(rows) - failed_count,
            "failed_count": failed_count,
            "decoder_failed_count": sum(1 for row in rows if row.get("decoder_error") is not None),
            "structured_failed_count": sum(1 for row in rows if row.get("structured_error") is not None),
            "semantic_final_count": sum(1 for row in rows if row.get("prediction_source") == "decoder"),
            "hybrid_or_structured_count": sum(1 for row in rows if row.get("prediction_source") == "structured"),
            "status": "completed_with_errors" if failed_count else "completed",
            "analysis_path": str((analysis_dir / "prediction_analysis.json").resolve()),
        },
    )


if __name__ == "__main__":
    main()
