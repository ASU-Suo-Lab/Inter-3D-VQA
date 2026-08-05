from __future__ import annotations

import argparse
import json
from pathlib import Path

from geovlm_intersection.backbones import load_lion_runtime, load_qwen3_vl_runtime, prepare_qwen3_vl_inputs
from geovlm_intersection.config.common import (
    DATASET_VERSION_DEFAULTS,
    DEFAULT_INFO_PKL,
    DEFAULT_LION_QUALITY,
    ensure_worktree_layout,
    load_validated_features_manifest,
    resolve_dataset_version_paths,
    validate_dataset_version,
)
from geovlm_intersection.data import build_info_index, load_point_cloud_xyzit, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check GeoVLM-Intersection v5 backbone and sample prerequisites.")
    parser.add_argument("--dataset-version", default="v5", choices=["v5"], help="GeoVLM currently supports only v5.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared directory to validate.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional work directory used to validate extracted features.")
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL, help="SunLakes infos pickle.")
    parser.add_argument("--split", default="val_eval", help="Prepared split JSON to validate.")
    parser.add_argument("--sample-index", type=int, default=0, help="Prepared sample index used for smoke checks.")
    parser.add_argument("--lion-quality", default=DEFAULT_LION_QUALITY, choices=["low", "mid", "high"], help="LION checkpoint tier.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_version = validate_dataset_version(args.dataset_version)
    resolved = resolve_dataset_version_paths(
        dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    val_rows = load_prepared_records(prepared_dir, split="val")
    val_eval_rows = load_prepared_records(prepared_dir, split="val_eval")
    val_question_ids = {str(row["question_id"]) for row in val_rows}
    val_eval_question_ids = {str(row["question_id"]) for row in val_eval_rows}
    if val_question_ids != val_eval_question_ids:
        raise ValueError(
            "GeoVLM expects val and val_eval to share one extracted feature set, but their question_id sets differ."
        )
    records = load_prepared_records(prepared_dir, split=args.split)
    if not records:
        raise ValueError(f"Prepared split is empty: {prepared_dir / f'{args.split}.json'}")
    if args.sample_index < 0 or args.sample_index >= len(records):
        raise IndexError(f"Sample index out of range: {args.sample_index} for {len(records)} records")

    qa_json = Path(DATASET_VERSION_DEFAULTS[dataset_version]["qa_json"]).resolve()
    prompt_bundle = load_prompt_bundle(qa_json)
    validate_subtemplates((record["subtemplate"] for record in records), prompt_bundle)

    info_index = build_info_index(args.info_pkl.resolve())
    sample = resolve_prepared_sample(
        records[args.sample_index],
        info_index,
        dataset_version=dataset_version,
        prepared_split=args.split,
        prepared_index=args.sample_index,
    )
    point_cloud = load_point_cloud_xyzit(sample.point_cloud_path)
    qwen_runtime = load_qwen3_vl_runtime()
    qwen_inputs = prepare_qwen3_vl_inputs(qwen_runtime, sample, prompt_bundle=prompt_bundle)
    lion_runtime = load_lion_runtime(args.lion_quality)
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    features_manifest_summary: dict[str, object] | None = None
    if worktree["features_manifest"].is_file():
        features_manifest = load_validated_features_manifest(
            worktree,
            dataset_version=dataset_version,
            required_splits=("val", "val_eval"),
        )
        features_manifest_summary = {
            "feature_storage": features_manifest["feature_storage"],
            "feature_layout_version": features_manifest["feature_layout_version"],
            "split_aliases": features_manifest["split_aliases"],
            "splits": sorted(features_manifest["splits"]),
        }
    else:
        stale_val_eval_frame_dir = worktree["features"] / "val_eval" / "frames"
        if stale_val_eval_frame_dir.is_dir():
            has_payloads = any(path.is_file() for path in stale_val_eval_frame_dir.iterdir())
            if has_payloads:
                raise ValueError(
                    "Found stale features/val_eval/frames without a valid feature manifest. "
                    "Remove stale val_eval frame payloads and re-run extract."
                )

    summary = {
        "dataset_version": dataset_version,
        "prepared_dir": str(prepared_dir),
        "work_dir": str(worktree["work_dir"]),
        "prepared_split": args.split,
        "prepared_records": len(records),
        "sample_index": args.sample_index,
        "sample": {
            "question_id": sample.question_id,
            "frame_token": sample.frame_token,
            "subtemplate": sample.subtemplate,
            "question": sample.question,
            "image_paths": [str(path) for path in sample.image_paths],
            "point_cloud_path": str(sample.point_cloud_path),
            "point_cloud_shape": list(point_cloud.shape),
            "camera_views": [
                {
                    "cam_key": view.cam_key,
                    "image_name": view.image_name,
                    "view_direction": view.view_direction,
                }
                for view in sample.camera_views
            ],
            "camera_intrinsics_shapes": [list(view.camera_intrinsics.shape) for view in sample.camera_views],
            "lidar2image_shapes": [list(view.lidar2image.shape) for view in sample.camera_views],
            "structured_target_keys": sorted((sample.structured_targets or {}).keys()),
        },
        "qwen3_vl": {
            "model_dir": str(qwen_runtime.model_dir),
            "processor": qwen_runtime.processor_name,
            "config": qwen_runtime.config_name,
            "text_hidden_size": qwen_runtime.text_hidden_size,
            "vision_hidden_size": qwen_runtime.vision_hidden_size,
            "qa_json": str(prompt_bundle["qa_json"]),
            "prompt_mode": prompt_bundle["prompt_mode"],
            "prompt_version": prompt_bundle["prompt_version"],
            "subtemplate_patch_style": prompt_bundle["subtemplate_patch_style"],
            "question_input_ids_shape": list(qwen_inputs.question_input_ids.shape),
            "image_patch_tokens_shape": list(qwen_inputs.image_patch_tokens.shape),
            "image_patch_mask_shape": list(qwen_inputs.image_patch_mask.shape),
            "image_grid_thw_shape": list(qwen_inputs.image_grid_thw.shape),
            "image_token_counts": list(qwen_inputs.image_token_counts),
            "system_prompt_length": len(qwen_inputs.system_prompt),
            "user_prompt_length": len(qwen_inputs.user_prompt),
            "full_prompt_length": len(qwen_inputs.full_prompt),
        },
        "lion": {
            "config_path": str(lion_runtime.config_path),
            "checkpoint_path": str(lion_runtime.checkpoint_path),
            "backbone_class": lion_runtime.backbone_class.__name__,
            "detector_class": lion_runtime.detector_class.__name__,
            "config_name": lion_runtime.config_name,
        },
        "features_manifest": features_manifest_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
