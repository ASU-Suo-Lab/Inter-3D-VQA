from __future__ import annotations

import argparse
import json
from pathlib import Path

from geovlm_intersection.backbones import load_lion_runtime, load_qwen3_vl_runtime, prepare_qwen3_vl_inputs
from geovlm_intersection.config.common import DATASET_VERSION_DEFAULTS, DEFAULT_INFO_PKL, DEFAULT_LION_QUALITY, validate_dataset_version
from geovlm_intersection.data import build_info_index, load_point_cloud_xyzit, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.data.v5_io import resolve_default_prepared_dir
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real-sample GeoVLM input-contract smoke test on v5.")
    parser.add_argument("--dataset-version", default="v5", choices=["v5"], help="GeoVLM currently supports only v5.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared directory to use.")
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL, help="SunLakes infos pickle.")
    parser.add_argument("--split", default="val_eval", help="Prepared split JSON to inspect.")
    parser.add_argument("--sample-index", type=int, default=0, help="Prepared sample index.")
    parser.add_argument("--lion-quality", default=DEFAULT_LION_QUALITY, choices=["low", "mid", "high"], help="LION checkpoint tier.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_version = validate_dataset_version(args.dataset_version)
    prepared_dir = (args.prepared_dir or resolve_default_prepared_dir(dataset_version)).resolve()
    records = load_prepared_records(prepared_dir, split=args.split)
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

    summary = {
        "sample": {
            "question_id": sample.question_id,
            "frame_token": sample.frame_token,
            "scene_id": sample.scene_id,
            "subtemplate": sample.subtemplate,
            "chapter": sample.chapter,
            "section": sample.section,
            "question": sample.question,
            "answer": sample.answer,
            "camera_views": [
                {
                    "prepared_index": view.prepared_index,
                    "cam_key": view.cam_key,
                    "image_name": view.image_name,
                    "view_direction": view.view_direction,
                    "image_path": str(view.image_path),
                    "camera_intrinsics_shape": list(view.camera_intrinsics.shape),
                    "lidar2camera_shape": list(view.lidar2camera.shape),
                    "lidar2image_shape": list(view.lidar2image.shape),
                    "camera2lidar_shape": list(view.camera2lidar.shape),
                }
                for view in sample.camera_views
            ],
            "point_cloud_shape": list(point_cloud.shape),
            "point_cloud_min_xyz": [float(point_cloud[:, idx].min()) for idx in range(3)],
            "point_cloud_max_xyz": [float(point_cloud[:, idx].max()) for idx in range(3)],
            "structured_targets": sample.structured_targets,
        },
        "qwen3_vl_contract": {
            "processor": qwen_runtime.processor_name,
            "text_hidden_size": qwen_runtime.text_hidden_size,
            "vision_hidden_size": qwen_runtime.vision_hidden_size,
            "qa_json": str(prompt_bundle["qa_json"]),
            "prompt_mode": prompt_bundle["prompt_mode"],
            "prompt_version": prompt_bundle["prompt_version"],
            "subtemplate_patch_style": prompt_bundle["subtemplate_patch_style"],
            "question_input_ids_shape": list(qwen_inputs.question_input_ids.shape),
            "question_attention_mask_shape": list(qwen_inputs.question_attention_mask.shape),
            "image_patch_tokens_shape": list(qwen_inputs.image_patch_tokens.shape),
            "image_patch_mask_shape": list(qwen_inputs.image_patch_mask.shape),
            "image_grid_thw": qwen_inputs.image_grid_thw.tolist(),
            "image_token_counts": list(qwen_inputs.image_token_counts),
            "camera_names": list(qwen_inputs.camera_names),
            "view_directions": [view.view_direction for view in sample.camera_views],
            "system_prompt_length": len(qwen_inputs.system_prompt),
            "user_prompt_length": len(qwen_inputs.user_prompt),
            "full_prompt_length": len(qwen_inputs.full_prompt),
            "full_prompt_preview": qwen_inputs.full_prompt[:600],
        },
        "lion_runtime": {
            "config_path": str(lion_runtime.config_path),
            "checkpoint_path": str(lion_runtime.checkpoint_path),
            "backbone_class": lion_runtime.backbone_class.__name__,
            "detector_class": lion_runtime.detector_class.__name__,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
