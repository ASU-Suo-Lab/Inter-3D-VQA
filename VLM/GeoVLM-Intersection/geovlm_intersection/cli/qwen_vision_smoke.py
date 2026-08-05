from __future__ import annotations

import argparse
import json
from pathlib import Path

from geovlm_intersection.backbones import (
    build_qwen3_vl_model_runtime,
    extract_qwen3_vl_vision_features,
    load_qwen3_vl_runtime,
    prepare_qwen3_vl_inputs,
)
from geovlm_intersection.config.common import DATASET_VERSION_DEFAULTS, DEFAULT_INFO_PKL, validate_dataset_version
from geovlm_intersection.data import build_info_index, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.data.v5_io import resolve_default_prepared_dir
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-sample Qwen3-VL vision-feature smoke using the assembled GeoVLM prompt."
    )
    parser.add_argument("--dataset-version", default="v5", choices=["v5"], help="GeoVLM currently supports only v5.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared directory to use.")
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL, help="SunLakes infos pickle.")
    parser.add_argument("--split", default="val_eval", help="Prepared split JSON to inspect.")
    parser.add_argument("--sample-index", type=int, default=0, help="Prepared sample index.")
    parser.add_argument("--device", default="cuda:0", help="Device for Qwen3-VL vision extraction.")
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
    qwen_runtime = load_qwen3_vl_runtime()
    qwen_inputs = prepare_qwen3_vl_inputs(qwen_runtime, sample, prompt_bundle=prompt_bundle)
    qwen_model_runtime = build_qwen3_vl_model_runtime(base_runtime=qwen_runtime, device=args.device)
    vision_features = extract_qwen3_vl_vision_features(qwen_model_runtime, qwen_inputs)

    summary = {
        "sample": {
            "question_id": sample.question_id,
            "frame_token": sample.frame_token,
            "subtemplate": sample.subtemplate,
            "question": sample.question,
        },
        "prompting": {
            "qa_json": str(prompt_bundle["qa_json"]),
            "prompt_mode": prompt_bundle["prompt_mode"],
            "prompt_version": prompt_bundle["prompt_version"],
            "subtemplate_patch_style": prompt_bundle["subtemplate_patch_style"],
            "full_prompt_length": len(qwen_inputs.full_prompt),
            "full_prompt_preview": qwen_inputs.full_prompt[:600],
        },
        "processor_contract": {
            "pixel_values_shape": list(qwen_inputs.pixel_values.shape),
            "image_patch_tokens_shape": list(qwen_inputs.image_patch_tokens.shape),
            "image_grid_thw": qwen_inputs.image_grid_thw.tolist(),
            "image_token_counts": list(qwen_inputs.image_token_counts),
        },
        "vision_features": {
            "device": qwen_model_runtime.device,
            "hidden_size": vision_features.hidden_size,
            "image_tokens_shape": list(vision_features.image_tokens.shape),
            "image_mask_shape": list(vision_features.image_mask.shape),
            "image_token_counts": list(vision_features.image_token_counts),
            "per_image_feature_shapes": [list(shape) for shape in vision_features.per_image_feature_shapes],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
