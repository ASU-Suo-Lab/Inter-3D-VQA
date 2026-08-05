from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geovlm_intersection.backbones import (
    build_lion_model_runtime,
    build_qwen3_vl_model_runtime,
    embed_qwen3_vl_question_ids,
    extract_lion_tokens,
    extract_qwen3_vl_vision_features,
    load_qwen3_vl_runtime,
    prepare_qwen3_vl_inputs,
)
from geovlm_intersection.config.common import DATASET_VERSION_DEFAULTS, DEFAULT_INFO_PKL, DEFAULT_LION_QUALITY, validate_dataset_version
from geovlm_intersection.data import SUBTEMPLATE_TO_INDEX, build_info_index, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.data.v5_io import resolve_default_prepared_dir
from geovlm_intersection.models.architecture import GeoVLMConfig, GeoVLMModel
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-sample GeoVLM forward smoke using Qwen3-VL vision features and LION geometry tokens."
    )
    parser.add_argument("--dataset-version", default="v5", choices=["v5"], help="GeoVLM currently supports only v5.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared directory to use.")
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL, help="SunLakes infos pickle.")
    parser.add_argument("--split", default="val_eval", help="Prepared split JSON to inspect.")
    parser.add_argument("--sample-index", type=int, default=0, help="Prepared sample index.")
    parser.add_argument("--lion-quality", default=DEFAULT_LION_QUALITY, choices=["low", "mid", "high"], help="LION checkpoint tier.")
    parser.add_argument("--max-objects", type=int, default=128, help="Maximum number of LION detections to keep.")
    parser.add_argument("--qwen-device", default="cuda:0", help="Device for Qwen3-VL vision feature extraction.")
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
    qwen_question = embed_qwen3_vl_question_ids(
        qwen_runtime,
        qwen_inputs.question_input_ids,
        qwen_inputs.question_attention_mask,
    )
    qwen_model_runtime = build_qwen3_vl_model_runtime(base_runtime=qwen_runtime, device=args.qwen_device)
    qwen_vision = extract_qwen3_vl_vision_features(qwen_model_runtime, qwen_inputs)
    lion_runtime = build_lion_model_runtime(args.lion_quality)
    lion_outputs = extract_lion_tokens(sample, lion_runtime, max_objects=args.max_objects)

    model = GeoVLMModel(
        GeoVLMConfig(
            image_token_dim=qwen_vision.image_tokens.shape[-1],
            bev_token_dim=lion_outputs.bev_tokens.shape[-1],
            object_token_dim=lion_outputs.object_tokens.shape[-1],
            question_token_dim=qwen_question.embeddings.shape[-1],
        )
    )
    encoded_inputs = model.encode_inputs(
        image_tokens=qwen_vision.image_tokens.float(),
        bev_tokens=lion_outputs.bev_tokens.float().cpu(),
        object_tokens=lion_outputs.object_tokens.float().cpu(),
        raw_object_tokens=lion_outputs.raw_object_tokens.float().cpu(),
        question_tokens=qwen_question.embeddings.float().cpu(),
        subtemplate_ids=qwen_question.embeddings.new_tensor(
            [SUBTEMPLATE_TO_INDEX[sample.subtemplate]],
            dtype=torch.long,
        ),
    )
    outputs = model.forward_encoded(encoded_inputs)

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
            "system_prompt_length": len(qwen_inputs.system_prompt),
            "user_prompt_length": len(qwen_inputs.user_prompt),
            "full_prompt_length": len(qwen_inputs.full_prompt),
            "full_prompt_preview": qwen_inputs.full_prompt[:600],
        },
        "contracts": {
            "pixel_values_shape": list(qwen_inputs.pixel_values.shape),
            "image_patch_tokens_shape": list(qwen_inputs.image_patch_tokens.shape),
            "qwen_vision_tokens_shape": list(qwen_vision.image_tokens.shape),
            "qwen_vision_mask_shape": list(qwen_vision.image_mask.shape),
            "qwen_vision_feature_shapes": [list(shape) for shape in qwen_vision.per_image_feature_shapes],
            "bev_tokens_shape": list(lion_outputs.bev_tokens.shape),
            "object_tokens_shape": list(lion_outputs.object_tokens.shape),
            "raw_object_tokens_shape": list(lion_outputs.raw_object_tokens.shape),
            "question_tokens_shape": list(qwen_question.embeddings.shape),
            "lion_bev_feature_shape": list(lion_outputs.bev_feature_shape),
            "lion_bev_grid_size": list(lion_outputs.bev_grid_size),
            "detected_objects": int(lion_outputs.pred_boxes.shape[0]),
        },
        "compressed_contracts": {
            "image_tokens_shape": list(encoded_inputs.image_tokens.shape),
            "bev_tokens_shape": list(encoded_inputs.bev_tokens.shape),
            "object_tokens_shape": list(encoded_inputs.object_tokens.shape),
            "raw_object_tokens_shape": list(encoded_inputs.raw_object_tokens.shape),
            "question_tokens_shape": list(encoded_inputs.question_tokens.shape),
            "image_token_budget_per_camera": model.config.image_token_budget_per_camera,
            "bev_token_budget": model.config.bev_token_budget,
            "object_token_budget": model.config.object_token_budget,
            "question_token_budget": model.config.question_token_budget,
        },
        "geo_outputs": {key: list(value.shape) for key, value in outputs.items()},
        "detections_preview": {
            "pred_boxes_shape": list(lion_outputs.pred_boxes.shape),
            "pred_scores_top5": [float(x) for x in lion_outputs.pred_scores[:5].detach().cpu().tolist()],
            "pred_labels_top5": [int(x) for x in lion_outputs.pred_labels[:5].detach().cpu().tolist()],
            "query_feature_dim": int(lion_outputs.query_feature_dim),
            "object_local_feature_dim": int(lion_outputs.object_local_feature_dim),
        },
        "notes": {
            "question_bridge": "uses real Qwen3-VL token embeddings from local safetensors over the assembled system+user+subtemplate prompt",
            "image_branch": "uses real Qwen3-VL get_image_features() outputs over processor pixel_values; decoder path is not wired yet",
            "lion_device": lion_runtime.device,
            "qwen_device": qwen_model_runtime.device,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
