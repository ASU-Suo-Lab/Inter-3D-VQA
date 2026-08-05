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
from geovlm_intersection.data import (
    SUBTEMPLATE_TO_INDEX,
    build_info_index,
    build_structured_supervision,
    load_prepared_records,
    resolve_prepared_sample,
    supervision_to_device_dict,
)
from geovlm_intersection.data.v5_io import resolve_default_prepared_dir
from geovlm_intersection.models import GeoVLMConfig, GeoVLMModel, compute_structured_losses
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single real-sample GeoVLM train-step smoke with Qwen3-VL + LION + structured losses."
    )
    parser.add_argument("--dataset-version", default="v5", choices=["v5"], help="GeoVLM currently supports only v5.")
    parser.add_argument("--prepared-dir", type=Path, default=None, help="Prepared directory to use.")
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL, help="SunLakes infos pickle.")
    parser.add_argument("--split", default="train", help="Prepared split JSON to inspect.")
    parser.add_argument("--sample-index", type=int, default=0, help="Prepared sample index.")
    parser.add_argument("--lion-quality", default=DEFAULT_LION_QUALITY, choices=["low", "mid", "high"], help="LION checkpoint tier.")
    parser.add_argument("--max-objects", type=int, default=128, help="Maximum number of LION detections to keep.")
    parser.add_argument("--qwen-device", default="cuda:0", help="Device for Qwen3-VL vision feature extraction.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the smoke optimizer.")
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
    supervision = build_structured_supervision(sample)
    if supervision is None:
        raise ValueError(f"Sample subtemplate is not yet supported by structured supervision: {sample.subtemplate}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    encoded_inputs = model.encode_inputs(
        image_tokens=qwen_vision.image_tokens.float(),
        bev_tokens=lion_outputs.bev_tokens.float().cpu(),
        object_tokens=lion_outputs.object_tokens.float().cpu(),
        raw_object_tokens=lion_outputs.raw_object_tokens.float().cpu(),
        question_tokens=qwen_question.embeddings.float().cpu(),
        subtemplate_ids=torch.tensor([SUBTEMPLATE_TO_INDEX[sample.subtemplate]], dtype=torch.long),
    )
    outputs = model.forward_encoded(encoded_inputs)
    target_tensors = supervision_to_device_dict(supervision, device=outputs["scene_pooled"].device)
    loss_output = compute_structured_losses(outputs, target_tensors)

    optimizer.zero_grad(set_to_none=True)
    loss_output.total_loss.backward()
    optimizer.step()

    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm_sq += float(parameter.grad.detach().pow(2).sum().item())

    summary = {
        "sample": {
            "question_id": sample.question_id,
            "subtemplate": sample.subtemplate,
            "question": sample.question,
        },
        "prompting": {
            "prompt_mode": prompt_bundle["prompt_mode"],
            "prompt_version": prompt_bundle["prompt_version"],
            "full_prompt_length": len(qwen_inputs.full_prompt),
        },
        "contracts": {
            "qwen_vision_tokens_shape": list(qwen_vision.image_tokens.shape),
            "lion_bev_tokens_shape": list(lion_outputs.bev_tokens.shape),
            "lion_object_tokens_shape": list(lion_outputs.object_tokens.shape),
            "lion_raw_object_tokens_shape": list(lion_outputs.raw_object_tokens.shape),
            "question_tokens_shape": list(qwen_question.embeddings.shape),
            "compressed_image_tokens_shape": list(encoded_inputs.image_tokens.shape),
            "compressed_bev_tokens_shape": list(encoded_inputs.bev_tokens.shape),
            "compressed_object_tokens_shape": list(encoded_inputs.object_tokens.shape),
            "compressed_raw_object_tokens_shape": list(encoded_inputs.raw_object_tokens.shape),
            "compressed_question_tokens_shape": list(encoded_inputs.question_tokens.shape),
        },
        "supervision": supervision.to_dict(),
        "loss": {
            "total": float(loss_output.total_loss.detach().cpu().item()),
            "components": {key: float(value.detach().cpu().item()) for key, value in loss_output.components.items()},
            "grad_norm_l2": grad_norm_sq**0.5,
        },
        "notes": {
            "trainable_path": "fusion core + structured heads only; Qwen3-VL and LION are feature extractors in this smoke path",
            "lion_device": lion_runtime.device,
            "qwen_device": qwen_model_runtime.device,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
