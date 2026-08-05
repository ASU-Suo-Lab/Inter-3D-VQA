from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from opendrivevla_v5.config.eval import MAX_NEW_TOKENS
from opendrivevla_v5.config.common import (
    DEFAULT_BEST_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_FEATURE_VAL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_DIR,
    DEFAULT_PREDS_JSONL,
    DEFAULT_PREPARED_DIR,
    DEFAULT_WORK_DIR,
    resolve_dataset_version_paths,
)
from opendrivevla_v5.data.collator import IntersectionV5Collator
from opendrivevla_v5.data.dataset import IntersectionV5QADataset
from opendrivevla_v5.utils.dist import cleanup_distributed, init_distributed, resolve_local_rank_device, synchronize_distributed
from opendrivevla_v5.utils.io import ensure, load_json
from opendrivevla_v5.utils.modeling import load_inference_model
from opendrivevla_v5.utils.tensors import change_tensor_to_float16, change_tensor_to_float32, move_data_to_device


ANSWER_BLOCK_RE = re.compile(r"<answer_start>(.*?)<answer_end>", re.DOTALL | re.IGNORECASE)
TRAJ_BLOCK_RE = re.compile(r"<traj_start>.*?<traj_end>", re.DOTALL | re.IGNORECASE)
WAYPOINT_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)")
TAG_RE = re.compile(r"<[^>]+>")


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenDriveVLA on the strict Intersection V5 QA dataset.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--infos-pkl", default=None)
    parser.add_argument("--uniad-pth-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def is_trajectory_question(batch) -> bool:
    return batch.get("subtemplate") == "r_object_future_trajectory"


def extract_trajectory_text(raw_text: str, max_points: int = 4) -> str:
    text = (raw_text or "").strip()
    match = TRAJ_BLOCK_RE.search(text)
    if not match:
        return ""
    points = WAYPOINT_RE.findall(match.group(0))
    if not points:
        return match.group(0).strip()
    points = points[:max_points]
    normalized = ",".join(f"({float(x):.2f},{float(y):.2f})" for x, y in points)
    return f"<traj_start>[{normalized}]<traj_end>"


def extract_answer_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    match = ANSWER_BLOCK_RE.search(text)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def generate_prediction(model, tokenizer, batch, args):
    input_ids = batch["input_ids"]
    generation_kwargs = {
        "do_sample": False,
        "temperature": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "uniad_pth": batch["uniad_pth"],
    }
    with torch.inference_mode():
        outputs = model.generate(input_ids, **generation_kwargs)

    generated_ids = outputs[:, input_ids.shape[1] :] if outputs.shape[1] > input_ids.shape[1] else outputs
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    prediction = extract_trajectory_text(raw_output) if is_trajectory_question(batch) else extract_answer_text(raw_output)
    return raw_output, prediction


def merge_rank_outputs(final_output: Path, world_size: int) -> None:
    rows: List[dict] = []
    partial_paths = []
    for rank in range(world_size):
        partial = final_output.with_name(f"{final_output.stem}.rank{rank}{final_output.suffix}")
        partial_paths.append(partial)
        with partial.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: row["question_id"])
    with final_output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    for partial in partial_paths:
        partial.unlink(missing_ok=True)


def resolve_model_path(args) -> Path:
    if args.model_path:
        return Path(args.model_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    best_dir = work_dir / DEFAULT_CHECKPOINT_DIR.name / DEFAULT_BEST_DIR.name
    if best_dir.is_dir():
        return best_dir
    last_dir = work_dir / DEFAULT_CHECKPOINT_DIR.name / "last.pth"
    if last_dir.is_dir():
        return last_dir
    return Path(DEFAULT_MODEL_PATH).resolve()


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device_name = resolve_local_rank_device(args.device)
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    if args.uniad_pth_dir is None:
        args.uniad_pth_dir = str(work_dir / DEFAULT_FEATURE_VAL_DIR.name)
    if args.output is None:
        args.output = str(work_dir / DEFAULT_PREDICTION_DIR.name / DEFAULT_PREDS_JSONL.name)
    manifest_path = prepared_dir / "split_manifest.json"
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path if world_size == 1 else output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")
    primary_error: Exception | None = None
    current_question_id: str | None = None
    current_frame_token: str | None = None
    current_stage = "setup"
    try:
        model_path = resolve_model_path(args)
        current_stage = "load_model"
        tokenizer, model = load_inference_model(str(model_path), args.device, args.attn_implementation)
        prepared_dir = Path(args.prepared_dir).resolve()
        current_stage = "load_dataset"
        dataset = IntersectionV5QADataset(
            tokenizer=tokenizer,
            qa_json=str(Path(args.qa_json).resolve() if args.qa_json else prepared_dir / "qa_val.json"),
            infos_pkl=str(Path(args.infos_pkl).resolve() if args.infos_pkl else prepared_dir / "infos_val.pkl"),
            device=torch.device(device_name),
            llava_test_mode=True,
            include_visual_tokens=True,
            uniad_pth_dir=str(Path(args.uniad_pth_dir).resolve()),
            max_samples=args.limit,
        )
        if world_size > 1:
            dataset.samples = dataset.samples[rank::world_size]

        print(
            f"[forward] rank={rank} local_rank={local_rank} device={device_name} "
            f"shard={len(dataset)} output={partial_output}",
            flush=True,
        )

        collator = IntersectionV5Collator(tokenizer=tokenizer, llava_test_mode=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

        start_time = time.time()
        current_stage = "generate"
        with partial_output.open("w", encoding="utf-8") as file:
            for batch in tqdm(dataloader, ncols=80, disable=rank != 0):
                current_question_id = batch["question_id"]
                current_frame_token = batch["frame_token"]
                batch = move_data_to_device(batch, device_name)
                model_dtype = next(model.parameters()).dtype
                if model_dtype == torch.float16:
                    batch["uniad_pth"] = change_tensor_to_float16(batch["uniad_pth"])
                elif model_dtype == torch.float32:
                    batch["uniad_pth"] = change_tensor_to_float32(batch["uniad_pth"])

                raw_output, prediction = generate_prediction(model, tokenizer, batch, args)
                row = {
                    "question_id": batch["question_id"],
                    "frame_token": batch["frame_token"],
                    "question": batch["question"],
                    "reference": batch["reference"],
                    "category": batch.get("category"),
                    "subtemplate": batch.get("subtemplate"),
                    "prediction": prediction,
                    "raw_output": raw_output,
                }
                current_stage = "write_output"
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                current_stage = "generate"

        if world_size > 1:
            current_stage = "sync_before_merge"
            synchronize_distributed("forward output merge")
        if rank == 0 and world_size > 1:
            current_stage = "merge_outputs"
            merge_rank_outputs(output_path, world_size)
        elapsed = time.time() - start_time
        if rank == 0:
            print(f"[forward] model={model_path} samples={len(dataset)} elapsed={elapsed:.1f}s", flush=True)
    except Exception as exc:
        primary_error = exc
        context = []
        if current_question_id is not None:
            context.append(f"question_id={current_question_id}")
        if current_frame_token is not None:
            context.append(f"frame_token={current_frame_token}")
        context_text = " " + " ".join(context) if context else ""
        print(
            f"[forward] rank={rank} local_rank={local_rank} stage={current_stage}{context_text} "
            f"failed with {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        cleanup_distributed(suppress_errors=primary_error is not None)


if __name__ == "__main__":
    main()
