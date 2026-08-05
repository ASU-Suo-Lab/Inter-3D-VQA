from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from llava.model.builder import load_senna_pretrained_model
from senna_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_PREPARED_DIR, DEFAULT_VISION_TOWER, DEFAULT_WORK_DIR, INFERENCE_DEFAULTS, resolve_dataset_version_paths, worktree_paths
from senna_v5.utils.dist import shard_sequence
from senna_v5.utils.inference import generate_multi_image_answer
from senna_v5.utils.io import dump_json, dump_jsonl, ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Senna V5 forward on prepared val data.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
    parser.add_argument("--num-beams", type=int, default=INFERENCE_DEFAULTS["num_beams"])
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    return parser.parse_args()


def init_dist() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_dist()
    device = f"cuda:{local_rank}"
    ensure(torch.cuda.is_available(), "CUDA is required for Senna V5 forward.")

    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    worktree = worktree_paths(work_dir)
    data_path = Path(args.data_path).resolve() if args.data_path is not None else prepared_dir / "val_eval.json"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else worktree["predictions"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve() if args.model_path is not None else worktree["best_checkpoint"]
    ensure(model_path.is_dir(), f"Model path not found: {model_path}")
    manifest_path = prepared_dir / "split_manifest.json"
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )

    rows = load_json(data_path)
    ensure(isinstance(rows, list) and rows, f"Evaluation data is empty: {data_path}")
    shard = shard_sequence(rows, rank, world_size)
    print(f"[forward] rank={rank} local_rank={local_rank} device={device} shard={len(shard)}/{len(rows)} output_dir={output_dir}")

    tokenizer, model, image_processor, _ = load_senna_pretrained_model(
        str(model_path),
        None,
        model_name="llava",
        device=device,
        device_map=device,
        vision_tower=str(Path(args.vision_tower).resolve()),
    )
    model.to(device)
    model.eval()

    rank_rows = []
    for sample in tqdm(shard, disable=rank != 0):
        prediction = generate_multi_image_answer(
            prompt=str(sample["prompt"]),
            image_paths=sample["images"],
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_mode=args.conv_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        rank_rows.append(
            {
                "question_id": str(sample["question_id"]),
                "scene_id": str(sample["scene_id"]),
                "frame_token": str(sample["frame_token"]),
                "chapter": str(sample["chapter"]),
                "section": str(sample["section"]),
                "subtemplate": str(sample["subtemplate"]),
                "question": str(sample["question"]),
                "reference_answer": str(sample["answer"]),
                "prediction": prediction,
            }
        )

    rank_path = output_dir / f"predictions_rank{rank}.jsonl"
    dump_jsonl(rank_path, rank_rows)
    dump_json(
        output_dir / f"predictions_rank{rank}.meta.json",
        {
            "dataset_version": str(resolved["dataset_version"]),
            "rank": rank,
            "world_size": world_size,
            "data_path": str(data_path),
            "model_path": str(model_path),
        },
    )
    barrier()

    if rank == 0:
        merged_rows = []
        for shard_rank in range(world_size):
            shard_path = output_dir / f"predictions_rank{shard_rank}.jsonl"
            meta_path = output_dir / f"predictions_rank{shard_rank}.meta.json"
            ensure(shard_path.is_file(), f"Missing shard prediction file: {shard_path}")
            ensure(meta_path.is_file(), f"Missing shard metadata file: {meta_path}")
            shard_meta = load_json(meta_path)
            ensure(
                str(shard_meta.get("dataset_version")) == str(resolved["dataset_version"]),
                f"Prediction shard version mismatch in {meta_path}: expected {resolved['dataset_version']}, found {shard_meta.get('dataset_version')}",
            )
            with shard_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        merged_rows.append(json.loads(line))
        merged_path = output_dir / "merged_predictions.jsonl"
        dump_jsonl(merged_path, merged_rows)
        print(f"Wrote merged predictions to {merged_path}")

    barrier()
    destroy_dist()


if __name__ == "__main__":
    main()
