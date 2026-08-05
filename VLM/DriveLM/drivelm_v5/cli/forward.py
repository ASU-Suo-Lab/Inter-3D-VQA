from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drivelm_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_LLAMAA_DIR, DEFAULT_PREPARED_DIR, DEFAULT_WORK_DIR, INFERENCE_DEFAULTS, resolve_dataset_version_paths, worktree_paths
from drivelm_v5.data.dataset import DriveLMV5InferenceCollator, DriveLMV5InferenceDataset
from drivelm_v5.utils.dist import barrier, cleanup_distributed, init_distributed
from drivelm_v5.utils.imports import add_llama_adapter_to_path
from drivelm_v5.utils.io import ensure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed DriveLM V5 inference.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--llama-dir", default=os.environ.get("DRIVELM_LLAMA_DIR", str(DEFAULT_LLAMAA_DIR)))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
    return parser.parse_args()


def ensure_llama_dir(path: Path) -> None:
    ensure(path.is_dir(), f"LLaMA base directory not found: {path}")
    ensure((path / "tokenizer.model").is_file(), f"Missing tokenizer.model under {path}")
    ensure((path / "7B").is_dir(), f"Missing 7B directory under {path}")
    ensure((path / "7B" / "params.json").is_file(), f"Missing params.json under {path / '7B'}")
    ensure(list((path / "7B").glob("*.pth")), f"No LLaMA checkpoint shards found under {path / '7B'}")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    add_llama_adapter_to_path()

    from llama import format_prompt
    from llama.llama_adapter import LLaMA_adapter

    ensure(torch.cuda.is_available(), "CUDA is required for DriveLM V5 inference.")
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    llama_dir = Path(args.llama_dir).resolve()
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint is not None else worktree_paths(work_dir)["best_checkpoint"]
    data_path = Path(args.data_path).resolve() if args.data_path is not None else prepared_dir / "val_eval.json"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else worktree_paths(work_dir)["predictions"]
    manifest_path = prepared_dir / "split_manifest.json"
    ensure_llama_dir(llama_dir)
    ensure(checkpoint.is_file(), f"Checkpoint not found: {checkpoint}")
    ensure(data_path.is_file(), f"Inference data not found: {data_path}")
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )

    try:
        model = LLaMA_adapter(str(llama_dir / "7B"), str(llama_dir / "tokenizer.model"), max_batch_size=max(args.batch_size, 32))
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        missing = model.load_state_dict(state_dict, strict=False)
        ensure(not missing.unexpected_keys, f"Unexpected keys while loading {checkpoint}: {missing.unexpected_keys}")
        model.to(device)
        model.eval()

        all_samples = json.loads(data_path.read_text(encoding="utf-8"))
        ensure(isinstance(all_samples, list) and all_samples, f"{data_path} does not contain a non-empty list.")
        shard = all_samples[rank::world_size]
        print(f"[forward] rank={rank} local_rank={local_rank} device={device} shard={len(shard)}/{len(all_samples)} output_dir={output_dir}")

        rank_data_path = output_dir / f"_rank{rank}_shard.json"
        rank_data_path.write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
        dataset = DriveLMV5InferenceDataset(rank_data_path, prompt_formatter=format_prompt)
        collator = DriveLMV5InferenceCollator()
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

        rank_output = output_dir / f"predictions_rank{rank}.jsonl"
        rank_meta = output_dir / f"predictions_rank{rank}.meta.json"
        with rank_output.open("w", encoding="utf-8") as output_file:
            iterator = tqdm(data_loader, desc="Forward", leave=True) if rank == 0 else data_loader
            with torch.inference_mode():
                for batch in iterator:
                    images = batch["images"].to(device, non_blocking=True)
                    prompts = batch["prompts"]
                    predictions = model.generate(
                        images,
                        prompts,
                        max_gen_len=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    for record, prediction in zip(batch["records"], predictions):
                        row = {
                            "question_id": record.question_id,
                            "scene_id": record.scene_id,
                            "frame_token": record.frame_token,
                            "chapter": record.chapter,
                            "section": record.section,
                            "subtemplate": record.subtemplate,
                            "question": record.question,
                            "reference_answer": record.answer,
                            "prediction": prediction.strip(),
                        }
                        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        rank_meta.write_text(
            json.dumps(
                {
                    "dataset_version": str(resolved["dataset_version"]),
                    "rank": rank,
                    "world_size": world_size,
                    "data_path": str(data_path),
                    "checkpoint": str(checkpoint),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        barrier()
        if rank == 0:
            merged_output = output_dir / "merged_predictions.jsonl"
            merged_output.parent.mkdir(parents=True, exist_ok=True)
            with merged_output.open("w", encoding="utf-8") as merged_file:
                for worker_rank in range(world_size):
                    worker_file = output_dir / f"predictions_rank{worker_rank}.jsonl"
                    worker_meta = output_dir / f"predictions_rank{worker_rank}.meta.json"
                    ensure(worker_file.is_file(), f"Missing rank output file: {worker_file}")
                    ensure(worker_meta.is_file(), f"Missing rank metadata file: {worker_meta}")
                    shard_meta = json.loads(worker_meta.read_text(encoding="utf-8"))
                    ensure(
                        str(shard_meta.get("dataset_version")) == str(resolved["dataset_version"]),
                        f"Prediction shard version mismatch in {worker_meta}: expected {resolved['dataset_version']}, found {shard_meta.get('dataset_version')}",
                    )
                    merged_file.write(worker_file.read_text(encoding="utf-8"))
            print(f"Wrote merged predictions to {merged_output}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
