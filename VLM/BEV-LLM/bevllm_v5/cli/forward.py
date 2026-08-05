from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bevllm_v5.config.common import DEFAULT_BEST_DIR, DEFAULT_DATASET_VERSION, DEFAULT_FEATURE_VAL_DIR, DEFAULT_PREPARED_DIR, DEFAULT_PREDS_JSONL, DEFAULT_WORK_DIR, INFERENCE_DEFAULTS, resolve_dataset_version_paths
from bevllm_v5.data.dataset import StrictIntersectionV5InferenceCollator, StrictIntersectionV5InferenceDataset
from bevllm_v5.utils.dist import cleanup_distributed, init_distributed, resolve_local_rank_device, shard_sequence, synchronize_distributed
from bevllm_v5.utils.io import dump_json, ensure, load_json
from bevllm_v5.utils.modeling import build_runtime_model, load_checkpoint_config, load_runtime_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed BEV-LLM V5 inference.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=INFERENCE_DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=INFERENCE_DEFAULTS["num_workers"])
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def decode_predictions(tokenizer, outputs, input_ids):
    generated_ids = outputs[:, input_ids.shape[1] :] if outputs.shape[1] > input_ids.shape[1] else outputs
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def resolve_checkpoint_config(args: argparse.Namespace) -> dict:
    payload = load_checkpoint_config(args.checkpoint)
    ensure("model_config" in payload, f"Checkpoint config is missing model_config: {args.checkpoint}")
    return payload


def merge_rank_outputs(final_output: Path, world_size: int) -> None:
    rows = []
    partial_paths = []
    for rank in range(world_size):
        partial = final_output.with_name(f"{final_output.stem}.rank{rank}{final_output.suffix}")
        partial_paths.append(partial)
        ensure(partial.is_file(), f"Missing partial prediction file: {partial}")
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


def rank_output_path(final_output: Path, rank: int) -> Path:
    return final_output.with_name(f"{final_output.stem}.rank{rank}{final_output.suffix}")


def rank_meta_path(final_output: Path, rank: int) -> Path:
    partial = rank_output_path(final_output, rank)
    return partial.with_name(f"{partial.stem}.meta.json")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device_name = resolve_local_rank_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BEV-LLM V5 inference.")
    device = torch.device(device_name)
    output_path = Path(args.output).resolve() if args.output is not None else Path(DEFAULT_PREDS_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path if world_size == 1 else rank_output_path(output_path, rank)
    primary_error: Exception | None = None
    try:
        resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        work_dir = Path(resolved["work_dir"]).resolve()
        if args.output is None:
            output_path = work_dir / "predictions" / Path(DEFAULT_PREDS_JSONL).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_output = output_path if world_size == 1 else rank_output_path(output_path, rank)
        if args.feature_dir is None:
            args.feature_dir = str(work_dir / "features" / "val")
        if args.checkpoint is None:
            args.checkpoint = str(work_dir / "checkpoints" / "best")
        manifest_path = prepared_dir / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )
        feature_manifest_path = work_dir / "features" / "feature_manifest.json"
        if feature_manifest_path.is_file():
            feature_manifest = load_json(feature_manifest_path)
            ensure(
                str(feature_manifest.get("dataset_version")) == str(resolved["dataset_version"]),
                f"Feature version mismatch: expected {resolved['dataset_version']}, found {feature_manifest.get('dataset_version')}",
            )
        checkpoint_payload = resolve_checkpoint_config(args)
        model_config = checkpoint_payload["model_config"]
        train_args = checkpoint_payload.get("train_args") or {}
        checkpoint_dataset_version = train_args.get("dataset_version")
        if checkpoint_dataset_version is not None:
            ensure(
                str(checkpoint_dataset_version) == str(resolved["dataset_version"]),
                f"Checkpoint version mismatch: expected {resolved['dataset_version']}, found {checkpoint_dataset_version}",
            )
        model, tokenizer = build_runtime_model(model_config)
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint_file = checkpoint_path / "checkpoint.pt" if checkpoint_path.is_dir() else checkpoint_path
        load_runtime_checkpoint(model, checkpoint_file)
        model.to(device)
        model.eval()
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = model.generation_config.eos_token_id
        ensure(pad_token_id is not None, "BEV-LLM V5 forward requires a non-null tokenizer.pad_token_id.")
        ensure(eos_token_id is not None, "BEV-LLM V5 forward requires a non-null generation eos_token_id.")

        prepared_dir = Path(args.prepared_dir).resolve()
        data_path = Path(args.data_path).resolve() if args.data_path else prepared_dir / "val.json"
        feature_dir = Path(args.feature_dir).resolve()
        dataset = StrictIntersectionV5InferenceDataset(
            tokenizer=tokenizer,
            data_path=str(data_path),
            feature_dir=str(feature_dir),
            max_length=int(model_config.get("tokenizer_model_max_length", 2048)),
        )
        if args.limit is not None:
            dataset.samples = dataset.samples[: args.limit]
        dataset.samples = shard_sequence(dataset.samples, rank, world_size)

        print(
            f"[forward] rank={rank} local_rank={local_rank} device={device} "
            f"shard={len(dataset.samples)} output={partial_output}",
            flush=True,
        )

        collator = StrictIntersectionV5InferenceCollator(
            tokenizer=tokenizer,
            max_length=int(model_config.get("tokenizer_model_max_length", 2048)),
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

        start_time = time.time()
        with partial_output.open("w", encoding="utf-8") as file:
            iterator = tqdm(data_loader, desc="Forward", leave=True) if rank == 0 else data_loader
            with torch.inference_mode():
                for batch in iterator:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    bevs = batch["bev"].to(device, non_blocking=True)
                    generation_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.temperature > 0,
                    }
                    if args.temperature > 0:
                        generation_kwargs["temperature"] = args.temperature
                        generation_kwargs["top_p"] = args.top_p
                    outputs = model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        bevs=bevs,
                        view=batch["view"],
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        **generation_kwargs,
                    )
                    decoded = decode_predictions(tokenizer, outputs, input_ids)
                    for record, prediction in zip(batch["records"], decoded):
                        row = {
                            "question_id": record.question_id,
                            "scene_id": record.scene_id,
                            "frame_token": record.frame_token,
                            "chapter": record.chapter,
                            "section": record.section,
                            "subtemplate": record.subtemplate,
                            "question": record.question,
                            "reference": record.answer,
                            "prediction": prediction,
                        }
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        dump_json(
            partial_output.with_name(f"{partial_output.stem}.meta.json"),
            {
                "dataset_version": str(resolved["dataset_version"]),
                "rank": rank,
                "world_size": world_size,
                "prepared_dir": str(prepared_dir),
                "checkpoint": str(args.checkpoint),
            },
        )

        if world_size > 1:
            synchronize_distributed("forward output merge")
        if rank == 0 and world_size > 1:
            for merge_rank in range(world_size):
                meta_path = rank_meta_path(output_path, merge_rank)
                ensure(meta_path.is_file(), f"Missing partial prediction metadata file: {meta_path}")
                shard_meta = load_json(meta_path)
                ensure(
                    str(shard_meta.get("dataset_version")) == str(resolved["dataset_version"]),
                    f"Prediction shard version mismatch in {meta_path}: expected {resolved['dataset_version']}, found {shard_meta.get('dataset_version')}",
                )
            merge_rank_outputs(output_path, world_size)
        elif rank == 0:
            print(f"Wrote merged predictions to {output_path}", flush=True)
        elapsed = time.time() - start_time
        if rank == 0 and world_size > 1:
            print(f"Wrote merged predictions to {output_path}", flush=True)
        if rank == 0:
            print(f"[forward] samples={len(dataset.samples)} elapsed={elapsed:.1f}s", flush=True)
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        cleanup_distributed(suppress_errors=primary_error is not None)


if __name__ == "__main__":
    main()
