from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
from uuid import uuid4

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from nuscenesqa_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from nuscenesqa_v5.data.dataset import BOS_ID, EOS_ID, IntersectionNuScenesQAEvalDataset, collate_eval, decode_answer_ids
from nuscenesqa_v5.data.templates import diagnose_prediction, normalize_prediction
from nuscenesqa_v5.utils.dist import barrier, destroy_process_group, init_distributed, is_main_process
from nuscenesqa_v5.utils.io import dump_json, dump_jsonl, ensure, load_json
from nuscenesqa_v5.utils.modeling import load_checkpoint_model, load_feature_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed forward for NuScenes-QA.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size_per_gpu"])
    parser.add_argument("--max-new-tokens", type=int, default=TRAINING_DEFAULTS["max_answer_chars"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def create_run_id(rank: int) -> str:
    payload = [""]
    if rank == 0:
        payload[0] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def merge_predictions(prediction_dir: Path, world_size: int, expected_run_id: str, expected_dataset_version: str) -> None:
    merged_rows = []
    for rank in range(world_size):
        part_path = prediction_dir / f"predictions_rank{rank}.jsonl"
        meta_path = prediction_dir / f"predictions_rank{rank}.meta.json"
        ensure(part_path.is_file(), f"Missing prediction shard: {part_path}")
        ensure(meta_path.is_file(), f"Missing prediction shard metadata: {meta_path}")
        meta = load_json(meta_path)
        ensure(meta.get("run_id") == expected_run_id, f"Prediction shard {meta_path} belongs to stale run_id={meta.get('run_id')}, expected {expected_run_id}. Re-run full forward.")
        ensure(int(meta.get("world_size", -1)) == world_size, f"Prediction shard {meta_path} was produced with world_size={meta.get('world_size')}, expected {world_size}. Re-run full forward.")
        ensure(meta.get("dataset_version") == expected_dataset_version, f"Prediction shard {meta_path} belongs to dataset_version={meta.get('dataset_version')}, expected {expected_dataset_version}. Re-run full forward.")
        with part_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    import json

                    merged_rows.append(json.loads(line))
    merged_rows.sort(key=lambda row: row["question_id"])
    dump_jsonl(prediction_dir / "merged_predictions.jsonl", merged_rows)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    ensure(torch.cuda.is_available(), "CUDA is required for forward.")
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    run_id = create_run_id(rank)

    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else worktree["best_checkpoint"]
    prediction_dir = worktree["predictions"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(split_manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared split_manifest dataset_version={split_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")

    feature_manifest = load_feature_manifest(worktree["feature_manifest"])
    model, payload = load_checkpoint_model(checkpoint, device)
    model.eval()
    checkpoint_config = payload.get("model_config") or {}
    ensure(
        int(checkpoint_config.get("object_feature_dim", -1)) == int(feature_manifest["object_feature_dim"]),
        "Checkpoint object_feature_dim does not match extracted feature manifest. Re-run extract and retrain before forward.",
    )
    ensure(
        int(checkpoint_config.get("bbox_feature_dim", -1)) == int(feature_manifest["bbox_feature_dim"]),
        "Checkpoint bbox_feature_dim does not match extracted feature manifest. Re-run extract and retrain before forward.",
    )
    train_args = payload.get("train_args") or {}
    ensure(
        str(train_args.get("dataset_version", resolved["dataset_version"])) == str(resolved["dataset_version"]),
        f"Checkpoint dataset_version={train_args.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )
    ensure(
        str(feature_manifest.get("dataset_version", resolved["dataset_version"])) == str(resolved["dataset_version"]),
        f"Feature manifest dataset_version={feature_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )
    max_question_chars = int(train_args.get("max_question_chars", TRAINING_DEFAULTS["max_question_chars"]))

    dataset = IntersectionNuScenesQAEvalDataset(
        records_path=prepared_dir / "val_eval.json",
        feature_root=worktree["features"] / "val",
        max_question_chars=max_question_chars,
    )
    shard_indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, shard_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_eval,
    )

    rows = []
    template_stats: dict[str, dict[str, object]] = defaultdict(lambda: {"count": 0, "normalized_count": 0, "reasons": Counter()})
    template_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for batch in loader:
        object_features = batch["object_features"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        question_ids = batch["question_ids"].to(device, non_blocking=True)
        decoder_prefix_ids = batch["decoder_prefix_ids"].to(device, non_blocking=True)
        decoder_prefix_mask = batch["decoder_prefix_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                object_features=object_features,
                bbox_features=bbox_features,
                question_ids=question_ids,
                decoder_prefix_ids=decoder_prefix_ids,
                decoder_prefix_mask=decoder_prefix_mask,
                max_new_tokens=args.max_new_tokens,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
            )
        for metadata, answer_ids in zip(batch["metadata"], generated, strict=True):
            prediction_raw = decode_answer_ids(answer_ids.cpu())
            prediction, normalization_reasons = normalize_prediction(
                metadata["subtemplate"],
                prediction_raw,
                metadata.get("decoder_prefix"),
                dataset_version=str(resolved["dataset_version"]),
            )
            diagnosis_reasons = diagnose_prediction(
                metadata["subtemplate"],
                prediction_raw,
                prediction,
                dataset_version=str(resolved["dataset_version"]),
            )
            reasons = [*normalization_reasons, *diagnosis_reasons]
            stats = template_stats[metadata["subtemplate"]]
            stats["count"] = int(stats["count"]) + 1
            if reasons:
                stats["normalized_count"] = int(stats["normalized_count"]) + 1
                counter = stats["reasons"]
                ensure(isinstance(counter, Counter), f"Unexpected counter type for {metadata['subtemplate']}")
                for reason in reasons:
                    counter[reason] += 1
                if len(template_examples[metadata["subtemplate"]]) < 3:
                    template_examples[metadata["subtemplate"]].append(
                        {
                            "question_id": metadata["question_id"],
                            "raw_prediction": prediction_raw,
                            "normalized_prediction": prediction,
                            "reasons": reasons,
                        }
                    )
            rows.append(
                {
                    "question_id": metadata["question_id"],
                    "scene_id": metadata["scene_id"],
                    "frame_token": metadata["frame_token"],
                    "chapter": metadata["chapter"],
                    "section": metadata["section"],
                    "subtemplate": metadata["subtemplate"],
                    "question": metadata["question"],
                    "reference_answer": metadata["reference_answer"],
                    "prediction": prediction,
                }
            )

    dump_jsonl(prediction_dir / f"predictions_rank{rank}.jsonl", rows)
    dump_json(
        prediction_dir / f"predictions_rank{rank}.meta.json",
        {
            "run_id": run_id,
            "rank": rank,
            "world_size": world_size,
            "dataset_version": str(resolved["dataset_version"]),
            "template_stats": {
                subtemplate: {
                    "count": int(payload["count"]),
                    "normalized_count": int(payload["normalized_count"]),
                    "reasons": dict(payload["reasons"]),
                }
                for subtemplate, payload in sorted(template_stats.items())
            },
            "template_examples": dict(sorted(template_examples.items())),
        },
    )
    barrier()
    if is_main_process():
        dump_json(
            prediction_dir / "forward_run.json",
            {"run_id": run_id, "world_size": world_size, "dataset_version": str(resolved["dataset_version"])},
        )
        merge_predictions(prediction_dir, world_size, run_id, str(resolved["dataset_version"]))
    destroy_process_group()


if __name__ == "__main__":
    main()
