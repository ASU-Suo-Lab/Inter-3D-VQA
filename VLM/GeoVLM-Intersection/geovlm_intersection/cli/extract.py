from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from geovlm_intersection.backbones import (
    build_lion_model_runtime,
    build_qwen3_vl_model_runtime,
    extract_lion_tokens,
    extract_qwen3_vl_vision_features,
    load_qwen3_vl_runtime,
    prepare_qwen3_vl_inputs,
)
from geovlm_intersection.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_LION_QUALITY,
    FEATURE_LAYOUT_VERSION,
    FRAME_ONLY_FEATURE_STORAGE,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from geovlm_intersection.data import (
    build_frame_storage_tensors,
    build_info_index,
    load_prepared_records,
    resolve_prepared_sample,
    save_feature_payload,
)
from geovlm_intersection.prompting import load_prompt_bundle, validate_subtemplates
from geovlm_intersection.utils import dump_json, ensure, load_json


ALL_SPLITS = ("train", "val", "val_eval")
PHYSICAL_FEATURE_SPLIT = {
    "train": "train",
    "val": "val",
    "val_eval": "val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen Qwen3-VL + LION features for GeoVLM.")
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL)
    parser.add_argument("--split", choices=[*ALL_SPLITS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lion-quality", choices=["low", "mid", "high"], default=DEFAULT_LION_QUALITY)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--lion-device", default="cuda")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _is_distributed() -> bool:
    return _env_int("WORLD_SIZE", 1) > 1


def _resolve_splits(split: str) -> tuple[str, ...]:
    return ALL_SPLITS if split == "all" else (split,)


def _resolve_physical_splits(logical_splits: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for split in logical_splits:
        physical = PHYSICAL_FEATURE_SPLIT[split]
        if physical not in ordered:
            ordered.append(physical)
    return tuple(ordered)


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _init_distributed(timeout_seconds: int) -> tuple[bool, int, int, int]:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return False, world_size, rank, local_rank
    ensure(torch.cuda.is_available(), "Distributed GeoVLM extract requires CUDA.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    return True, world_size, rank, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _resolve_device(spec: str, local_rank: int) -> str:
    if spec.startswith("cuda"):
        return f"cuda:{local_rank}" if _is_distributed() else spec
    return spec


def _maybe_relaunch_with_torchrun(args: argparse.Namespace) -> None:
    if _is_distributed():
        return
    if os.getenv("GEOVLM_DISABLE_AUTO_TORCHRUN") == "1":
        return
    if not (str(args.qwen_device).startswith("cuda") or str(args.lion_device).startswith("cuda")):
        return
    gpu_count = torch.cuda.device_count()
    nproc = min(4, gpu_count)
    if nproc <= 1:
        return
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        "29530",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)
    raise SystemExit(0)


def _rank_index_path(index_path: Path, rank: int) -> Path:
    return index_path.with_name(f"{index_path.stem}.rank{rank}{index_path.suffix}")


def _merge_rank_indexes(index_path: Path, world_size: int) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for rank in range(world_size):
        shard_path = _rank_index_path(index_path, rank)
        shard_rows = load_json(shard_path)
        ensure(isinstance(shard_rows, list), f"Rank shard index must be a JSON list: {shard_path}")
        merged.extend(shard_rows)
    merged.sort(key=lambda row: int(row["prepared_index"]))
    dump_json(index_path, merged)
    return merged


def _question_id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["question_id"]) for row in rows}


def _rewrite_index_rows_for_split(rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    return [{**row, "prepared_split": split} for row in rows]


def _group_rows_by_frame(indexed_rows: list[tuple[int, dict[str, Any]]]) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    frame_order: list[str] = []
    for prepared_index, row in indexed_rows:
        frame_token = str(row["frame_token"])
        if frame_token not in grouped:
            grouped[frame_token] = []
            frame_order.append(frame_token)
        grouped[frame_token].append((prepared_index, row))
    return [(frame_token, grouped[frame_token]) for frame_token in frame_order]


def _partition_frame_groups(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    world_size: int,
) -> list[list[tuple[str, list[tuple[int, dict[str, Any]]]]]]:
    frame_groups = _group_rows_by_frame(indexed_rows)
    if world_size <= 1:
        return [frame_groups]
    shards: list[list[tuple[str, list[tuple[int, dict[str, Any]]]]]] = [[] for _ in range(world_size)]
    shard_loads = [0 for _ in range(world_size)]
    for frame_group in frame_groups:
        shard_index = min(range(world_size), key=lambda idx: (shard_loads[idx], idx))
        shards[shard_index].append(frame_group)
        shard_loads[shard_index] += len(frame_group[1])
    return shards


def main() -> None:
    args = parse_args()
    _maybe_relaunch_with_torchrun(args)
    distributed, world_size, rank, local_rank = _init_distributed(args.ddp_timeout_seconds)
    try:
        resolved = resolve_dataset_version_paths(
            args.dataset_version,
            prepared_dir=args.prepared_dir,
            work_dir=args.work_dir,
        )
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        manifest = load_json(prepared_dir / "split_manifest.json")
        ensure(isinstance(manifest, dict), f"split_manifest.json must be an object: {prepared_dir / 'split_manifest.json'}")
        ensure(
            manifest.get("dataset_version") == resolved["dataset_version"],
            f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
        )
        worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
        prompt_bundle = load_prompt_bundle(Path(resolved["qa_json"]).resolve())
        info_index = build_info_index(args.info_pkl.resolve())

        qwen_device = _resolve_device(args.qwen_device, local_rank)
        lion_device = _resolve_device(args.lion_device, local_rank)
        qwen_runtime = load_qwen3_vl_runtime()
        qwen_model_runtime = build_qwen3_vl_model_runtime(base_runtime=qwen_runtime, device=qwen_device)
        lion_runtime = build_lion_model_runtime(args.lion_quality, device=lion_device)

        split_index_paths = {
            "train": worktree["feature_index_train"],
            "val": worktree["feature_index_val"],
            "val_eval": worktree["feature_index_val_eval"],
        }
        feature_root = worktree["features"]
        feature_root.mkdir(parents=True, exist_ok=True)
        manifest_splits: dict[str, dict[str, object]] = {}
        logical_splits = _resolve_splits(args.split)
        if "val" in logical_splits or "val_eval" in logical_splits:
            val_rows_full = load_prepared_records(prepared_dir, split="val")
            val_eval_rows_full = load_prepared_records(prepared_dir, split="val_eval")
            ensure(
                _question_id_set(val_rows_full) == _question_id_set(val_eval_rows_full),
                "GeoVLM shares val_eval extracted features with val, but val and val_eval question_id sets differ.",
            )
        processed_physical_splits = _resolve_physical_splits(logical_splits)
        if "val" in processed_physical_splits:
            stale_val_eval_frame_dir = feature_root / "val_eval" / "frames"
            if stale_val_eval_frame_dir.is_dir():
                has_payloads = any(path.is_file() for path in stale_val_eval_frame_dir.iterdir())
                ensure(
                    not has_payloads,
                    "Feature layout is stale; val_eval must alias val and should not have its own frame directory. "
                    "Remove stale features/val_eval/frames and re-run extract.",
                )
        for physical_split in processed_physical_splits:
            rows = load_prepared_records(prepared_dir, split=physical_split)
            validate_subtemplates((row["subtemplate"] for row in rows), prompt_bundle)
            if args.limit is not None:
                rows = rows[: args.limit]
            split_dir = feature_root / physical_split
            split_dir.mkdir(parents=True, exist_ok=True)
            split_frame_dir = split_dir / "frames"
            split_frame_dir.mkdir(parents=True, exist_ok=True)
            indexed_rows = list(enumerate(rows))
            all_frame_groups = _partition_frame_groups(indexed_rows, world_size if distributed else 1)
            local_frame_groups = all_frame_groups[rank]
            shard_index_path = _rank_index_path(split_index_paths[physical_split], rank)
            feature_index: list[dict[str, object]] = []
            total_local_samples = sum(len(frame_group_rows) for _, frame_group_rows in local_frame_groups)
            progress = tqdm(
                total=total_local_samples,
                desc=f"Extract:{physical_split}:r{rank}",
                unit="sample",
                disable=not _is_main_process(rank),
            )
            for frame_token, frame_group_rows in local_frame_groups:
                first_prepared_index, first_row = frame_group_rows[0]
                frame_sample = resolve_prepared_sample(
                    first_row,
                    info_index,
                    dataset_version=resolved["dataset_version"],
                    prepared_split=physical_split,
                    prepared_index=first_prepared_index,
                )
                frame_qwen_inputs = prepare_qwen3_vl_inputs(qwen_runtime, frame_sample, prompt_bundle=prompt_bundle)
                frame_qwen_vision = extract_qwen3_vl_vision_features(qwen_model_runtime, frame_qwen_inputs)
                lion_outputs = extract_lion_tokens(frame_sample, lion_runtime, max_objects=args.max_objects)
                frame_storage = build_frame_storage_tensors(
                    image_tokens=frame_qwen_vision.image_tokens.squeeze(0),
                    bev_tokens=lion_outputs.bev_tokens.squeeze(0),
                    object_tokens=lion_outputs.object_tokens.squeeze(0),
                    raw_object_tokens=lion_outputs.raw_object_tokens.squeeze(0),
                    image_token_budget_per_camera=256,
                    bev_token_budget=1024,
                    object_token_budget=128,
                )
                frame_feature_path = (split_frame_dir / f"{frame_token}.pt").resolve()
                save_feature_payload(
                    frame_feature_path,
                    {
                        **frame_storage,
                        "frame_token": frame_sample.frame_token,
                        "feature_layout_version": FEATURE_LAYOUT_VERSION,
                        "image_token_dim": int(frame_qwen_vision.image_tokens.shape[-1]),
                        "bev_token_dim": int(lion_outputs.bev_tokens.shape[-1]),
                        "object_token_dim": int(lion_outputs.object_tokens.shape[-1]),
                        "raw_object_token_dim": int(lion_outputs.raw_object_tokens.shape[-1]),
                        "raw_image_token_counts": list(frame_qwen_vision.image_token_counts),
                        "raw_bev_token_count": int(lion_outputs.bev_tokens.shape[1]),
                        "raw_object_token_count": int(lion_outputs.object_tokens.shape[1]),
                        "bev_grid_size": list(lion_outputs.bev_grid_size),
                        "query_feature_dim": int(lion_outputs.query_feature_dim),
                        "object_local_feature_dim": int(lion_outputs.object_local_feature_dim),
                        "raw_object_boxes": lion_outputs.pred_boxes.detach().float().cpu(),
                        "raw_object_scores": lion_outputs.pred_scores.detach().float().cpu(),
                        "raw_object_labels": lion_outputs.pred_labels.detach().float().cpu(),
                    },
                )
                for prepared_index, row in frame_group_rows:
                    sample = resolve_prepared_sample(
                        row,
                        info_index,
                        dataset_version=resolved["dataset_version"],
                        prepared_split=physical_split,
                        prepared_index=prepared_index,
                    )
                    feature_index.append(
                        {
                            "question_id": sample.question_id,
                            "frame_token": sample.frame_token,
                            "subtemplate": sample.subtemplate,
                            "frame_feature_path": str(frame_feature_path),
                            "prepared_split": physical_split,
                            "prepared_index": prepared_index,
                            "rank": rank,
                            "feature_storage": FRAME_ONLY_FEATURE_STORAGE,
                            "feature_layout_version": FEATURE_LAYOUT_VERSION,
                        }
                    )
                    if _is_main_process(rank):
                        progress.update(1)
                        progress.set_postfix(question_id=sample.question_id, frame_token=frame_token)
            dump_json(shard_index_path, feature_index)
            if distributed:
                dist.barrier()
            if _is_main_process(rank):
                merged_rows = _merge_rank_indexes(split_index_paths[physical_split], world_size if distributed else 1)
                logical_outputs = ("train",) if physical_split == "train" else ("val", "val_eval")
                for logical_split in logical_outputs:
                    logical_rows = _rewrite_index_rows_for_split(merged_rows, logical_split)
                    dump_json(split_index_paths[logical_split], logical_rows)
                    manifest_splits[logical_split] = {
                        "count": len(logical_rows),
                        "unique_frames": len({str(row["frame_token"]) for row in rows}),
                        "index_path": str(split_index_paths[logical_split]),
                        "rank_shards": world_size if distributed else 1,
                        "partitioning": "frame_token",
                        "frame_feature_dir": str(split_frame_dir),
                        "feature_storage": FRAME_ONLY_FEATURE_STORAGE,
                        "question_features_stored": False,
                        "physical_split": physical_split,
                    }
                    if logical_split != physical_split:
                        manifest_splits[logical_split]["alias_of"] = physical_split
            if distributed:
                dist.barrier()

        if _is_main_process(rank):
            split_aliases = {"val_eval": "val"} if "val" in manifest_splits else {}
            dump_json(
                worktree["features_manifest"],
                {
                    "dataset_version": resolved["dataset_version"],
                    "prepared_dir": str(prepared_dir),
                    "qa_json": str(resolved["qa_json"]),
                    "info_pkl": str(args.info_pkl.resolve()),
                    "lion_quality": args.lion_quality,
                    "max_objects": args.max_objects,
                    "qwen_device": qwen_model_runtime.device,
                    "lion_device": lion_runtime.device,
                    "distributed": distributed,
                    "world_size": world_size,
                    "feature_storage": FRAME_ONLY_FEATURE_STORAGE,
                    "feature_layout_version": FEATURE_LAYOUT_VERSION,
                    "split_aliases": split_aliases,
                    "feature_token_budgets": {
                        "image_token_budget_per_camera": 256,
                        "bev_token_budget": 1024,
                        "object_token_budget": 128,
                        "question_token_budget": 256,
                    },
                    "splits": manifest_splits,
                },
            )
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
