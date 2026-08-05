#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from intersection_lidar_utils import REPO_ROOT, ensure, load_json, resolve_dataset_version_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare standalone Intersection LiDAR frame manifests from QA JSON.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default="v5")
    parser.add_argument("--qa-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_data_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/data/"):
        return (REPO_ROOT / normalized.lstrip("/")).resolve()
    path = Path(normalized)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def split_scenes(scene_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    ensure(0.0 < train_ratio < 1.0, f"train_ratio must be between 0 and 1, got {train_ratio}")
    shuffled = sorted(scene_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * (1.0 - train_ratio)))
    ensure(val_count < len(shuffled), "Need at least one scene in both train and validation splits.")
    return set(shuffled[val_count:]), set(shuffled[:val_count])


def main() -> None:
    args = parse_args()
    qa_json = (args.qa_json or REPO_ROOT / f"intersection_qa_pairs_{args.dataset_version}.json").resolve()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.output_dir)
    output_dir = resolved["prepared_dir"]

    payload = load_json(qa_json)
    ensure(isinstance(payload, dict), f"Expected a JSON object in {qa_json}")
    metadata = payload.get("metadata", {})
    ensure(metadata.get("version") == args.dataset_version, f"QA version does not match {args.dataset_version}: {qa_json}")
    qa_pairs = payload.get("qa_pairs")
    ensure(isinstance(qa_pairs, list) and qa_pairs, f"No QA pairs found in {qa_json}")

    frames_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for qa in qa_pairs:
        scene_id = str(qa.get("scene_id", ""))
        frame_token = str(qa.get("frame_token", ""))
        point_cloud_path = str(qa.get("point_cloud_path", ""))
        ensure(scene_id and frame_token and point_cloud_path, "QA row is missing scene_id, frame_token, or point_cloud_path.")
        key = (scene_id, frame_token)
        frames_by_key.setdefault(
            key,
            {
                "scene_id": scene_id,
                "frame_token": frame_token,
                "point_cloud_path": str(resolve_data_path(point_cloud_path)),
            },
        )

    train_scenes, val_scenes = split_scenes(
        sorted({scene_id for scene_id, _ in frames_by_key}),
        args.train_ratio,
        args.seed,
    )
    frames = [frames_by_key[key] for key in sorted(frames_by_key)]
    train_frames = [row for row in frames if row["scene_id"] in train_scenes]
    val_frames = [row for row in frames if row["scene_id"] in val_scenes]
    ensure(train_frames and val_frames, "Frame split produced an empty partition.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (output_dir / "frames_train.json", train_frames),
        (output_dir / "frames_val.json", val_frames),
    ):
        with path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)

    manifest = {
        "dataset_version": args.dataset_version,
        "qa_json": str(qa_json),
        "output_dir": str(output_dir),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "scene_counts": {"train": len(train_scenes), "val": len(val_scenes)},
        "frame_counts": {"train": len(train_frames), "val": len(val_frames)},
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
