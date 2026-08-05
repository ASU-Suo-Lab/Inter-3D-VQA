#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


from intersection_lidar_utils import DEFAULT_DATASET_VERSION, ensure, load_frame_rows, resolve_dataset_version_paths


LIDAR_ROI_X_MIN = -60.0
LIDAR_ROI_Y_MIN = -60.0
LIDAR_ROI_Z_MIN = -8.0
LIDAR_ROI_X_MAX = 90.0
LIDAR_ROI_Y_MAX = 80.0
LIDAR_ROI_Z_MAX = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed raw point-cloud patch tokens for LlamaFactory LiDAR light-encoder training."
    )
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--object-token-budget", type=int, default=16)
    parser.add_argument("--scene-token-budget", type=int, default=256)
    parser.add_argument("--neighbor-count", type=int, default=32)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _frame_token_sort_key(frame_token: str) -> tuple[int, int]:
    parts = frame_token.split("-", maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return 0, 0


def _load_point_cloud(point_cloud_path: Path) -> np.ndarray:
    suffix = point_cloud_path.suffix.lower()
    if suffix == ".npy":
        points = np.load(point_cloud_path)
    elif suffix == ".npz":
        archive = np.load(point_cloud_path)
        points = archive["points"] if "points" in archive else archive[next(iter(archive.files))]
    elif suffix == ".pt":
        points = torch.load(point_cloud_path, map_location="cpu")
        if isinstance(points, dict):
            points = points["points"] if "points" in points else points[next(iter(points))]
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
    elif suffix == ".bin":
        flat = np.fromfile(point_cloud_path, dtype=np.float32)
        if flat.size % 5 == 0:
            points = flat.reshape(-1, 5)
        elif flat.size % 4 == 0:
            points = flat.reshape(-1, 4)
        else:
            raise ValueError(f"Cannot infer point dimension for binary point cloud: {point_cloud_path}")
    else:
        raise ValueError(f"Unsupported point cloud file type: {point_cloud_path}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Point cloud array must be 2D, got shape {points.shape} from {point_cloud_path}")
    if points.shape[1] < 4:
        pad = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, pad], axis=1)
    return points[:, :4]


def _crop_point_cloud(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    roi_mask = (
        (points[:, 0] >= LIDAR_ROI_X_MIN)
        & (points[:, 0] <= LIDAR_ROI_X_MAX)
        & (points[:, 1] >= LIDAR_ROI_Y_MIN)
        & (points[:, 1] <= LIDAR_ROI_Y_MAX)
        & (points[:, 2] >= LIDAR_ROI_Z_MIN)
        & (points[:, 2] <= LIDAR_ROI_Z_MAX)
    )
    cropped = points[roi_mask]
    return cropped if cropped.shape[0] > 0 else points


def _farthest_point_sampling(points_xyz: np.ndarray, num_centroids: int) -> np.ndarray:
    num_points = points_xyz.shape[0]
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    centroids = np.zeros((num_centroids,), dtype=np.int64)
    distances = np.full((num_points,), np.inf, dtype=np.float32)
    current = int(np.argmax(np.sum(points_xyz ** 2, axis=1)))
    for idx in range(num_centroids):
        centroids[idx] = current
        centroid_xyz = points_xyz[current]
        current_distances = np.sum((points_xyz - centroid_xyz) ** 2, axis=1)
        distances = np.minimum(distances, current_distances)
        current = int(np.argmax(distances))
    return centroids


def _extract_patch_tokens(point_cloud_path: Path, token_budget: int, neighbor_count: int) -> torch.Tensor:
    token_dim = neighbor_count * 4
    points = _crop_point_cloud(_load_point_cloud(point_cloud_path))
    if token_budget <= 0:
        return torch.zeros((0, token_dim), dtype=torch.float32)
    if points.shape[0] == 0:
        return torch.zeros((token_budget, token_dim), dtype=torch.float32)

    xyz = points[:, :3]
    sampled_indices = _farthest_point_sampling(xyz, token_budget)
    tokens = np.zeros((token_budget, token_dim), dtype=np.float32)

    for token_idx, centroid_idx in enumerate(sampled_indices):
        centroid_xyz = xyz[centroid_idx]
        distances = np.sum((xyz - centroid_xyz) ** 2, axis=1)
        neighbor_indices = np.argsort(distances)[:neighbor_count]
        neighborhood = np.zeros((neighbor_count, 4), dtype=np.float32)
        selected = points[neighbor_indices]
        keep = min(selected.shape[0], neighbor_count)
        if keep > 0:
            neighborhood[:keep] = selected[:keep]
            neighborhood[:keep, :3] -= centroid_xyz
        tokens[token_idx] = neighborhood.reshape(-1)

    return torch.from_numpy(tokens)


def main() -> None:
    args = parse_args()
    ensure(args.object_token_budget > 0, f"--object-token-budget must be positive, got: {args.object_token_budget}")
    ensure(args.scene_token_budget > 0, f"--scene-token-budget must be positive, got: {args.scene_token_budget}")
    ensure(args.neighbor_count > 0, f"--neighbor-count must be positive, got: {args.neighbor_count}")

    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = resolved["prepared_dir"]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolved["work_dir"] / "raw_patch_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = load_frame_rows(prepared_dir, args.split, args.limit)
    scene_to_frame_tokens: dict[str, list[str]] = {}
    for frame_row in frame_rows:
        scene_id = str(frame_row["scene_id"])
        scene_to_frame_tokens.setdefault(scene_id, []).append(str(frame_row["frame_token"]))
    for scene_id, tokens in scene_to_frame_tokens.items():
        scene_to_frame_tokens[scene_id] = sorted(tokens, key=_frame_token_sort_key)

    written = 0
    skipped = 0
    last_object_tokens: torch.Tensor | None = None
    last_scene_tokens: torch.Tensor | None = None
    progress = tqdm(frame_rows, desc="Extract raw patch tokens", unit="frame")
    for frame_row in progress:
        frame_token = str(frame_row["frame_token"])
        point_cloud_path = Path(str(frame_row["point_cloud_path"])).resolve()
        ensure(point_cloud_path.is_file(), f"Point cloud file not found: {point_cloud_path}")

        object_output_path = output_dir / f"{frame_token}.object.pt"
        scene_output_path = output_dir / f"{frame_token}.scene.pt"
        if object_output_path.exists() and scene_output_path.exists() and not args.overwrite:
            skipped += 1
            progress.set_postfix(written=written, skipped=skipped)
            continue

        object_tokens = _extract_patch_tokens(
            point_cloud_path,
            token_budget=args.object_token_budget,
            neighbor_count=args.neighbor_count,
        )
        scene_tokens = _extract_patch_tokens(
            point_cloud_path,
            token_budget=args.scene_token_budget,
            neighbor_count=args.neighbor_count,
        )

        torch.save(
            {
                "lidar_object_tokens": object_tokens,
                "token_source": "raw_patch",
                "neighbor_count": args.neighbor_count,
                "point_feature_dim": 4,
            },
            object_output_path,
        )
        torch.save(
            {
                "lidar_scene_tokens": scene_tokens,
                "token_source": "raw_patch",
                "neighbor_count": args.neighbor_count,
                "point_feature_dim": 4,
            },
            scene_output_path,
        )
        last_object_tokens = object_tokens
        last_scene_tokens = scene_tokens
        written += 1
        progress.set_postfix(written=written, skipped=skipped)

    summary = {
        "dataset_version": args.dataset_version,
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "source": "rawpatch_objroute",
        "object_token_budget": args.object_token_budget,
        "scene_token_budget": args.scene_token_budget,
        "neighbor_count": args.neighbor_count,
        "object_token_dim": int(last_object_tokens.shape[-1]) if last_object_tokens is not None else None,
        "object_token_count": int(last_object_tokens.shape[0]) if last_object_tokens is not None else None,
        "scene_token_dim": int(last_scene_tokens.shape[-1]) if last_scene_tokens is not None else None,
        "scene_token_count": int(last_scene_tokens.shape[0]) if last_scene_tokens is not None else None,
        "written": written,
        "skipped": skipped,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
