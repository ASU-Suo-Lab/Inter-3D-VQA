#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOVLM_ROOT = REPO_ROOT / "VLM" / "GeoVLM-Intersection"
if str(GEOVLM_ROOT) not in sys.path:
    sys.path.insert(0, str(GEOVLM_ROOT))

from geovlm_intersection.backbones.lion_token_adapter import build_lion_model_runtime, extract_lion_tokens  # noqa: E402
from geovlm_intersection.config.common import DEFAULT_LION_QUALITY  # noqa: E402
from intersection_lidar_utils import DEFAULT_DATASET_VERSION, ensure, load_frame_rows, resolve_dataset_version_paths

LIDAR_ROI_X_MIN = -60.0
LIDAR_ROI_Y_MIN = -60.0
LIDAR_ROI_Z_MIN = -8.0
LIDAR_ROI_X_MAX = 90.0
LIDAR_ROI_Y_MAX = 80.0
LIDAR_ROI_Z_MAX = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level LION BEV tokens for LlamaFactory LiDAR prefix training.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lion-quality", choices=["low", "mid", "high"], default=DEFAULT_LION_QUALITY)
    parser.add_argument("--scene-token-budget", type=int, default=256)
    parser.add_argument("--object-candidate-limit", dest="object_candidate_limit", type=int, default=32)
    parser.add_argument("--max-objects", dest="object_candidate_limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--object-score-threshold", type=float, default=0.4)
    parser.add_argument("--object-geometry-points", type=int, default=0)
    parser.add_argument("--geometry-token-budget", type=int, default=0)
    parser.add_argument("--geometry-neighbors", type=int, default=64)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _filter_object_candidates(
    object_tokens: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_label_names: list[str],
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
    pre_filter_count = int(pred_scores.shape[0])
    if pre_filter_count == 0 or score_threshold <= 0:
        return object_tokens, pred_boxes, pred_scores, pred_labels, pred_label_names, pre_filter_count

    keep_mask = pred_scores >= score_threshold
    if not torch.any(keep_mask):
        empty_tokens = object_tokens[:0]
        empty_boxes = pred_boxes[:0]
        empty_scores = pred_scores[:0]
        empty_labels = pred_labels[:0]
        return empty_tokens, empty_boxes, empty_scores, empty_labels, [], pre_filter_count

    keep_list = keep_mask.detach().cpu().tolist()
    filtered_names = [name for name, keep in zip(pred_label_names, keep_list, strict=False) if keep]
    return (
        object_tokens[keep_mask],
        pred_boxes[keep_mask],
        pred_scores[keep_mask],
        pred_labels[keep_mask],
        filtered_names,
        pre_filter_count,
    )


def _frame_token_sort_key(frame_token: str) -> tuple[int, int]:
    parts = frame_token.split("-", maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return 0, 0


def _build_minimal_sample(frame_row: dict[str, object]) -> SimpleNamespace:
    frame_token = str(frame_row["frame_token"])
    point_cloud_path = Path(str(frame_row["point_cloud_path"])).resolve()
    ensure(point_cloud_path.is_file(), f"Point cloud file not found: {point_cloud_path}")
    return SimpleNamespace(frame_token=frame_token, point_cloud_path=point_cloud_path)


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
    for i in range(num_centroids):
        centroids[i] = current
        centroid_xyz = points_xyz[current]
        current_distances = np.sum((points_xyz - centroid_xyz) ** 2, axis=1)
        distances = np.minimum(distances, current_distances)
        current = int(np.argmax(distances))
    return centroids


def _extract_geometry_tokens(
    point_cloud_path: Path,
    token_budget: int,
    neighbor_count: int,
) -> torch.Tensor:
    points = _crop_point_cloud(_load_point_cloud(point_cloud_path))
    token_dim = neighbor_count * 4
    if points.shape[0] == 0:
        return torch.zeros((token_budget, token_dim), dtype=torch.float32)

    xyz = points[:, :3]
    sampled_indices = _farthest_point_sampling(xyz, token_budget)
    geometry_tokens = np.zeros((token_budget, token_dim), dtype=np.float32)

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
        geometry_tokens[token_idx] = neighborhood.reshape(-1)

    return torch.from_numpy(geometry_tokens)


def _extract_object_geometry_tokens(
    point_cloud_path: Path,
    pred_boxes: torch.Tensor,
    points_per_object: int,
) -> torch.Tensor:
    points = _crop_point_cloud(_load_point_cloud(point_cloud_path))
    if pred_boxes.numel() == 0:
        return torch.zeros((0, points_per_object * 4), dtype=torch.float32)

    geometry_tokens = np.zeros((pred_boxes.shape[0], points_per_object * 4), dtype=np.float32)
    if points.shape[0] == 0:
        return torch.from_numpy(geometry_tokens)

    xyz = points[:, :3]
    intensity = points[:, 3:4]
    pred_boxes_np = pred_boxes.detach().cpu().numpy().astype(np.float32)
    for object_idx, box in enumerate(pred_boxes_np):
        center = box[:3]
        dims = np.maximum(box[3:6], 1e-3)
        half_dims = dims * np.array([0.6, 0.6, 0.55], dtype=np.float32)
        box_mask = (
            (np.abs(xyz[:, 0] - center[0]) <= half_dims[0])
            & (np.abs(xyz[:, 1] - center[1]) <= half_dims[1])
            & (np.abs(xyz[:, 2] - center[2]) <= half_dims[2])
        )
        selected_points = points[box_mask]
        if selected_points.shape[0] < points_per_object:
            distances = np.sum((xyz - center[None, :]) ** 2, axis=1)
            neighbor_indices = np.argsort(distances)[:points_per_object]
            selected_points = points[neighbor_indices]
        elif selected_points.shape[0] > points_per_object:
            distances = np.sum((selected_points[:, :3] - center[None, :]) ** 2, axis=1)
            neighbor_indices = np.argsort(distances)[:points_per_object]
            selected_points = selected_points[neighbor_indices]

        local_points = np.zeros((points_per_object, 4), dtype=np.float32)
        keep = min(selected_points.shape[0], points_per_object)
        if keep > 0:
            local_points[:keep, :3] = selected_points[:keep, :3] - center[None, :]
            local_points[:keep, 3:4] = selected_points[:keep, 3:4]
        geometry_tokens[object_idx] = local_points.reshape(-1)

    return torch.from_numpy(geometry_tokens)


def main() -> None:
    args = parse_args()
    ensure(torch.cuda.is_available(), "CUDA is required for LION extraction.")

    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = resolved["prepared_dir"]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolved["work_dir"] / "bev_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = build_lion_model_runtime(args.lion_quality, device="cuda:0")
    frame_rows = load_frame_rows(prepared_dir, args.split, args.limit)
    frame_token_filter = {str(frame_row["frame_token"]) for frame_row in frame_rows}
    scene_to_frame_tokens: dict[str, list[str]] = {}
    for frame_row in frame_rows:
        scene_id = str(frame_row["scene_id"])
        scene_to_frame_tokens.setdefault(scene_id, []).append(str(frame_row["frame_token"]))
    for scene_id, tokens in scene_to_frame_tokens.items():
        scene_to_frame_tokens[scene_id] = sorted(tokens, key=_frame_token_sort_key)
    written = 0
    skipped = 0
    last_scene_tokens: torch.Tensor | None = None
    last_object_tokens: torch.Tensor | None = None
    last_object_geometry_tokens: torch.Tensor | None = None
    last_geometry_tokens: torch.Tensor | None = None
    last_bev_grid_size: tuple[int, int] | None = None
    progress = tqdm(frame_rows, desc="Extract LION scene/object", unit="frame")
    for frame_row in progress:
        frame_token = str(frame_row["frame_token"])
        scene_output_path = output_dir / f"{frame_token}.scene.pt"
        object_output_path = output_dir / f"{frame_token}.object.pt"
        object_geometry_output_path = output_dir / f"{frame_token}.object_geometry.pt"
        geometry_output_path = output_dir / f"{frame_token}.geometry.pt"
        object_geometry_ready = args.object_geometry_points <= 0 or object_geometry_output_path.exists()
        geometry_ready = args.geometry_token_budget <= 0 or geometry_output_path.exists()
        if (
            scene_output_path.exists()
            and object_output_path.exists()
            and object_geometry_ready
            and geometry_ready
            and not args.overwrite
        ):
            skipped += 1
            progress.set_postfix(written=written, skipped=skipped)
            continue

        sample = _build_minimal_sample(frame_row)
        lion_outputs = extract_lion_tokens(
            sample,
            runtime,
            max_objects=args.object_candidate_limit,
            bev_token_budget=args.scene_token_budget,
        )
        scene_tokens = lion_outputs.bev_tokens.squeeze(0).detach().cpu()
        object_tokens = lion_outputs.object_tokens.squeeze(0).detach().cpu()
        pred_boxes = lion_outputs.pred_boxes.detach().cpu()
        pred_scores = lion_outputs.pred_scores.detach().cpu()
        pred_labels = lion_outputs.pred_labels.detach().cpu()
        class_names = list(runtime.cfg.CLASS_NAMES)
        pred_label_names: list[str] = []
        for label in pred_labels.tolist():
            label_index = int(label)
            if 1 <= label_index <= len(class_names):
                label_index -= 1
            pred_label_names.append(class_names[label_index] if 0 <= label_index < len(class_names) else "unknown")
        object_tokens, pred_boxes, pred_scores, pred_labels, pred_label_names, pre_filter_object_count = _filter_object_candidates(
            object_tokens,
            pred_boxes,
            pred_scores,
            pred_labels,
            pred_label_names,
            args.object_score_threshold,
        )
        torch.save(
            {
                "lidar_scene_tokens": scene_tokens,
                "bev_grid_size": [int(v) for v in lion_outputs.bev_grid_size],
            },
            scene_output_path,
        )
        torch.save(
            {
                "lidar_object_tokens": object_tokens,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels,
                "pred_label_names": pred_label_names,
                "object_candidate_limit": args.object_candidate_limit,
                "object_score_threshold": args.object_score_threshold,
                "pre_filter_object_count": pre_filter_object_count,
            },
            object_output_path,
        )
        if args.object_geometry_points > 0:
            object_geometry_tokens = _extract_object_geometry_tokens(
                sample.point_cloud_path,
                pred_boxes=pred_boxes,
                points_per_object=args.object_geometry_points,
            )
            torch.save(
                {
                    "lidar_object_geometry_tokens": object_geometry_tokens,
                    "object_geometry_points": args.object_geometry_points,
                },
                object_geometry_output_path,
            )
            last_object_geometry_tokens = object_geometry_tokens
        if args.geometry_token_budget > 0:
            geometry_tokens = _extract_geometry_tokens(
                sample.point_cloud_path,
                token_budget=args.geometry_token_budget,
                neighbor_count=args.geometry_neighbors,
            )
            torch.save(
                {
                    "lidar_geometry_tokens": geometry_tokens,
                    "geometry_neighbor_count": args.geometry_neighbors,
                },
                geometry_output_path,
            )
            last_geometry_tokens = geometry_tokens
        last_scene_tokens = scene_tokens
        last_object_tokens = object_tokens
        last_bev_grid_size = tuple(int(v) for v in lion_outputs.bev_grid_size)
        written += 1
        progress.set_postfix(written=written, skipped=skipped)

    summary = {
        "dataset_version": args.dataset_version,
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "source": (
            "lion_rawscene_objroute"
            if args.object_geometry_points > 0
            else "lion_sceneobj_objgeom"
            if args.geometry_token_budget > 0
            else "lion_sceneobj"
        ),
        "lion_quality": args.lion_quality,
        "scene_token_budget": args.scene_token_budget,
        "object_candidate_limit": args.object_candidate_limit,
        "object_score_threshold": args.object_score_threshold,
        "object_geometry_points": args.object_geometry_points,
        "object_geometry_token_dim": int(last_object_geometry_tokens.shape[-1]) if last_object_geometry_tokens is not None else None,
        "object_geometry_token_count": int(last_object_geometry_tokens.shape[0]) if last_object_geometry_tokens is not None else None,
        "geometry_token_budget": args.geometry_token_budget,
        "geometry_neighbor_count": args.geometry_neighbors,
        "scene_token_dim": int(last_scene_tokens.shape[-1]) if last_scene_tokens is not None else None,
        "scene_token_count": int(last_scene_tokens.shape[0]) if last_scene_tokens is not None else None,
        "object_token_dim": int(last_object_tokens.shape[-1]) if last_object_tokens is not None else None,
        "object_token_count": int(last_object_tokens.shape[0]) if last_object_tokens is not None else None,
        "geometry_token_dim": int(last_geometry_tokens.shape[-1]) if last_geometry_tokens is not None else None,
        "geometry_token_count": int(last_geometry_tokens.shape[0]) if last_geometry_tokens is not None else None,
        "bev_grid_size": list(last_bev_grid_size) if last_bev_grid_size is not None else None,
        "written": written,
        "skipped": skipped,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
