from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nuscenesqa_v5.data.templates import object_label_text
from nuscenesqa_v5.utils.io import dump_json, ensure

VISIBILITY_SCORE = {
    "fully visible": 1.0,
    "mostly visible": 0.75,
    "partly occluded": 0.5,
    "heavily occluded": 0.25,
    "not visible": 0.0,
}


def _sort_indices_by_points(info: Mapping[str, Any], count: int) -> np.ndarray:
    points = info.get("num_lidar_pts")
    if isinstance(points, np.ndarray) and points.shape[0] == count:
        return np.argsort(-points.astype(np.float32))
    return np.arange(count, dtype=np.int64)


def _visibility_score(info: Mapping[str, Any], tracking_id: str, annotation_key: str) -> float:
    ensure(annotation_key in info, f"Info row is missing required annotation key: {annotation_key}")
    annotations = info.get(annotation_key) or {}
    visibility = annotations.get("visibility_by_track_id") or {}
    label = visibility.get(tracking_id)
    if label is None:
        return 1.0
    return VISIBILITY_SCORE.get(str(label), 1.0)


def build_frame_features_raw(
    info: Mapping[str, Any],
    label_vocab: Sequence[str],
    object_limit: int,
    track_type_lookup: Mapping[str, str],
    annotation_key: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    gt_boxes_raw = info.get("gt_boxes")
    gt_names_raw = info.get("gt_names")
    velocities_raw = info.get("gt_boxes_velocity")
    tracking_ids = list(info.get("tracking_id") or [])
    gt_boxes = np.asarray(gt_boxes_raw if gt_boxes_raw is not None else np.zeros((0, 9), dtype=np.float32), dtype=np.float32)
    gt_names = np.asarray(gt_names_raw if gt_names_raw is not None else np.zeros((0,), dtype=object))
    velocities = np.asarray(velocities_raw if velocities_raw is not None else np.zeros((0, 2), dtype=np.float32), dtype=np.float32)
    count = int(gt_boxes.shape[0])
    if count == 0:
        return (
            np.zeros((object_limit, len(label_vocab) + 4), dtype=np.float32),
            np.zeros((object_limit, 7), dtype=np.float32),
            0,
        )

    order = _sort_indices_by_points(info, count)[:object_limit]
    object_features = np.zeros((object_limit, len(label_vocab) + 4), dtype=np.float32)
    bbox_features = np.zeros((object_limit, 7), dtype=np.float32)
    num_lidar_pts_raw = info.get("num_lidar_pts")
    num_lidar_pts = np.asarray(num_lidar_pts_raw if num_lidar_pts_raw is not None else np.zeros((count,), dtype=np.float32), dtype=np.float32)

    for out_idx, src_idx in enumerate(order.tolist()):
        tracking_id = str(tracking_ids[src_idx]) if src_idx < len(tracking_ids) else ""
        label = object_label_text(track_type_lookup.get(tracking_id, "unknown object"))
        if label in label_to_idx:
            object_features[out_idx, label_to_idx[label]] = 1.0
        vx = float(velocities[src_idx, 0]) if src_idx < velocities.shape[0] else 0.0
        vy = float(velocities[src_idx, 1]) if src_idx < velocities.shape[0] else 0.0
        speed = math.sqrt(vx * vx + vy * vy)
        object_features[out_idx, -4:] = np.array(
            [
                vx,
                vy,
                speed,
                _visibility_score(info, tracking_id, annotation_key),
            ],
            dtype=np.float32,
        )
        bbox_features[out_idx] = gt_boxes[src_idx, :7]

    return object_features.astype(np.float32), bbox_features.astype(np.float32), int(order.shape[0])


def _normalize_valid_rows(values: np.ndarray, mean: np.ndarray, std: np.ndarray, valid_count: int) -> np.ndarray:
    normalized = values.copy()
    if valid_count > 0:
        normalized[:valid_count] = (normalized[:valid_count] - mean) / std
    return normalized.astype(np.float32)


def compute_feature_stats(feature_rows: Sequence[tuple[np.ndarray, np.ndarray, int]]) -> dict[str, list[float]]:
    object_parts = []
    bbox_parts = []
    for object_features, bbox_features, valid_count in feature_rows:
        if valid_count <= 0:
            continue
        object_parts.append(object_features[:valid_count, -4:])
        bbox_parts.append(bbox_features[:valid_count, :7])
    ensure(object_parts and bbox_parts, "Training feature stats require at least one valid object row.")
    object_concat = np.concatenate(object_parts, axis=0).astype(np.float32)
    bbox_concat = np.concatenate(bbox_parts, axis=0).astype(np.float32)
    object_mean = object_concat.mean(axis=0)
    object_std = np.maximum(object_concat.std(axis=0), 1e-6)
    bbox_mean = bbox_concat.mean(axis=0)
    bbox_std = np.maximum(bbox_concat.std(axis=0), 1e-6)
    return {
        "continuous_feature_names": ["vx", "vy", "speed", "visibility"],
        "continuous_mean": object_mean.tolist(),
        "continuous_std": object_std.tolist(),
        "bbox_mean": bbox_mean.tolist(),
        "bbox_std": bbox_std.tolist(),
    }


def normalize_frame_features(
    object_features: np.ndarray,
    bbox_features: np.ndarray,
    valid_count: int,
    stats: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    object_mean = np.asarray(stats["continuous_mean"], dtype=np.float32)
    object_std = np.asarray(stats["continuous_std"], dtype=np.float32)
    bbox_mean = np.asarray(stats["bbox_mean"], dtype=np.float32)
    bbox_std = np.asarray(stats["bbox_std"], dtype=np.float32)
    object_normalized = object_features.copy()
    object_normalized[:, -4:] = _normalize_valid_rows(object_features[:, -4:], object_mean, object_std, valid_count)
    bbox_normalized = _normalize_valid_rows(bbox_features[:, :7], bbox_mean, bbox_std, valid_count)
    return object_normalized.astype(np.float32), bbox_normalized.astype(np.float32)


def write_feature_manifest(
    path: Path,
    *,
    dataset_version: str,
    label_vocab: Sequence[str],
    object_limit: int,
    object_feature_dim: int,
    bbox_feature_dim: int,
    feature_stats: Mapping[str, Sequence[float]],
) -> None:
    dump_json(
        path,
        {
            "dataset_version": str(dataset_version),
            "label_vocab": list(label_vocab),
            "object_limit": int(object_limit),
            "object_feature_dim": int(object_feature_dim),
            "bbox_feature_dim": int(bbox_feature_dim),
            "continuous_feature_names": list(feature_stats["continuous_feature_names"]),
            "continuous_mean": list(feature_stats["continuous_mean"]),
            "continuous_std": list(feature_stats["continuous_std"]),
            "bbox_mean": list(feature_stats["bbox_mean"]),
            "bbox_std": list(feature_stats["bbox_std"]),
        },
    )


def save_frame_feature(path: Path, object_features: np.ndarray, bbox_features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure(object_features.ndim == 2, f"Expected object_features rank 2, got {object_features.shape}")
    ensure(bbox_features.ndim == 2, f"Expected bbox_features rank 2, got {bbox_features.shape}")
    np.savez_compressed(path, object_features=object_features, bbox_features=bbox_features)
