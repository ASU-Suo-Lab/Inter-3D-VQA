from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geovlm_intersection.config.common import DATASET_VERSION_DEFAULTS, DEFAULT_INFO_PKL, LLM_ROOT

# Sensor keys are preserved from the upstream info pickle, but GeoVLM should reason
# in terms of the actual scene view direction rather than the physical key name.
# For SunLakes v5, the installed camera facing directions are opposite the sensor key.
CAMERA_ORDER = (
    ("CAM_SOUTH", "north_image", "north"),
    ("CAM_NORTH", "south_image", "south"),
    ("CAM_WEST", "east_image", "east"),
    ("CAM_EAST", "west_image", "west"),
)


@dataclass(frozen=True)
class CameraView:
    prepared_index: int
    cam_key: str
    image_name: str
    view_direction: str
    image_path: Path
    camera_intrinsics: np.ndarray
    lidar2camera: np.ndarray
    lidar2image: np.ndarray
    camera2lidar: np.ndarray


@dataclass(frozen=True)
class PreparedSample:
    dataset_version: str
    prepared_split: str
    prepared_index: int
    question_id: str
    frame_token: str
    scene_id: str
    chapter: str
    section: str
    subtemplate: str
    question: str
    answer: str
    point_cloud_path: Path
    image_paths: tuple[Path, ...]
    camera_views: tuple[CameraView, ...]
    structured_targets: dict[str, Any] | None
    raw_record: dict[str, Any]
    info_record: dict[str, Any]


def _resolve_data_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (LLM_ROOT / path).resolve()


def load_prepared_records(prepared_dir: Path, split: str = "val_eval") -> list[dict[str, Any]]:
    split_path = prepared_dir / f"{split}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing prepared split file: {split_path}")
    records = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Prepared split must be a JSON list: {split_path}")
    return records


def build_info_index(info_pkl: Path = DEFAULT_INFO_PKL) -> dict[str, dict[str, Any]]:
    if not info_pkl.is_file():
        raise FileNotFoundError(f"Missing info pickle: {info_pkl}")
    with info_pkl.open("rb") as file:
        infos = pickle.load(file)
    if not isinstance(infos, list):
        raise ValueError(f"Expected info pickle to contain a list, got: {type(infos).__name__}")
    index: dict[str, dict[str, Any]] = {}
    for row in infos:
        if not isinstance(row, dict) or "token" not in row:
            raise ValueError("Each info row must be a dict with a 'token' key.")
        index[str(row["token"])] = row
    return index


def load_point_cloud_xyzit(point_cloud_path: Path) -> np.ndarray:
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"Missing point cloud file: {point_cloud_path}")
    raw = np.fromfile(point_cloud_path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Point cloud file is empty: {point_cloud_path}")
    if raw.size % 5 != 0:
        raise ValueError(
            f"Expected point cloud to be float32 Nx5 (x,y,z,intensity,time), got flat size {raw.size}: {point_cloud_path}"
        )
    return raw.reshape(-1, 5)


def resolve_prepared_sample(
    record: dict[str, Any],
    info_index: dict[str, dict[str, Any]],
    *,
    dataset_version: str = "v5",
    prepared_split: str = "val_eval",
    prepared_index: int = 0,
) -> PreparedSample:
    if dataset_version != "v5":
        raise ValueError(f"GeoVLM-Intersection currently supports only v5, got: {dataset_version}")

    frame_token = str(record["frame_token"])
    info_row = info_index.get(frame_token)
    if info_row is None:
        raise KeyError(f"Frame token not found in info index: {frame_token}")

    images = record.get("images")
    if not isinstance(images, list) or len(images) != len(CAMERA_ORDER):
        raise ValueError(f"Prepared sample must contain {len(CAMERA_ORDER)} image paths, got: {images}")
    image_paths = tuple(_resolve_data_path(path) for path in images)

    point_cloud_path = _resolve_data_path(record["point_cloud_path"])
    info_lidar_path = _resolve_data_path(info_row["lidar_path"])
    if point_cloud_path != info_lidar_path:
        raise ValueError(
            f"Prepared point cloud path does not match info lidar path:\nprepared={point_cloud_path}\ninfo={info_lidar_path}"
        )

    cam_rows = info_row.get("cams")
    if not isinstance(cam_rows, dict):
        raise ValueError(f"Info row missing 'cams' mapping for frame token: {frame_token}")

    camera_views: list[CameraView] = []
    for prepared_idx, ((cam_key, image_name, view_direction), prepared_image_path) in enumerate(zip(CAMERA_ORDER, image_paths)):
        cam_info = cam_rows.get(cam_key)
        if not isinstance(cam_info, dict):
            raise KeyError(f"Missing camera entry {cam_key} for frame token: {frame_token}")
        info_image_path = _resolve_data_path(cam_info["image_paths"])
        if prepared_image_path != info_image_path:
            raise ValueError(
                f"Prepared image order does not match {cam_key}:\nprepared={prepared_image_path}\ninfo={info_image_path}"
            )
        camera_views.append(
            CameraView(
                prepared_index=prepared_idx,
                cam_key=cam_key,
                image_name=image_name,
                view_direction=view_direction,
                image_path=prepared_image_path,
                camera_intrinsics=np.asarray(cam_info["camera_intrinsics"], dtype=np.float32),
                lidar2camera=np.asarray(cam_info["lidar2camera"], dtype=np.float32),
                lidar2image=np.asarray(cam_info["lidar2image"], dtype=np.float32),
                camera2lidar=np.asarray(cam_info["camera2lidar"], dtype=np.float32),
            )
        )

    return PreparedSample(
        dataset_version=dataset_version,
        prepared_split=prepared_split,
        prepared_index=prepared_index,
        question_id=str(record["question_id"]),
        frame_token=frame_token,
        scene_id=str(record["scene_id"]),
        chapter=str(record["chapter"]),
        section=str(record["section"]),
        subtemplate=str(record["subtemplate"]),
        question=str(record["question"]),
        answer=str(record["answer"]),
        point_cloud_path=point_cloud_path,
        image_paths=image_paths,
        camera_views=tuple(camera_views),
        structured_targets=record.get("structured_targets"),
        raw_record=record,
        info_record=info_row,
    )


def resolve_default_prepared_dir(dataset_version: str = "v5") -> Path:
    if dataset_version not in DATASET_VERSION_DEFAULTS:
        raise ValueError(f"Unsupported dataset version: {dataset_version}")
    return Path(DATASET_VERSION_DEFAULTS[dataset_version]["prepared_dir"]).resolve()
