from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_ROOT = REPO_ROOT / "VLM"
PREPARED_DATA_ROOT = REPO_ROOT / "data" / "intersection_lidar_prepared"
DEFAULT_DATASET_VERSION = "v5"

DATASET_VERSION_DEFAULTS = {
    "v5": {
        "prepared_dir": PREPARED_DATA_ROOT / "v5",
        "work_dir": VLM_ROOT / "work_dirs" / "intersection_lidar_v5",
    },
    "v6": {
        "prepared_dir": PREPARED_DATA_ROOT / "v6",
        "work_dir": VLM_ROOT / "work_dirs" / "intersection_lidar_v6",
    },
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_frame_rows(prepared_dir: Path, split: str, limit: int | None) -> list[dict[str, object]]:
    split_names = ("train", "val") if split == "all" else (split,)
    frame_rows: list[dict[str, object]] = []
    for split_name in split_names:
        frame_path = prepared_dir / f"frames_{split_name}.json"
        ensure(frame_path.is_file(), f"Prepared frame manifest not found: {frame_path}")
        rows = load_json(frame_path)
        ensure(isinstance(rows, list) and rows, f"{frame_path} does not contain a non-empty list.")
        frame_rows.extend(rows)
    return frame_rows[:limit] if limit is not None else frame_rows


def resolve_dataset_version_paths(
    dataset_version: str,
    *,
    prepared_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Path]:
    if dataset_version not in DATASET_VERSION_DEFAULTS:
        raise ValueError(f"Unsupported dataset_version={dataset_version}")

    defaults = DATASET_VERSION_DEFAULTS[dataset_version]
    return {
        "prepared_dir": Path(prepared_dir if prepared_dir is not None else defaults["prepared_dir"]).expanduser().resolve(),
        "work_dir": Path(work_dir if work_dir is not None else defaults["work_dir"]).expanduser().resolve(),
    }
