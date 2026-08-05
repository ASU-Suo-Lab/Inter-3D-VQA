from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..extras.lidar_template import (
    extract_subtemplate_from_prompt,
    lidar_template_family_id_from_subtemplate,
)


COUNT_TARGET_DIM = 2
MOTION_TARGET_DIM = 3
COORD_TARGET_DIM = 10

COUNT_SCALE = 20.0
DISTANCE_SCALE = 80.0
SPEED_SCALE = 30.0
ACCELERATION_SCALE = 10.0
GLOBAL_COORD_SCALE = 100.0
IMAGE_COORD_SCALE = 2000.0
WAYPOINT_COORD_SCALE = 50.0

@lru_cache(maxsize=1)
def _load_eval_modules() -> tuple[Any, Any]:
    utils_dir = Path(__file__).resolve().parents[4] / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))

    base_module = importlib.import_module("evaluate_intersection_vqa")
    v6_module = importlib.import_module("evaluate_intersection_vqa_v6")
    return base_module, v6_module


def _extract_template_id(prompt: list[dict[str, str]]) -> str | None:
    return extract_subtemplate_from_prompt(prompt)


def build_lidar_template_family_id(prompt: list[dict[str, str]]) -> int:
    return lidar_template_family_id_from_subtemplate(_extract_template_id(prompt))


def _extract_bucket_values(base_module: Any, parsed_answer: dict[str, Any], patterns: list[str]) -> list[float]:
    if not patterns:
        return []
    flat = base_module.flatten_structure(parsed_answer)
    keys = base_module.resolve_field_patterns(flat, patterns)
    values: list[float] = []
    for key in keys:
        value = flat.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return values


def build_numeric_supervision(
    prompt: list[dict[str, str]],
    response: list[dict[str, str]],
) -> dict[str, list[float]]:
    count_targets = [0.0] * COUNT_TARGET_DIM
    count_mask = [0.0] * COUNT_TARGET_DIM
    motion_targets = [0.0] * MOTION_TARGET_DIM
    motion_mask = [0.0] * MOTION_TARGET_DIM
    coord_targets = [0.0] * COORD_TARGET_DIM
    coord_mask = [0.0] * COORD_TARGET_DIM

    template_id = _extract_template_id(prompt)
    if template_id is None or len(response) != 1:
        return {
            "numeric_count_targets": count_targets,
            "numeric_count_mask": count_mask,
            "numeric_motion_targets": motion_targets,
            "numeric_motion_mask": motion_mask,
            "numeric_coord_targets": coord_targets,
            "numeric_coord_mask": coord_mask,
        }

    answer_text = response[0].get("content", "")
    base_module, v6_module = _load_eval_modules()
    if template_id in getattr(v6_module, "V6_TEMPLATE_SPECS", {}):
        spec = v6_module.V6_TEMPLATE_SPECS[template_id]
    else:
        spec = getattr(base_module, "V5_TEMPLATE_SPECS", {}).get(template_id)

    if spec is None:
        return {
            "numeric_count_targets": count_targets,
            "numeric_count_mask": count_mask,
            "numeric_motion_targets": motion_targets,
            "numeric_motion_mask": motion_mask,
            "numeric_coord_targets": coord_targets,
            "numeric_coord_mask": coord_mask,
        }

    parsed_answer = spec.parser(answer_text)
    if not parsed_answer:
        return {
            "numeric_count_targets": count_targets,
            "numeric_count_mask": count_mask,
            "numeric_motion_targets": motion_targets,
            "numeric_motion_mask": motion_mask,
            "numeric_coord_targets": coord_targets,
            "numeric_coord_mask": coord_mask,
        }

    bucket_patterns = spec.numeric_fields(parsed_answer)

    count_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("count", []))
    for idx, value in enumerate(count_values[:COUNT_TARGET_DIM]):
        count_targets[idx] = float(value) / COUNT_SCALE
        count_mask[idx] = 1.0

    distance_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("distance", []))
    if distance_values:
        motion_targets[0] = float(distance_values[0]) / DISTANCE_SCALE
        motion_mask[0] = 1.0

    speed_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("speed", []))
    if speed_values:
        motion_targets[1] = float(speed_values[0]) / SPEED_SCALE
        motion_mask[1] = 1.0

    acceleration_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("acceleration", []))
    if acceleration_values:
        motion_targets[2] = float(acceleration_values[0]) / ACCELERATION_SCALE
        motion_mask[2] = 1.0

    global_xy_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("global_3d_xy", []))
    for idx, value in enumerate(global_xy_values[:2]):
        coord_targets[idx] = float(value) / GLOBAL_COORD_SCALE
        coord_mask[idx] = 1.0

    image_xy_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("image_2d_xy", []))
    for idx, value in enumerate(image_xy_values[:2]):
        coord_targets[2 + idx] = float(value) / IMAGE_COORD_SCALE
        coord_mask[2 + idx] = 1.0

    waypoint_values = _extract_bucket_values(base_module, parsed_answer, bucket_patterns.get("waypoint_xy", []))
    max_waypoint_values = min(len(waypoint_values), COORD_TARGET_DIM - 4)
    for idx in range(max_waypoint_values):
        coord_targets[4 + idx] = float(waypoint_values[idx]) / WAYPOINT_COORD_SCALE
        coord_mask[4 + idx] = 1.0

    return {
        "numeric_count_targets": count_targets,
        "numeric_count_mask": count_mask,
        "numeric_motion_targets": motion_targets,
        "numeric_motion_mask": motion_mask,
        "numeric_coord_targets": coord_targets,
        "numeric_coord_mask": coord_mask,
    }
