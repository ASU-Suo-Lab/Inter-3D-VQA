from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from opendrivevla_v5.config.common import DEFAULT_VAL_RATIO, DEFAULT_VAL_SCENES, LLM_ROOT, STRICT_CAM_ORDER


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_path(value: Any) -> str:
    normalized = str(value).replace("\\", "/")
    if normalized.startswith("/data/"):
        return str((LLM_ROOT / normalized.lstrip("/")).resolve())
    if normalized.startswith("data/"):
        return str((LLM_ROOT / normalized).resolve())
    return normalized


def normalize_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: normalize_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_paths(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(normalize_paths(value) for value in payload)
    if isinstance(payload, str):
        return normalize_path(payload)
    return payload


def validate_normalized_info(info: Dict[str, Any]) -> Dict[str, Any]:
    token = str(info.get("token"))
    lidar_path = Path(str(info["lidar_path"]))
    ensure(lidar_path.is_file(), f"LiDAR file not found for {token}: {lidar_path}")
    cams = info.get("cams") or {}
    actual_keys = tuple(cams.keys())
    ensure(
        actual_keys == STRICT_CAM_ORDER or set(actual_keys) == set(STRICT_CAM_ORDER),
        f"Unexpected camera keys for {token}: {sorted(actual_keys)}",
    )

    ordered_cams: Dict[str, Any] = {}
    for cam_key in STRICT_CAM_ORDER:
        ensure(cam_key in cams, f"Missing {cam_key} for {token}")
        cam_info = cams[cam_key]
        image_path = Path(str(cam_info["image_paths"]))
        ensure(image_path.is_file(), f"Image file not found for {token} {cam_key}: {image_path}")
        ordered_cams[cam_key] = cam_info

    info["cams"] = ordered_cams
    return info


def resolve_sidecar_images(qa_pair: Dict[str, Any]) -> List[str]:
    image_fields = ("north_image_path", "east_image_path", "south_image_path", "west_image_path")
    images: List[str] = []
    for field_name in image_fields:
        raw_path = qa_pair.get(field_name)
        ensure(raw_path, f"Missing {field_name} in QA pair {qa_pair.get('question_id')}")
        resolved = normalize_path(raw_path)
        ensure(Path(resolved).is_file(), f"Sidecar image not found for {field_name}: {resolved}")
        images.append(resolved)
    return images


def build_scene_split(
    scene_counts: Dict[str, int],
    val_scenes: int = DEFAULT_VAL_SCENES,
    target_ratio: float = DEFAULT_VAL_RATIO,
) -> Tuple[List[str], List[str], int]:
    ranked = sorted(scene_counts.items(), key=lambda item: (-item[1], item[0]))
    ensure(0 < val_scenes < len(ranked), f"val_scenes must be in [1, {len(ranked) - 1}], got {val_scenes}")

    target_qas = round(sum(scene_counts.values()) * target_ratio)
    rank_index = {scene_id: index for index, (scene_id, _) in enumerate(ranked)}
    remaining = list(ranked)
    val_scene_ids: List[str] = []
    current_qas = 0

    for pick_index in range(val_scenes):
        slots_left = val_scenes - pick_index
        if len(remaining) == slots_left:
            for scene_id, count in remaining:
                val_scene_ids.append(scene_id)
                current_qas += count
            remaining.clear()
            break

        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                abs((current_qas + remaining[idx][1]) - target_qas),
                abs(((target_qas - current_qas) / slots_left) - remaining[idx][1]),
                rank_index[remaining[idx][0]],
            ),
        )
        scene_id, count = remaining.pop(best_idx)
        val_scene_ids.append(scene_id)
        current_qas += count

    val_scene_set = set(val_scene_ids)
    train_scene_ids = [scene_id for scene_id, _ in ranked if scene_id not in val_scene_set]
    return train_scene_ids, val_scene_ids, target_qas


def load_prepared_infos(infos_pkl: Path) -> List[Dict[str, Any]]:
    infos = load_pickle(infos_pkl)
    if isinstance(infos, dict) and "infos" in infos:
        infos = infos["infos"]
    ensure(isinstance(infos, list), f"Unsupported infos payload type in {infos_pkl}: {type(infos).__name__}")
    return infos


def count_scene_distribution(qa_pairs: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(str(qa_pair["scene_id"]) for qa_pair in qa_pairs)

