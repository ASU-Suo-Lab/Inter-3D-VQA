from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from nuscenesqa_v5.config.common import DEFAULT_CAMERA_ORDER
from nuscenesqa_v5.utils.io import dump_json, dump_jsonl, ensure, load_json, load_pickle, normalize_data_path

REQUIRED_QA_KEYS = {
    "question_id",
    "scene_id",
    "frame_token",
    "question",
    "answer",
    "subtemplate",
    "chapter",
    "section",
    "north_image_path",
    "south_image_path",
    "east_image_path",
    "west_image_path",
    "point_cloud_path",
}


def load_source_qa_pairs(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    qa_pairs = payload.get("qa_pairs") if isinstance(payload, dict) else payload
    ensure(isinstance(qa_pairs, list) and qa_pairs, f"{path} does not contain a non-empty QA list.")
    missing = sorted(REQUIRED_QA_KEYS - set(qa_pairs[0].keys()))
    ensure(not missing, f"QA rows are missing required keys: {missing}")
    return qa_pairs


def load_infos(path: Path) -> list[dict[str, Any]]:
    payload = load_pickle(path)
    infos = payload["infos"] if isinstance(payload, dict) and "infos" in payload else payload
    ensure(isinstance(infos, list) and infos, f"Unexpected infos payload in {path}")
    return infos


def build_info_lookup(infos: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for info in infos:
        key = (str(info["scene_id"]), str(info["token"]))
        lookup[key] = dict(info)
    return lookup


def build_scene_split(scene_counts: Mapping[str, int], val_scenes: int, target_ratio: float) -> Tuple[list[str], list[str], int]:
    ranked = sorted(scene_counts.items(), key=lambda item: (-item[1], item[0]))
    ensure(0 < val_scenes < len(ranked), f"val_scenes must be in [1, {len(ranked) - 1}], got {val_scenes}")
    target_qas = round(sum(scene_counts.values()) * target_ratio)
    rank_index = {scene_id: index for index, (scene_id, _) in enumerate(ranked)}
    remaining = list(ranked)
    val_scene_ids: list[str] = []
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


def normalized_images(record: Mapping[str, Any]) -> list[str]:
    image_paths = [normalize_data_path(str(record[f"{view}_image_path"])) for view in DEFAULT_CAMERA_ORDER]
    for path in image_paths:
        ensure(Path(path).is_file(), f"Image file not found: {path}")
    return image_paths


def normalized_point_cloud(record: Mapping[str, Any]) -> str:
    path = normalize_data_path(str(record["point_cloud_path"]))
    ensure(Path(path).is_file(), f"Point cloud file not found: {path}")
    return path


def build_train_record(record: Mapping[str, Any], point_cloud_path: str, images: Sequence[str]) -> dict[str, Any]:
    return {
        "question_id": str(record["question_id"]),
        "scene_id": str(record["scene_id"]),
        "frame_token": str(record["frame_token"]),
        "question": str(record["question"]).strip(),
        "answer": str(record["answer"]).strip(),
        "chapter": str(record["chapter"]),
        "section": str(record["section"]),
        "subtemplate": str(record["subtemplate"]),
        "point_cloud_path": point_cloud_path,
        "images": list(images),
        "structured_targets": record.get("structured_targets", {}),
    }


def build_eval_record(record: Mapping[str, Any], point_cloud_path: str, images: Sequence[str]) -> dict[str, Any]:
    payload = build_train_record(record, point_cloud_path, images)
    payload["id"] = payload["question_id"]
    return payload


def build_sidecar_row(record: Mapping[str, Any], point_cloud_path: str, images: Sequence[str], split: str) -> dict[str, Any]:
    return {
        "question_id": str(record["question_id"]),
        "scene_id": str(record["scene_id"]),
        "frame_token": str(record["frame_token"]),
        "question": str(record["question"]).strip(),
        "answer": str(record["answer"]).strip(),
        "chapter": str(record["chapter"]),
        "section": str(record["section"]),
        "subtemplate": str(record["subtemplate"]),
        "structured_targets": record.get("structured_targets", {}),
        "point_cloud_path": point_cloud_path,
        "images": list(images),
        "split": split,
    }


def build_frame_record(record: Mapping[str, Any], point_cloud_path: str, images: Sequence[str], split: str) -> dict[str, Any]:
    return {
        "scene_id": str(record["scene_id"]),
        "frame_token": str(record["frame_token"]),
        "point_cloud_path": point_cloud_path,
        "images": list(images),
        "split": split,
    }


def summarize_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "qas": len(rows),
        "frames": len({str(row["frame_token"]) for row in rows}),
        "scenes": len({str(row["scene_id"]) for row in rows}),
    }


def build_object_label_vocab(infos: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = sorted({str(label) for info in infos for label in info.get("gt_names", [])})
    ensure(labels, "No object labels found in matched infos.")
    return labels

