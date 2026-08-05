from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from bevllm_v5.config.common import DEFAULT_VIEW_ID
from bevllm_v5.utils.io import dump_json, dump_jsonl, dump_pickle, ensure, load_json, load_pickle, normalize_data_path


REQUIRED_QA_KEYS = {
    "question_id",
    "scene_id",
    "frame_token",
    "question",
    "answer",
    "chapter",
    "section",
    "subtemplate",
    "north_image_path",
    "south_image_path",
    "east_image_path",
    "west_image_path",
    "point_cloud_path",
}

CAMERA_ORDER = ("north", "south", "east", "west")


def load_source_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    qa_pairs = payload.get("qa_pairs") if isinstance(payload, dict) else payload
    ensure(isinstance(qa_pairs, list) and qa_pairs, f"{path} does not contain a non-empty QA list.")
    missing = sorted(REQUIRED_QA_KEYS - set(qa_pairs[0].keys()))
    ensure(not missing, f"QA rows are missing required keys: {missing}")
    return qa_pairs


def load_infos(path: Path) -> List[Dict[str, Any]]:
    payload = load_pickle(path)
    infos = payload["infos"] if isinstance(payload, dict) and "infos" in payload else payload
    ensure(isinstance(infos, list) and infos, f"Unexpected infos payload in {path}")
    return infos


def build_info_lookup(infos: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(str(info["scene_id"]), str(info["token"])): dict(info) for info in infos}


def build_scene_split(scene_counts: Mapping[str, int], val_scenes: int, target_ratio: float) -> Tuple[List[str], List[str], int]:
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


def normalized_images(record: Mapping[str, Any]) -> List[str]:
    image_paths = [normalize_data_path(str(record[f"{camera}_image_path"])) for camera in CAMERA_ORDER]
    for image_path in image_paths:
        ensure(Path(image_path).is_file(), f"Image file not found: {image_path}")
    return image_paths


def build_qa_record(record: Mapping[str, Any], images: Sequence[str]) -> Dict[str, Any]:
    return {
        "id": str(record["question_id"]),
        "question_id": str(record["question_id"]),
        "scene_id": str(record["scene_id"]),
        "frame_token": str(record["frame_token"]),
        "question": str(record["question"]).strip(),
        "answer": str(record["answer"]).strip(),
        "chapter": str(record["chapter"]),
        "section": str(record["section"]),
        "subtemplate": str(record["subtemplate"]),
        "point_cloud_path": normalize_data_path(str(record["point_cloud_path"])),
        "images": list(images),
        "view": int(record.get("view", DEFAULT_VIEW_ID)),
        "structured_targets": record.get("structured_targets", {}),
        "metadata": {
            "chapter": str(record["chapter"]),
            "section": str(record["section"]),
            "subtemplate": str(record["subtemplate"]),
        },
    }


def build_sidecar_row(record: Mapping[str, Any], images: Sequence[str], split: str) -> Dict[str, Any]:
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
        "images": list(images),
        "split": split,
    }


def build_frame_records(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_frame: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["frame_token"]))
        if key in by_frame:
            continue
        images = row["images"] if "images" in row else normalized_images(row)
        by_frame[key] = {
            "scene_id": str(row["scene_id"]),
            "frame_token": str(row["frame_token"]),
            "point_cloud_path": normalize_data_path(str(row["point_cloud_path"])),
            "images": list(images),
            "view": int(row.get("view", DEFAULT_VIEW_ID)),
        }
    return [by_frame[key] for key in sorted(by_frame)]


def summarize_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "qas": len(rows),
        "frames": len({str(row["frame_token"]) for row in rows}),
        "scenes": len({str(row["scene_id"]) for row in rows}),
    }


__all__ = [
    "CAMERA_ORDER",
    "Counter",
    "build_frame_records",
    "build_info_lookup",
    "build_qa_record",
    "build_scene_split",
    "build_sidecar_row",
    "dump_json",
    "dump_jsonl",
    "dump_pickle",
    "ensure",
    "load_infos",
    "load_source_qa_pairs",
    "normalized_images",
    "summarize_counts",
]
