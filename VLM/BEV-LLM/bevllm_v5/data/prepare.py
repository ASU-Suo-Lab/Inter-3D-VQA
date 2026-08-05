from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

from bevllm_v5.data.common import (
    build_frame_records,
    build_info_lookup,
    build_qa_record,
    build_scene_split,
    build_sidecar_row,
    dump_json,
    dump_jsonl,
    dump_pickle,
    ensure,
    load_infos,
    load_source_qa_pairs,
    normalized_images,
    summarize_counts,
)


def prepare_intersection_v5(
    *,
    dataset_version: str,
    qa_json: Path,
    info_pkl: Path,
    output_dir: Path,
    val_scenes: int,
    val_ratio: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(info_pkl)
    info_lookup = build_info_lookup(infos)
    raw_rows = load_source_qa_pairs(qa_json)

    accessible_rows = []
    dropped_qas = 0
    for row in raw_rows:
        scene_id = str(row["scene_id"])
        frame_token = str(row["frame_token"])
        if (scene_id, frame_token) not in info_lookup:
            dropped_qas += 1
            continue
        row_copy = dict(row)
        row_copy["images"] = normalized_images(row)
        accessible_rows.append(row_copy)

    ensure(accessible_rows, f"No accessible QA rows remained after matching against {info_pkl}")

    scene_counts = Counter(str(row["scene_id"]) for row in accessible_rows)
    train_scene_ids, val_scene_ids, target_val_qas = build_scene_split(scene_counts, val_scenes, val_ratio)
    train_scene_set = set(train_scene_ids)
    val_scene_set = set(val_scene_ids)

    train_rows = [row for row in accessible_rows if str(row["scene_id"]) in train_scene_set]
    val_rows = [row for row in accessible_rows if str(row["scene_id"]) in val_scene_set]
    ensure(train_rows and val_rows, "Train/val split produced an empty partition.")

    train_qas = [build_qa_record(row, row["images"]) for row in train_rows]
    val_qas = [build_qa_record(row, row["images"]) for row in val_rows]
    train_frames = build_frame_records(train_rows)
    val_frames = build_frame_records(val_rows)
    sidecar_train = [build_sidecar_row(row, row["images"], "train") for row in train_rows]
    sidecar_val = [build_sidecar_row(row, row["images"], "val") for row in val_rows]

    train_frame_keys = {(row["scene_id"], row["frame_token"]) for row in train_frames}
    val_frame_keys = {(row["scene_id"], row["frame_token"]) for row in val_frames}
    infos_train = [info_lookup[key] for key in sorted(train_frame_keys)]
    infos_val = [info_lookup[key] for key in sorted(val_frame_keys)]

    dump_json(output_dir / "train.json", train_qas)
    dump_json(output_dir / "val.json", val_qas)
    dump_json(output_dir / "frames_train.json", train_frames)
    dump_json(output_dir / "frames_val.json", val_frames)
    dump_jsonl(output_dir / "sidecar_train.jsonl", sidecar_train)
    dump_jsonl(output_dir / "sidecar_val.jsonl", sidecar_val)
    dump_jsonl(output_dir / "intersection_vqa_eval_sidecar.jsonl", [*sidecar_train, *sidecar_val])
    dump_pickle(output_dir / "infos_train.pkl", {"infos": infos_train})
    dump_pickle(output_dir / "infos_val.pkl", {"infos": infos_val})

    manifest = {
        "dataset_version": dataset_version,
        "qa_json": str(qa_json),
        "info_pkl": str(info_pkl),
        "output_dir": str(output_dir),
        "split_policy": {
            "type": "scene_level_greedy",
            "val_scenes": val_scenes,
            "val_ratio_target": val_ratio,
            "target_val_qas": target_val_qas,
        },
        "counts": {
            "source_qas": len(raw_rows),
            "source_frames": len({str(row['frame_token']) for row in raw_rows}),
            "source_scenes": len({str(row['scene_id']) for row in raw_rows}),
            "accessible_qas": len(accessible_rows),
            "accessible_frames": len({str(row['frame_token']) for row in accessible_rows}),
            "accessible_scenes": len({str(row['scene_id']) for row in accessible_rows}),
            "train_qas": len(train_rows),
            "val_qas": len(val_rows),
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "train_scenes": len(train_scene_ids),
            "val_scenes": len(val_scene_ids),
            "dropped_qas": dropped_qas,
            "dropped_frames": len({str(row['frame_token']) for row in raw_rows}) - len({str(row['frame_token']) for row in accessible_rows}),
        },
        "train_scene_ids": train_scene_ids,
        "val_scene_ids": val_scene_ids,
        "train_summary": summarize_counts(train_rows),
        "val_summary": summarize_counts(val_rows),
    }
    dump_json(output_dir / "split_manifest.json", manifest)
    return manifest
