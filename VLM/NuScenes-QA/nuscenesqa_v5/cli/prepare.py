from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from nuscenesqa_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SCENES,
    resolve_dataset_version_paths,
)
from nuscenesqa_v5.data.common import (
    build_eval_record,
    build_frame_record,
    build_info_lookup,
    build_scene_split,
    build_sidecar_row,
    build_train_record,
    dump_json,
    dump_jsonl,
    ensure,
    load_infos,
    load_source_qa_pairs,
    normalized_images,
    normalized_point_cloud,
    summarize_counts,
)
from nuscenesqa_v5.data.templates import build_answer_label_vocab, supervision_fields


def is_trainable_qa(row: dict) -> bool:
    answer = row.get("answer")
    return (not row.get("placeholder", False)) and answer not in (None, "")


def build_random_scene_split(scene_ids: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ensure(0.0 < train_ratio < 1.0, f"train_ratio must be between 0 and 1, got {train_ratio}")
    shuffled = sorted(scene_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * (1.0 - train_ratio))))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    ensure(val_count > 0, "Need at least one validation scene after splitting.")
    val_scene_ids = shuffled[:val_count]
    train_scene_ids = shuffled[val_count:]
    ensure(train_scene_ids, "Need at least one training scene after splitting.")
    return train_scene_ids, val_scene_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NuScenes-QA intersection artifacts.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--val-scenes", type=int, default=DEFAULT_VAL_SCENES)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-policy", choices=["random_scene", "greedy"], default="random_scene")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        prepared_dir=args.output_dir,
    )
    qa_json = Path(resolved["qa_json"]).resolve()
    info_pkl = Path(args.info_pkl).resolve()
    output_dir = Path(resolved["prepared_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(info_pkl)
    info_lookup = build_info_lookup(infos)
    raw_rows = load_source_qa_pairs(qa_json)

    accessible_rows = []
    dropped_qas = 0
    for row in raw_rows:
        scene_id = str(row["scene_id"])
        frame_token = str(row["frame_token"])
        info = info_lookup.get((scene_id, frame_token))
        if info is None:
            dropped_qas += 1
            continue
        row_copy = dict(row)
        row_copy["images"] = normalized_images(row)
        row_copy["point_cloud_path"] = normalized_point_cloud(row)
        accessible_rows.append(row_copy)

    ensure(accessible_rows, f"No accessible QA rows remained after matching against {info_pkl}")
    trainable_rows = [row for row in accessible_rows if is_trainable_qa(row)]
    filtered_rows = [row for row in accessible_rows if not is_trainable_qa(row)]
    ensure(trainable_rows, "No trainable QA rows remained after filtering placeholder or empty-answer rows.")

    scene_counts = Counter(str(row["scene_id"]) for row in trainable_rows)
    if args.split_policy == "greedy":
        train_scene_ids, val_scene_ids, target_val_qas = build_scene_split(scene_counts, args.val_scenes, args.val_ratio)
    else:
        train_scene_ids, val_scene_ids = build_random_scene_split(
            sorted(scene_counts),
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        target_val_qas = round(len(trainable_rows) * (1.0 - args.train_ratio))
    train_scene_set = set(train_scene_ids)
    val_scene_set = set(val_scene_ids)

    train_rows = [row for row in trainable_rows if str(row["scene_id"]) in train_scene_set]
    val_rows = [row for row in trainable_rows if str(row["scene_id"]) in val_scene_set]
    ensure(train_rows and val_rows, "Train/val split produced an empty partition.")

    def with_supervision(record: dict, row: dict) -> dict:
        record.update(
            supervision_fields(
                record["subtemplate"],
                record.get("structured_targets") or {},
                record["answer"],
                dataset_version=str(resolved["dataset_version"]),
            )
        )
        return record

    train_records = [with_supervision(build_train_record(row, row["point_cloud_path"], row["images"]), row) for row in train_rows]
    val_records = [with_supervision(build_train_record(row, row["point_cloud_path"], row["images"]), row) for row in val_rows]
    val_eval = [with_supervision(build_eval_record(row, row["point_cloud_path"], row["images"]), row) for row in val_rows]
    sidecar_train = [build_sidecar_row(row, row["point_cloud_path"], row["images"], "train") for row in train_rows]
    sidecar_val = [build_sidecar_row(row, row["point_cloud_path"], row["images"], "val") for row in val_rows]

    train_frame_records = {}
    for row in train_rows:
        train_frame_records.setdefault(str(row["frame_token"]), build_frame_record(row, row["point_cloud_path"], row["images"], "train"))
    val_frame_records = {}
    for row in val_rows:
        val_frame_records.setdefault(str(row["frame_token"]), build_frame_record(row, row["point_cloud_path"], row["images"], "val"))

    label_vocab = build_answer_label_vocab(trainable_rows, dataset_version=str(resolved["dataset_version"]))

    dump_json(output_dir / "train.json", train_records)
    dump_json(output_dir / "val.json", val_records)
    dump_json(output_dir / "val_eval.json", val_eval)
    dump_json(output_dir / "frames_train.json", list(train_frame_records.values()))
    dump_json(output_dir / "frames_val.json", list(val_frame_records.values()))
    dump_json(output_dir / "object_label_vocab.json", label_vocab)
    dump_jsonl(output_dir / "sidecar_train.jsonl", sidecar_train)
    dump_jsonl(output_dir / "sidecar_val.jsonl", sidecar_val)

    manifest = {
        "qa_json": str(qa_json),
        "info_pkl": str(info_pkl),
        "output_dir": str(output_dir),
        "dataset_version": str(resolved["dataset_version"]),
        "split_policy": {
            "type": "scene_level_greedy" if args.split_policy == "greedy" else "scene_level_random",
            "val_scenes": args.val_scenes,
            "val_ratio_target": args.val_ratio,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "target_val_qas": target_val_qas,
        },
        "counts": {
            "source_qas": len(raw_rows),
            "source_frames": len({str(row['frame_token']) for row in raw_rows}),
            "source_scenes": len({str(row['scene_id']) for row in raw_rows}),
            "accessible_qas": len(accessible_rows),
            "accessible_frames": len({str(row['frame_token']) for row in accessible_rows}),
            "accessible_scenes": len({str(row['scene_id']) for row in accessible_rows}),
            "trainable_qas": len(trainable_rows),
            "filtered_non_trainable": len(filtered_rows),
            "placeholder_qas": sum(1 for row in filtered_rows if row.get("placeholder", False)),
            "empty_answer_qas": sum(1 for row in filtered_rows if row.get("answer") in (None, "")),
            "train_qas": len(train_rows),
            "val_qas": len(val_rows),
            "train_frames": len(train_frame_records),
            "val_frames": len(val_frame_records),
            "train_scenes": len(train_scene_ids),
            "val_scenes": len(val_scene_ids),
            "dropped_qas": dropped_qas,
        },
        "train_scene_ids": train_scene_ids,
        "val_scene_ids": val_scene_ids,
        "train_summary": summarize_counts(train_rows),
        "val_summary": summarize_counts(val_rows),
        "filtered_counts_by_subtemplate": dict(Counter(str(row.get("subtemplate", "unknown")) for row in filtered_rows)),
        "object_label_vocab": label_vocab,
    }
    dump_json(output_dir / "split_manifest.json", manifest)
    print(
        f"[prepare_nuscenesqa_{resolved['dataset_version']}] "
        f"train_scenes={len(train_scene_ids)} val_scenes={len(val_scene_ids)} "
        f"train_qas={len(train_rows)} val_qas={len(val_rows)} "
        f"train_frames={len(train_frame_records)} val_frames={len(val_frame_records)} "
        f"filtered_non_trainable={len(filtered_rows)} dropped_qas={dropped_qas}"
    )


if __name__ == "__main__":
    main()
