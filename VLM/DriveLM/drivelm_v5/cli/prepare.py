from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from drivelm_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_PREPARED_DIR,
    DEFAULT_QA_JSON,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SCENES,
    resolve_dataset_version_paths,
)
from drivelm_v5.data.common import (
    build_eval_record,
    build_info_lookup,
    build_llama_record,
    build_scene_split,
    build_sidecar_row,
    dump_json,
    dump_jsonl,
    ensure,
    load_infos,
    load_source_qa_pairs,
    normalized_images,
    summarize_counts,
    write_yaml_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DriveLM Intersection V5 artifacts.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--val-scenes", type=int, default=DEFAULT_VAL_SCENES)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, qa_json=args.qa_json, prepared_dir=args.output_dir)
    qa_json = Path(resolved["qa_json"]).resolve()
    info_pkl = Path(args.info_pkl).resolve()
    output_dir = Path(resolved["prepared_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(info_pkl)
    info_lookup = build_info_lookup(infos)
    raw_rows = load_source_qa_pairs(qa_json)

    accessible_rows = []
    dropped_rows = 0
    for row in raw_rows:
        scene_id = str(row["scene_id"])
        frame_token = str(row["frame_token"])
        if (scene_id, frame_token) not in info_lookup:
            dropped_rows += 1
            continue
        images = normalized_images(row)
        row_copy = dict(row)
        row_copy["images"] = images
        accessible_rows.append(row_copy)

    ensure(accessible_rows, f"No accessible QA rows remained after matching against {info_pkl}")

    scene_counts = Counter(str(row["scene_id"]) for row in accessible_rows)
    train_scene_ids, val_scene_ids, target_val_qas = build_scene_split(scene_counts, args.val_scenes, args.val_ratio)
    train_scene_set = set(train_scene_ids)
    val_scene_set = set(val_scene_ids)

    train_rows = [row for row in accessible_rows if str(row["scene_id"]) in train_scene_set]
    val_rows = [row for row in accessible_rows if str(row["scene_id"]) in val_scene_set]
    ensure(train_rows and val_rows, "Train/val split produced an empty partition.")

    train_llama = [build_llama_record(row, row["images"]) for row in train_rows]
    val_llama = [build_llama_record(row, row["images"]) for row in val_rows]
    train_eval = [build_eval_record(row, row["images"]) for row in train_rows]
    val_eval = [build_eval_record(row, row["images"]) for row in val_rows]
    sidecar_train = [build_sidecar_row(row, row["images"], "train") for row in train_rows]
    sidecar_val = [build_sidecar_row(row, row["images"], "val") for row in val_rows]

    dump_json(output_dir / "train_llama.json", train_llama)
    dump_json(output_dir / "val_llama.json", val_llama)
    dump_json(output_dir / "train_eval.json", train_eval)
    dump_json(output_dir / "val_eval.json", val_eval)
    dump_jsonl(output_dir / "sidecar_train.jsonl", sidecar_train)
    dump_jsonl(output_dir / "sidecar_val.jsonl", sidecar_val)
    dump_jsonl(output_dir / "intersection_vqa_eval_sidecar.jsonl", [*sidecar_train, *sidecar_val])
    write_yaml_meta(output_dir / "train_data_config.yaml", output_dir / "train_llama.json")
    write_yaml_meta(output_dir / "val_data_config.yaml", output_dir / "val_llama.json")

    manifest = {
        "dataset_version": str(resolved["dataset_version"]),
        "qa_json": str(qa_json),
        "info_pkl": str(info_pkl),
        "output_dir": str(output_dir),
        "split_policy": {
            "type": "scene_level_greedy",
            "val_scenes": args.val_scenes,
            "val_ratio_target": args.val_ratio,
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
            "train_frames": len({str(row['frame_token']) for row in train_rows}),
            "val_frames": len({str(row['frame_token']) for row in val_rows}),
            "train_scenes": len(train_scene_ids),
            "val_scenes": len(val_scene_ids),
            "dropped_qas": dropped_rows,
            "dropped_frames": len({str(row['frame_token']) for row in raw_rows}) - len({str(row['frame_token']) for row in accessible_rows}),
        },
        "train_scene_ids": train_scene_ids,
        "val_scene_ids": val_scene_ids,
        "train_summary": summarize_counts(train_rows),
        "val_summary": summarize_counts(val_rows),
    }
    dump_json(output_dir / "split_manifest.json", manifest)

    print(
        "[prepare_drivelm_v5] "
        f"train_scenes={len(train_scene_ids)} val_scenes={len(val_scene_ids)} "
        f"train_qas={len(train_rows)} val_qas={len(val_rows)} "
        f"train_frames={len({str(row['frame_token']) for row in train_rows})} "
        f"val_frames={len({str(row['frame_token']) for row in val_rows})} "
        f"dropped_frames={manifest['counts']['dropped_frames']} dropped_qas={dropped_rows}"
    )


if __name__ == "__main__":
    main()
