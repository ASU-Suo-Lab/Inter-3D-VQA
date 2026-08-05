from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from traffixqwen_v5.config.common import DEFAULT_DATASET_VERSION, resolve_dataset_version_paths
from traffixqwen_v5.data.common import (
    DEFAULT_INFO_PKL,
    DEFAULT_PREPARED_DIR,
    DEFAULT_QA_JSON,
    DEFAULT_TEMPORAL_WINDOW,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SCENES,
    DEFAULT_VIEWS,
    build_image_sequence,
    build_system_prompt,
    build_user_prompt,
    build_scene_split,
    build_sidecar_rows,
    build_info_lookup,
    dump_json,
    dump_jsonl,
    ensure,
    load_infos_by_scene,
    load_source_qa_pairs,
    normalize_data_path,
    summarize_split_counts,
    validate_subtemplate_registry,
    validate_anchor_images,
    validate_views,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare strict TraffiX-Qwen Intersection V5 3-frame 4-view artifacts.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temporal-window", type=int, default=DEFAULT_TEMPORAL_WINDOW)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--val-scenes", type=int, default=DEFAULT_VAL_SCENES)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    return parser.parse_args()


def dataset_name(dataset_version: str, temporal_window: int) -> str:
    return f"intersectionqa_{dataset_version}_multiview_{temporal_window}frame"


def build_sample(
    qa_pair: Dict,
    scene_infos: List[Dict],
    dataset_version: str,
    temporal_window: int,
    views: List[str],
) -> Dict:
    anchor_images = validate_anchor_images(qa_pair, views)
    context_frame_tokens, image_paths = build_image_sequence(
        scene_infos=scene_infos,
        frame_token=str(qa_pair["frame_token"]),
        views=views,
        temporal_window=temporal_window,
    )
    midpoint = temporal_window // 2
    anchor_image_paths = image_paths[midpoint * len(views) : (midpoint + 1) * len(views)]
    for view_index, view in enumerate(views):
        ensure(
            anchor_images[view] == anchor_image_paths[view_index],
            f"Anchor image mismatch for question_id={qa_pair['question_id']} view={view}",
        )

    answer = str(qa_pair["answer"]).strip()
    chapter = str(qa_pair["chapter"])
    section = str(qa_pair["section"])
    subtemplate = str(qa_pair["subtemplate"])
    scene_id = str(qa_pair["scene_id"])
    frame_token = str(qa_pair["frame_token"])
    question = str(qa_pair["question"]).strip()
    system_prompt = build_system_prompt(views, temporal_window)
    return {
        "id": str(qa_pair["question_id"]),
        "sample_id": str(qa_pair["question_id"]),
        "dataset": dataset_name(dataset_version, temporal_window),
        "type": subtemplate,
        "chapter": chapter,
        "section": section,
        "subtemplate": subtemplate,
        "scene_id": scene_id,
        "frame_token": frame_token,
        "context_frame_tokens": context_frame_tokens,
        "point_cloud_path": normalize_data_path(str(qa_pair["point_cloud_path"])) if qa_pair.get("point_cloud_path") else None,
        "structured_targets": qa_pair.get("structured_targets", {}),
        "metadata": {
            "dataset": dataset_name(dataset_version, temporal_window),
            "dataset_version": dataset_version,
            "chapter": chapter,
            "section": section,
            "subtemplate": subtemplate,
            "scene_id": scene_id,
            "frame_token": frame_token,
            "context_frame_tokens": context_frame_tokens,
            "views": list(views),
            "temporal_window": temporal_window,
            "context_strategy": "clamp_to_scene_bounds",
            "prompt_style": "sample_system_plus_subtemplate_instruction",
        },
        "image": image_paths,
        "system": system_prompt,
        "conversations": [
            {
                "from": "human",
                "value": build_user_prompt(
                    question=question,
                    chapter=chapter,
                    section=section,
                    subtemplate=subtemplate,
                    dataset_version=dataset_version,
                    views=views,
                    temporal_window=temporal_window,
                ),
            },
            {
                "from": "gpt",
                "value": answer,
            },
        ],
        "question": question,
        "answer": answer,
    }


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, qa_json=args.qa_json, prepared_dir=args.output_dir)
    qa_json = Path(resolved["qa_json"]).resolve()
    info_pkl = Path(args.info_pkl).resolve()
    output_dir = Path(resolved["prepared_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure(args.temporal_window == DEFAULT_TEMPORAL_WINDOW, f"TraffiX-Qwen V5 only supports temporal_window={DEFAULT_TEMPORAL_WINDOW}")
    views = list(validate_views(args.views))

    qa_pairs = load_source_qa_pairs(qa_json)
    validate_subtemplate_registry({str(qa_pair["subtemplate"]) for qa_pair in qa_pairs}, str(resolved["dataset_version"]))
    scene_to_infos = load_infos_by_scene(info_pkl)
    info_lookup = build_info_lookup(scene_to_infos)

    converted_samples: List[Dict] = []
    for qa_pair in qa_pairs:
        scene_id = str(qa_pair["scene_id"])
        frame_token = str(qa_pair["frame_token"])
        ensure(scene_id in scene_to_infos, f"scene_id {scene_id} from question_id={qa_pair['question_id']} is missing in {info_pkl}")
        ensure((scene_id, frame_token) in info_lookup, f"frame_token {frame_token} from question_id={qa_pair['question_id']} is missing in {info_pkl}")
        converted_samples.append(build_sample(qa_pair, scene_to_infos[scene_id], str(resolved["dataset_version"]), args.temporal_window, views))

    scene_counts = Counter(str(qa_pair["scene_id"]) for qa_pair in qa_pairs)
    train_scene_ids, val_scene_ids, target_val_qas = build_scene_split(
        scene_counts=scene_counts,
        val_scenes=args.val_scenes,
        target_ratio=args.val_ratio,
    )
    scene_to_split = {
        **{scene_id: "train" for scene_id in train_scene_ids},
        **{scene_id: "val" for scene_id in val_scene_ids},
    }
    train_scene_set = set(train_scene_ids)
    val_scene_set = set(val_scene_ids)

    train_samples = [sample for sample in converted_samples if sample["scene_id"] in train_scene_set]
    val_samples = [sample for sample in converted_samples if sample["scene_id"] in val_scene_set]
    sidecar_all, sidecar_train, sidecar_val = build_sidecar_rows(qa_pairs, scene_to_split)

    manifest = {
        "dataset_version": str(resolved["dataset_version"]),
        "qa_json": str(qa_json),
        "info_pkl": str(info_pkl),
        "output_dir": str(output_dir),
        "dataset_name": dataset_name(str(resolved["dataset_version"]), args.temporal_window),
        "temporal_window": args.temporal_window,
        "views": views,
        "context_strategy": "clamp_to_scene_bounds",
        "split_policy": {
            "type": "scene_level_greedy",
            "val_scenes": args.val_scenes,
            "val_ratio_target": args.val_ratio,
            "target_val_qas": target_val_qas,
        },
        "counts": {
            "source_qas": len(qa_pairs),
            "source_frames": len({str(qa_pair["frame_token"]) for qa_pair in qa_pairs}),
            "source_scenes": len({str(qa_pair["scene_id"]) for qa_pair in qa_pairs}),
            "train_qas": len(train_samples),
            "val_qas": len(val_samples),
            "train_frames": len({sample["frame_token"] for sample in train_samples}),
            "val_frames": len({sample["frame_token"] for sample in val_samples}),
            "train_scenes": len(train_scene_ids),
            "val_scenes": len(val_scene_ids),
            "dropped_qas": 0,
            "dropped_frames": 0,
        },
        "train_scene_ids": train_scene_ids,
        "val_scene_ids": val_scene_ids,
        "train_split_summary": summarize_split_counts(train_samples),
        "val_split_summary": summarize_split_counts(val_samples),
    }

    dump_json(output_dir / "train.json", train_samples)
    dump_json(output_dir / "val.json", val_samples)
    dump_jsonl(output_dir / "intersection_vqa_eval_sidecar.jsonl", sidecar_all)
    dump_jsonl(output_dir / "sidecar_train.jsonl", sidecar_train)
    dump_jsonl(output_dir / "sidecar_val.jsonl", sidecar_val)
    dump_json(output_dir / "split_manifest.json", manifest)

    print(
        "[prepare_traffixqwen_v5] "
        f"train_scenes={len(train_scene_ids)} val_scenes={len(val_scene_ids)} "
        f"train_qas={len(train_samples)} val_qas={len(val_samples)} "
        f"train_frames={len({sample['frame_token'] for sample in train_samples})} "
        f"val_frames={len({sample['frame_token'] for sample in val_samples})} "
        "dropped_frames=0 dropped_qas=0"
    )


if __name__ == "__main__":
    main()
