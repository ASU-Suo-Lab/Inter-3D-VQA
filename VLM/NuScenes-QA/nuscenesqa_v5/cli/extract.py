from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from nuscenesqa_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    FEATURE_DEFAULTS,
    MODEL_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from nuscenesqa_v5.data.common import build_info_lookup, load_infos
from nuscenesqa_v5.data.extract import (
    build_frame_features_raw,
    compute_feature_stats,
    normalize_frame_features,
    save_frame_feature,
    write_feature_manifest,
)
from nuscenesqa_v5.data.templates import build_answer_type_lookup
from nuscenesqa_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract repo-local object features for NuScenes-QA.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--object-limit", type=int, default=MODEL_DEFAULTS["object_limit"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    info_pkl = Path(args.info_pkl).resolve()
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    split_manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(split_manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared split_manifest dataset_version={split_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")

    label_vocab = load_json(prepared_dir / "object_label_vocab.json")
    ensure(isinstance(label_vocab, list) and label_vocab, f"Invalid object label vocab in {prepared_dir}")
    train_records = load_json(prepared_dir / "train.json")
    val_records = load_json(prepared_dir / "val.json")
    ensure(isinstance(train_records, list) and train_records, f"Invalid train records in {prepared_dir / 'train.json'}")
    ensure(isinstance(val_records, list) and val_records, f"Invalid val records in {prepared_dir / 'val.json'}")
    answer_type_lookup = build_answer_type_lookup([*train_records, *val_records], dataset_version=str(resolved["dataset_version"]))
    infos = load_infos(info_pkl)
    info_lookup = build_info_lookup(infos)

    example_object_dim = None
    pending: list[tuple[Path, np.ndarray, np.ndarray, int]] = []
    train_feature_rows: list[tuple[np.ndarray, np.ndarray, int]] = []
    for split in ("train", "val"):
        frames = load_json(prepared_dir / f"frames_{split}.json")
        ensure(isinstance(frames, list) and frames, f"No frame records found for split={split}")
        output_dir = worktree["features"] / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(frames, desc=f"Extract {split}"):
            key = (str(row["scene_id"]), str(row["frame_token"]))
            info = info_lookup.get(key)
            ensure(info is not None, f"Missing info for scene={row['scene_id']} frame={row['frame_token']}")
            track_types = answer_type_lookup.get(key, {})
            object_features, bbox_features, valid_count = build_frame_features_raw(
                info,
                label_vocab,
                args.object_limit,
                track_types,
                str(resolved["annotation_key"]),
            )
            if example_object_dim is None:
                example_object_dim = object_features.shape[-1]
            pending.append((output_dir / f"{row['frame_token']}.npz", object_features, bbox_features, valid_count))
            if split == "train":
                train_feature_rows.append((object_features, bbox_features, valid_count))

    ensure(example_object_dim is not None, "Feature extraction did not produce any feature tensors.")
    feature_stats = compute_feature_stats(train_feature_rows)
    for output_path, object_features, bbox_features, valid_count in pending:
        object_features_norm, bbox_features_norm = normalize_frame_features(object_features, bbox_features, valid_count, feature_stats)
        save_frame_feature(output_path, object_features_norm, bbox_features_norm)
    write_feature_manifest(
        worktree["feature_manifest"],
        dataset_version=str(resolved["dataset_version"]),
        label_vocab=label_vocab,
        object_limit=args.object_limit,
        object_feature_dim=example_object_dim,
        bbox_feature_dim=FEATURE_DEFAULTS["bbox_feature_dim"],
        feature_stats=feature_stats,
    )
    print(
        f"[extract_nuscenesqa_{resolved['dataset_version']}] "
        f"features_dir={worktree['features']} object_feature_dim={example_object_dim} "
        f"bbox_feature_dim={FEATURE_DEFAULTS['bbox_feature_dim']}"
    )


if __name__ == "__main__":
    main()
