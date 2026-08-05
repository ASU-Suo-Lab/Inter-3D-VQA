from __future__ import annotations

import argparse
from pathlib import Path

from opendrivevla_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_PREPARED_DIR,
    DEFAULT_QA_JSON,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SCENES,
    resolve_dataset_version_paths,
)
from opendrivevla_v5.data.prepare import prepare_intersection_v5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare strict Linux-only OpenDriveVLA V5 artifacts.")
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
    manifest = prepare_intersection_v5(
        dataset_version=str(resolved["dataset_version"]),
        qa_json=Path(resolved["qa_json"]).resolve(),
        info_pkl=Path(args.info_pkl).resolve(),
        output_dir=Path(resolved["prepared_dir"]).resolve(),
        val_scenes=args.val_scenes,
        val_ratio=args.val_ratio,
    )
    counts = manifest["counts"]
    print(
        "[prepare_intersection_v5] "
        f"train_scenes={counts['train_scenes']} val_scenes={counts['val_scenes']} "
        f"train_qas={counts['train_qas']} val_qas={counts['val_qas']} "
        f"train_frames={counts['train_frames']} val_frames={counts['val_frames']} "
        f"dropped_frames={counts['dropped_frames']} dropped_qas={counts['dropped_qas']}"
    )


if __name__ == "__main__":
    main()
