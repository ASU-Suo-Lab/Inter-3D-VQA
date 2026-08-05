from __future__ import annotations

import argparse
from pathlib import Path

from gpt54_vlm.config.common import DEFAULT_DATASET_VERSION, FORWARD_DEFAULTS, resolve_dataset_version_paths
from gpt54_vlm.utils.io import ensure, load_json
from gpt54_vlm.utils.secrets import resolve_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPT-5.1 intersection inference prerequisites.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--sample-count", type=int, default=FORWARD_DEFAULTS["check_image_count"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
        evaluator=args.evaluator,
    )
    resolve_api_key()
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    evaluator = Path(resolved["evaluator"]).resolve()
    manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(isinstance(manifest, dict), "split_manifest.json must be a JSON object.")
    ensure(manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    ensure(evaluator.is_file(), f"Missing evaluator: {evaluator}")
    records = load_json(prepared_dir / "val_eval.json")
    ensure(isinstance(records, list) and records, f"No evaluation rows found in {prepared_dir / 'val_eval.json'}")
    sample_count = min(int(args.sample_count), len(records))
    for row in records[:sample_count]:
        ensure(isinstance(row, dict), "Each val_eval row must be an object.")
        images = row.get("images")
        ensure(isinstance(images, list) and len(images) == 4, f"Each row must have exactly 4 images: {row.get('question_id')}")
        for image in images:
            ensure(Path(str(image)).is_file(), f"Missing image: {image}")


if __name__ == "__main__":
    main()
