from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from traffixqwen_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_MODEL_NAME_OR_PATH, DEFAULT_VISION_TOWER, resolve_dataset_version_paths
from traffixqwen_v5.data.common import ensure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the TraffiX-Qwen V5 runtime environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_NAME_OR_PATH))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    model_path = Path(args.model_path).resolve()
    vision_tower = Path(args.vision_tower).resolve()

    ensure(model_path.exists(), f"Base model path not found: {model_path}")
    ensure(vision_tower.exists(), f"Vision tower path not found: {vision_tower}")
    ensure(Path(resolved["qa_json"]).is_file(), f"QA JSON not found: {resolved['qa_json']}")
    ensure(Path(resolved["evaluator"]).is_file(), f"Evaluator not found: {resolved['evaluator']}")
    if args.require_prepared:
        manifest_path = Path(resolved["prepared_dir"]) / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    ensure(torch.cuda.is_available(), "CUDA is required for TraffiX-Qwen V5.")
    gpu_count = torch.cuda.device_count()
    ensure(gpu_count == 4, f"TraffiX-Qwen V5 expects 4 GPUs, found {gpu_count}")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "<unknown>")
    print(
        f"[check_env] env={conda_env} dataset_version={resolved['dataset_version']} cuda={torch.version.cuda} "
        f"gpu_count={gpu_count} model_path={model_path} vision_tower={vision_tower}"
    )


if __name__ == "__main__":
    main()
