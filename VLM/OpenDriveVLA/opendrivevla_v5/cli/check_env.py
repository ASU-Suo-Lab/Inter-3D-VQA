from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openDriveVLA")

import torch

from opendrivevla_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_ENV_NAME, DEFAULT_INFO_PKL, DEFAULT_MODEL_PATH, default_env_file, default_env_setup_script, resolve_dataset_version_paths
from opendrivevla_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the OpenDriveVLA V5 runtime environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--infos-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-evaluate", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def require_module(module_name: str) -> None:
    importlib.import_module(module_name)


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    ensure(default_env_file().is_file(), f"Environment file not found: {default_env_file()}")
    ensure(default_env_setup_script().is_file(), f"Setup script not found: {default_env_setup_script()}")
    ensure(Path(args.model_path).resolve().is_dir(), f"Model path not found: {args.model_path}")
    ensure(Path(resolved["qa_json"]).resolve().is_file(), f"QA JSON not found: {resolved['qa_json']}")
    ensure(Path(args.infos_pkl).resolve().is_file(), f"Infos PKL not found: {args.infos_pkl}")

    require_module("transformers")
    require_module("peft")
    require_module("datasets")
    require_module("sentence_transformers")
    require_module("bert_score")
    require_module("matplotlib")
    require_module("llava")
    require_module("mmcv")
    require_module("mmdet3d")
    if args.require_evaluate:
        ensure(Path(resolved["evaluator"]).resolve().is_file(), f"Evaluator not found: {resolved['evaluator']}")
    if args.require_prepared:
        manifest_path = Path(resolved["prepared_dir"]) / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )

    ensure(torch.cuda.is_available(), "CUDA is not available.")
    ensure(torch.cuda.device_count() >= args.expected_gpus, f"Expected at least {args.expected_gpus} GPUs, got {torch.cuda.device_count()}.")

    print(
        "[check_env] "
        f"env={DEFAULT_ENV_NAME} dataset_version={resolved['dataset_version']} "
        f"cuda={torch.version.cuda} gpu_count={torch.cuda.device_count()} "
        f"model_path={Path(args.model_path).resolve()}"
    )


if __name__ == "__main__":
    main()
