from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from nuscenesqa_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    resolve_dataset_version_paths,
    worktree_paths,
)
from nuscenesqa_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the NuScenes-QA environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--require-evaluate", action="store_true")
    return parser.parse_args()


def require_module(module_name: str) -> None:
    importlib.import_module(module_name)


def parse_os_release() -> dict[str, str]:
    payload = Path("/etc/os-release").read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def require_ubuntu_2204() -> None:
    values = parse_os_release()
    ensure(values.get("ID", "").lower() == "ubuntu", "Ubuntu is required.")
    ensure(values.get("VERSION_ID") == "22.04", "Ubuntu 22.04 is required.")


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    require_ubuntu_2204()
    for module_name in ("numpy", "torch", "matplotlib", "yaml", "tqdm"):
        require_module(module_name)
    if args.require_evaluate:
        for module_name in ("transformers", "bert_score", "sentence_transformers", "pandas", "sklearn"):
            require_module(module_name)

    ensure(torch.cuda.is_available(), "CUDA is required.")
    ensure(torch.cuda.device_count() == args.expected_gpus, f"Expected {args.expected_gpus} GPUs, found {torch.cuda.device_count()}.")
    ensure(Path(resolved["qa_json"]).resolve().is_file(), f"QA json not found: {resolved['qa_json']}")
    ensure(Path(args.info_pkl).resolve().is_file(), f"Info pickle not found: {args.info_pkl}")
    ensure(Path(resolved["evaluator"]).resolve().is_file(), f"Evaluator not found: {resolved['evaluator']}")

    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    if args.require_prepared:
        for name in ("train.json", "val.json", "val_eval.json", "frames_train.json", "frames_val.json", "sidecar_val.jsonl", "split_manifest.json", "object_label_vocab.json"):
            ensure((prepared_dir / name).is_file(), f"Prepared artifact not found: {prepared_dir / name}")
        train_rows = load_json(prepared_dir / "train.json")
        ensure(isinstance(train_rows, list) and train_rows, f"Prepared train rows are invalid: {prepared_dir / 'train.json'}")
        ensure("supervision_answer" in train_rows[0], "Prepared train rows are missing supervision_answer. Re-run prepare.")
        ensure("decoder_prefix" in train_rows[0], "Prepared train rows are missing decoder_prefix. Re-run prepare.")
        label_vocab = load_json(prepared_dir / "object_label_vocab.json")
        ensure("unknown object" in label_vocab, "Prepared label vocab is stale. Re-run prepare.")
        manifest = load_json(prepared_dir / "split_manifest.json")
        ensure(manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared split_manifest dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")

    if args.require_features:
        worktree = worktree_paths(Path(resolved["work_dir"]).resolve())
        ensure(worktree["feature_manifest"].is_file(), f"Feature manifest not found: {worktree['feature_manifest']}")
        ensure((worktree["features"] / "train").is_dir(), f"Train feature dir not found: {worktree['features'] / 'train'}")
        ensure((worktree["features"] / "val").is_dir(), f"Val feature dir not found: {worktree['features'] / 'val'}")
        feature_manifest = load_json(worktree["feature_manifest"])
        ensure("continuous_mean" in feature_manifest and "bbox_mean" in feature_manifest, "Feature manifest is stale. Re-run extract.")
        ensure(feature_manifest.get("dataset_version") == resolved["dataset_version"], f"Feature manifest dataset_version={feature_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")

    print(
        f"[check_env_nuscenesqa_{resolved['dataset_version']}] "
        f"cuda={torch.version.cuda} gpu_count={torch.cuda.device_count()} "
        f"prepared={'yes' if args.require_prepared else 'skip'} "
        f"features={'yes' if args.require_features else 'skip'} "
        f"evaluate={'yes' if args.require_evaluate else 'skip'}"
    )


if __name__ == "__main__":
    main()
