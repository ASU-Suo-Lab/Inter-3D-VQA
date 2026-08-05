from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import torch

from drivelm_v5.config.common import (
    DEFAULT_ADAPTER_PRETRAIN,
    DEFAULT_DATASET_VERSION,
    DEFAULT_ENV_NAME,
    DEFAULT_LLAMAA_DIR,
    default_env_file,
    default_env_setup_script,
    resolve_dataset_version_paths,
)
from drivelm_v5.utils.imports import add_llama_adapter_to_path
from drivelm_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the DriveLM V5 runtime environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--llama-dir", default=os.environ.get("DRIVELM_LLAMA_DIR", str(DEFAULT_LLAMAA_DIR)))
    parser.add_argument("--adapter-checkpoint", default=str(DEFAULT_ADAPTER_PRETRAIN))
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-evaluate", action="store_true")
    return parser.parse_args()


def require_module(module_name: str) -> None:
    importlib.import_module(module_name)


def ensure_llama_dir(path: Path) -> None:
    ensure(path.is_dir(), f"LLaMA base directory not found: {path}")
    ensure((path / "tokenizer.model").is_file(), f"Missing tokenizer.model under {path}")
    ensure((path / "7B").is_dir(), f"Missing 7B directory under {path}")
    ensure((path / "7B" / "params.json").is_file(), f"Missing params.json under {path / '7B'}")
    ensure(list((path / "7B").glob("*.pth")), f"No LLaMA checkpoint shards found under {path / '7B'}")


def main() -> None:
    args = parse_args()
    add_llama_adapter_to_path()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )

    ensure(default_env_file().is_file(), f"Environment file not found: {default_env_file()}")
    ensure(default_env_setup_script().is_file(), f"Setup script not found: {default_env_setup_script()}")
    ensure(Path(args.adapter_checkpoint).resolve().is_file(), f"Adapter checkpoint not found: {args.adapter_checkpoint}")
    ensure_llama_dir(Path(args.llama_dir).resolve())
    ensure(Path(resolved["qa_json"]).is_file(), f"QA JSON not found: {resolved['qa_json']}")

    require_module("clip")
    require_module("timm")
    require_module("fairscale")
    require_module("sentencepiece")
    require_module("cv2")
    require_module("matplotlib")
    require_module("yaml")
    require_module("pandas")

    if args.require_prepared:
        prepared_dir = Path(resolved["prepared_dir"])
        for filename in ("train_llama.json", "val_llama.json", "train_eval.json", "val_eval.json", "sidecar_val.jsonl", "split_manifest.json"):
            ensure((prepared_dir / filename).is_file(), f"Missing prepared artifact: {prepared_dir / filename}")
        split_manifest = load_json(prepared_dir / "split_manifest.json")
        ensure(
            split_manifest.get("dataset_version", "v5") == resolved["dataset_version"],
            f"Prepared split_manifest dataset_version={split_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
        )
    if args.require_evaluate:
        ensure(Path(resolved["evaluator"]).is_file(), f"Evaluator not found: {resolved['evaluator']}")

    ensure(torch.cuda.is_available(), "CUDA is not available.")
    ensure(torch.cuda.device_count() >= args.expected_gpus, f"Expected at least {args.expected_gpus} GPUs, got {torch.cuda.device_count()}.")

    print(
        "[check_env] "
        f"env={DEFAULT_ENV_NAME} cuda={torch.version.cuda} gpu_count={torch.cuda.device_count()} "
        f"dataset_version={resolved['dataset_version']} llama_dir={Path(args.llama_dir).resolve()} adapter_checkpoint={Path(args.adapter_checkpoint).resolve()}"
    )


if __name__ == "__main__":
    main()
