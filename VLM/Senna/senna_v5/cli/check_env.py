from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from senna_v5.config.common import DEFAULT_DATASET_VERSION, DEFAULT_ENV_NAME, DEFAULT_MODEL_NAME_OR_PATH, DEFAULT_VISION_TOWER, resolve_dataset_version_paths
from senna_v5.utils.io import ensure, load_json


REQUIRED_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "sentencepiece",
    "matplotlib",
    "pandas",
    "yaml",
    "sklearn",
    "timm",
    "einops",
    "scipy",
    "tqdm",
    "bert_score",
    "sentence_transformers",
    "cv2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Senna V5 environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--expected-env", default=DEFAULT_ENV_NAME)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_NAME_OR_PATH))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--require-prepared", action="store_true")
    return parser.parse_args()


def require_module(module_name: str) -> None:
    importlib.import_module(module_name)


def read_os_release() -> dict[str, str]:
    data: dict[str, str] = {}
    path = Path("/etc/os-release")
    ensure(path.is_file(), f"Missing OS metadata file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    os_release = read_os_release()
    ensure(os_release.get("ID") == "ubuntu", f"Unsupported OS ID: {os_release.get('ID')}")
    ensure(os_release.get("VERSION_ID") == "22.04", f"Unsupported Ubuntu version: {os_release.get('VERSION_ID')}")

    for module_name in REQUIRED_MODULES:
        require_module(module_name)

    import torch

    ensure(torch.cuda.is_available(), "CUDA is required for Senna V5.")
    ensure(torch.cuda.device_count() == args.expected_gpus, f"Expected {args.expected_gpus} GPUs, found {torch.cuda.device_count()}.")

    model_path = Path(args.model_path).resolve()
    vision_tower = Path(args.vision_tower).resolve()
    evaluator = Path(resolved["evaluator"]).resolve()
    ensure(Path(resolved["qa_json"]).is_file(), f"QA JSON not found: {resolved['qa_json']}")
    ensure(model_path.is_dir(), f"Model path not found: {model_path}")
    ensure((model_path / "config.json").is_file(), f"Missing config.json under {model_path}")
    ensure((model_path / "tokenizer.model").is_file(), f"Missing tokenizer.model under {model_path}")
    ensure((model_path / "pytorch_model.bin.index.json").is_file(), f"Missing pytorch_model.bin.index.json under {model_path}")
    ensure(vision_tower.is_dir(), f"Vision tower path not found: {vision_tower}")
    ensure((vision_tower / "config.json").is_file(), f"Missing config.json under {vision_tower}")
    ensure((vision_tower / "preprocessor_config.json").is_file(), f"Missing preprocessor_config.json under {vision_tower}")
    ensure(evaluator.is_file(), f"External evaluator not found: {evaluator}")
    if args.require_prepared:
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        for filename in ("train.json", "val.json", "val_eval.json", "sidecar_train.jsonl", "sidecar_val.jsonl", "split_manifest.json"):
            ensure((prepared_dir / filename).is_file(), f"Missing prepared artifact: {prepared_dir / filename}")
        split_manifest = load_json(prepared_dir / "split_manifest.json")
        ensure(
            split_manifest.get("dataset_version", "v5") == resolved["dataset_version"],
            f"Prepared split_manifest dataset_version={split_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
        )

    print(
        "[senna_v5.check_env] "
        f"os=ubuntu {os_release['VERSION_ID']} "
        f"cuda={torch.version.cuda} gpus={torch.cuda.device_count()} "
        f"dataset_version={resolved['dataset_version']} model={model_path} vision_tower={vision_tower} evaluator={evaluator}"
    )


if __name__ == "__main__":
    main()
