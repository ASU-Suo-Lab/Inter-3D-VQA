import argparse
import importlib
import json
import sys
from pathlib import Path

from omnidrive_v5.utils.paths import DEFAULT_DATASET_VERSION, DEFAULT_INFO_PKL, resolve_dataset_version_paths


REQUIRED_MODULES = (
    "torch",
    "mmcv",
    "mmdet",
    "mmseg",
    "mmdet3d",
    "transformers",
    "accelerate",
    "shapely",
    "matplotlib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the OmniDrive V5/V6 runtime environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-prepared", action="store_true")
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
    versions = {}
    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the active Python environment. "
            "Use the dedicated CUDA 12.8 environment before running OmniDrive V5."
        )

    for required_path in (
        Path(resolved["qa_json"]).resolve(),
        Path(args.info_pkl).resolve(),
        Path(resolved["evaluator"]).resolve(),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    if args.require_prepared:
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        for filename in (
            "infos_train.pkl",
            "infos_val.pkl",
            "sidecar_train.jsonl",
            "sidecar_val.jsonl",
            "intersection_vqa_eval_sidecar.jsonl",
            "split_manifest.json",
        ):
            path = prepared_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Prepared artifact not found: {path}")
        with (prepared_dir / "split_manifest.json").open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("dataset_version") != resolved["dataset_version"]:
            raise ValueError(
                "Prepared split manifest dataset_version mismatch: "
                f"expected {resolved['dataset_version']}, got {manifest.get('dataset_version')}"
            )

    print("[check_env] Python", sys.version.replace("\n", " "))
    print(f"[check_env] dataset_version={resolved['dataset_version']}")
    print(f"[check_env] qa_json={Path(resolved['qa_json']).resolve()}")
    print(f"[check_env] prepared_dir={Path(resolved['prepared_dir']).resolve()}")
    print(f"[check_env] work_dir={Path(resolved['work_dir']).resolve()}")
    print(f"[check_env] evaluator={Path(resolved['evaluator']).resolve()}")
    for module_name in REQUIRED_MODULES:
        print(f"[check_env] {module_name}={versions[module_name]}")
    print(f"[check_env] cuda_available={torch.cuda.is_available()}")
    print(f"[check_env] torch_cuda_version={torch.version.cuda}")


if __name__ == "__main__":
    main()
