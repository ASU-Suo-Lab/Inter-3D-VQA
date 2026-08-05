from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from bevllm_v5.config.common import (
    DEFAULT_BEVFUSION_CKPT,
    DEFAULT_BEVFUSION_CONFIG,
    DEFAULT_DATASET_VERSION,
    DEFAULT_EXPECTED_GPUS,
    DEFAULT_INFO_PKL,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_OPENPCDET_ROOT,
    DEFAULT_QA_JSON,
    resolve_dataset_version_paths,
)
from bevllm_v5.utils.hf import ensure_hf_runtime_ready
from bevllm_v5.utils.io import ensure, load_json
from bevllm_v5.utils.openpcdet import (
    ensure_openpcdet_on_path,
    load_openpcdet_cfg,
    probe_openpcdet_extract_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the BEV-LLM V5 runtime environment.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--expected-gpus", type=int, default=DEFAULT_EXPECTED_GPUS)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-evaluate", action="store_true")
    parser.add_argument("--require-extract", action="store_true")
    parser.add_argument("--openpcdet-root", default=str(DEFAULT_OPENPCDET_ROOT))
    parser.add_argument("--bevfusion-config", default=str(DEFAULT_BEVFUSION_CONFIG))
    parser.add_argument("--bevfusion-ckpt", default=str(DEFAULT_BEVFUSION_CKPT))
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
    ensure(Path(resolved["qa_json"]).resolve().is_file(), f"QA JSON not found: {resolved['qa_json']}")
    ensure(Path(args.info_pkl).resolve().is_file(), f"Infos PKL not found: {args.info_pkl}")

    require_module("transformers")
    require_module("huggingface_hub")
    require_module("peft")
    require_module("matplotlib")
    require_module("sentence_transformers")
    require_module("bert_score")
    if args.require_evaluate:
        ensure(Path(resolved["evaluator"]).resolve().is_file(), f"Evaluator not found: {resolved['evaluator']}")

    if args.require_prepared:
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        manifest_path = prepared_dir / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )

    cache_root = ensure_hf_runtime_ready(args.model_id, args.access_token, args.cache_dir)

    if args.require_extract:
        openpcdet_root = Path(args.openpcdet_root).resolve()
        bevfusion_config = Path(args.bevfusion_config).resolve()
        bevfusion_ckpt = Path(args.bevfusion_ckpt).resolve()
        ensure(openpcdet_root.is_dir(), f"OpenPCDet root not found: {openpcdet_root}")
        ensure(bevfusion_config.is_file(), f"BEVFusion config not found: {bevfusion_config}")
        ensure(bevfusion_ckpt.is_file(), f"BEVFusion checkpoint not found: {bevfusion_ckpt}")

        ensure_openpcdet_on_path(openpcdet_root)
        cfg = load_openpcdet_cfg(bevfusion_config, openpcdet_root)
        probe_openpcdet_extract_stack(openpcdet_root, cfg)

    ensure(torch.cuda.is_available(), "CUDA is not available.")
    ensure(
        torch.cuda.device_count() >= args.expected_gpus,
        f"Expected at least {args.expected_gpus} GPUs, got {torch.cuda.device_count()}.",
    )

    print(
        "[check_env] "
        f"cuda={torch.version.cuda} gpu_count={torch.cuda.device_count()} "
        f"dataset_version={resolved['dataset_version']} require_extract={args.require_extract} "
        f"model_id={args.model_id} cache_dir={cache_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
