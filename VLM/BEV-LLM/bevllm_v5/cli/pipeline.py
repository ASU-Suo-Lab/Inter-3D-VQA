from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bevllm_v5.config.common import (
    DEFAULT_BEVFUSION_CKPT,
    DEFAULT_BEVFUSION_CONFIG,
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_METRIC_DIR,
    DEFAULT_OPENPCDET_ROOT,
    DEFAULT_PREDS_JSONL,
    INFERENCE_DEFAULTS,
    MODEL_DEFAULTS,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from bevllm_v5.utils.io import ensure, load_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BEV-LLM strict Intersection V5 pipeline.")
    parser.add_argument("--stage", choices=["prepare", "check_env", "extract", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--openpcdet-root", default=os.environ.get("BEVLLM_OPENPCDET_ROOT", str(DEFAULT_OPENPCDET_ROOT)))
    parser.add_argument("--bevfusion-config", default=os.environ.get("BEVLLM_BEVFUSION_CONFIG", str(DEFAULT_BEVFUSION_CONFIG)))
    parser.add_argument("--bevfusion-ckpt", default=os.environ.get("BEVLLM_BEVFUSION_CKPT", str(DEFAULT_BEVFUSION_CKPT)))
    parser.add_argument("--feature-key", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--tokenizer-model-max-length", type=int, default=MODEL_DEFAULTS["tokenizer_model_max_length"])
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=MODEL_DEFAULTS["lora_config"]["r"])
    parser.add_argument("--lora-alpha", type=int, default=MODEL_DEFAULTS["lora_config"]["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--max-steps", type=int, default=TRAINING_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TRAINING_DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--learning-rate", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=TRAINING_DEFAULTS["warmup_ratio"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--log-every", type=int, default=TRAINING_DEFAULTS["log_every"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=4)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("[bevllm-pipeline] " + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def torchrun_command(module_name: str, nproc_per_node: int, master_port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        module_name,
    ]


def clear_prediction_dir(work_dir: Path) -> None:
    paths = ensure_worktree_layout(work_dir)
    prediction_dir = paths["predictions"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for path in prediction_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def split_features_complete(prepared_dir: Path, feature_dir: Path, split: str) -> bool:
    frame_manifest = prepared_dir / f"frames_{split}.json"
    if not frame_manifest.is_file():
        return False
    rows = load_json(frame_manifest)
    if not isinstance(rows, list) or not rows:
        return False
    if not feature_dir.is_dir():
        return False
    required_tokens = {str(row["frame_token"]) for row in rows}
    return all((feature_dir / f"{frame_token}.pt").is_file() for frame_token in required_tokens)


def extraction_outputs_complete(prepared_dir: Path, work_dir: Path) -> bool:
    paths = ensure_worktree_layout(work_dir)
    return split_features_complete(prepared_dir, paths["feature_train"], "train") and split_features_complete(
        prepared_dir,
        paths["feature_val"],
        "val",
    )


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    work_dir = Path(resolved["work_dir"]).resolve()
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    ensure_worktree_layout(work_dir)
    prediction_file = work_dir / "predictions" / Path(DEFAULT_PREDS_JSONL).name
    metrics_dir = work_dir / Path(DEFAULT_METRIC_DIR).name

    prepare_command = [
        sys.executable,
        "-m",
        "bevllm_v5.cli.prepare",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--qa-json",
        str(Path(resolved["qa_json"]).resolve()),
        "--info-pkl",
        str(Path(args.info_pkl).resolve()),
        "--output-dir",
        str(prepared_dir),
    ]
    check_env_command = [
        sys.executable,
        "-m",
        "bevllm_v5.cli.check_env",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--expected-gpus",
        str(args.expected_gpus),
        "--qa-json",
        str(Path(resolved["qa_json"]).resolve()),
        "--info-pkl",
        str(Path(args.info_pkl).resolve()),
        "--prepared-dir",
        str(prepared_dir),
        "--work-dir",
        str(work_dir),
        "--evaluator",
        str(Path(resolved["evaluator"]).resolve()),
        "--require-prepared",
        "--require-evaluate",
        "--model-id",
        args.model_id,
        "--cache-dir",
        str(Path(args.cache_dir).expanduser().resolve()),
    ]
    if args.access_token is not None:
        check_env_command.extend(["--access-token", args.access_token])

    extract_command = torchrun_command("bevllm_v5.cli.extract", args.nproc_per_node, args.master_port) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--prepared-dir",
        str(prepared_dir),
        "--info-pkl",
        str(Path(args.info_pkl).resolve()),
        "--work-dir",
        str(work_dir),
        "--openpcdet-root",
        str(Path(args.openpcdet_root).resolve()),
        "--bevfusion-config",
        str(Path(args.bevfusion_config).resolve()),
        "--bevfusion-ckpt",
        str(Path(args.bevfusion_ckpt).resolve()),
        "--split",
        "all",
        "--device",
        args.device,
    ]
    if args.feature_key:
        extract_command.extend(["--feature-key", args.feature_key])
    if args.limit is not None:
        extract_command.extend(["--limit", str(args.limit)])
    if args.overwrite_features:
        extract_command.append("--overwrite")

    train_command = torchrun_command("bevllm_v5.cli.train", args.nproc_per_node, args.master_port + 1) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--prepared-dir",
        str(prepared_dir),
        "--feature-train-dir",
        str(work_dir / "features" / "train"),
        "--feature-val-dir",
        str(work_dir / "features" / "val"),
        "--work-dir",
        str(work_dir),
        "--device",
        args.device,
        "--model-id",
        args.model_id,
        "--cache-dir",
        str(Path(args.cache_dir).expanduser().resolve()),
        "--tokenizer-model-max-length",
        str(args.tokenizer_model_max_length),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--log-every",
        str(args.log_every),
    ]
    if args.access_token is not None:
        train_command.extend(["--access-token", args.access_token])
    if args.disable_lora:
        train_command.append("--disable-lora")
    if args.gradient_checkpointing:
        train_command.append("--gradient-checkpointing")

    forward_command = torchrun_command("bevllm_v5.cli.forward", args.nproc_per_node, args.master_port + 2) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--checkpoint",
        str(work_dir / "checkpoints" / "best"),
        "--prepared-dir",
        str(prepared_dir),
        "--feature-dir",
        str(work_dir / "features" / "val"),
        "--output",
        str(prediction_file),
        "--work-dir",
        str(work_dir),
        "--device",
        args.device,
        "--batch-size",
        str(INFERENCE_DEFAULTS["batch_size"]),
        "--num-workers",
        str(INFERENCE_DEFAULTS["num_workers"]),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    if args.limit is not None:
        forward_command.extend(["--limit", str(args.limit)])

    evaluate_command = [
        sys.executable,
        "-m",
        "bevllm_v5.cli.evaluate",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--predictions",
        str(prediction_file),
        "--prepared-dir",
        str(prepared_dir),
        "--work-dir",
        str(work_dir),
        "--evaluator",
        str(Path(resolved["evaluator"]).resolve()),
        "--output-dir",
        str(metrics_dir),
        "--split",
        "val",
        "--bertscore-model",
        args.bertscore_model,
        "--sim-model",
        args.sim_model,
        "--device",
        args.device,
    ]
    if args.skip_semantic_metrics:
        evaluate_command.append("--skip-semantic-metrics")
    if args.limit is not None:
        evaluate_command.extend(["--limit", str(args.limit)])

    if args.stage in {"prepare", "all"}:
        run_command(prepare_command)
    needs_extract = args.stage == "extract" or (
        args.stage == "all" and (args.overwrite_features or not extraction_outputs_complete(prepared_dir, work_dir))
    )
    if needs_extract:
        check_env_command.append("--require-extract")
        check_env_command.extend(["--openpcdet-root", str(Path(args.openpcdet_root).resolve())])
        check_env_command.extend(["--bevfusion-config", str(Path(args.bevfusion_config).resolve())])
        check_env_command.extend(["--bevfusion-ckpt", str(Path(args.bevfusion_ckpt).resolve())])

    if args.stage in {"check_env", "all"}:
        run_command(check_env_command)
    if args.stage == "all" and not needs_extract:
        print("[bevllm-pipeline] skipping extract because cached train/val features are complete", flush=True)
    if args.stage in {"extract", "all"} and needs_extract:
        run_command(extract_command)
    if args.stage in {"train", "all"}:
        run_command(train_command)
    if args.stage in {"forward", "all"}:
        clear_prediction_dir(work_dir)
        run_command(forward_command)
    if args.stage in {"evaluate", "all"}:
        run_command(evaluate_command)


if __name__ == "__main__":
    main()
