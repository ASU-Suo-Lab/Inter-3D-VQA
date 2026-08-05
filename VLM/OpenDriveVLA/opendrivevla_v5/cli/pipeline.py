from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from opendrivevla_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_FEATURE_TRAIN_DIR,
    DEFAULT_FEATURE_VAL_DIR,
    DEFAULT_METRIC_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_DIR,
    DEFAULT_PREDS_JSONL,
    DEFAULT_INFO_PKL,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the strict OpenDriveVLA Intersection V5 pipeline.")
    parser.add_argument("--stage", choices=["prepare", "check_env", "extract", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--infos-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--train-mm-projector", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def run_command(command):
    print("[openDriveVLA-pipeline] " + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def torchrun_command(module_name: str, nproc_per_node: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc_per_node), "-m", module_name]


def main():
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
    prediction_file = work_dir / DEFAULT_PREDICTION_DIR.name / DEFAULT_PREDS_JSONL.name
    metrics_dir = work_dir / DEFAULT_METRIC_DIR.name

    prepare_command = [
        sys.executable,
        "-m",
        "opendrivevla_v5.cli.prepare",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--qa-json",
        str(Path(resolved["qa_json"]).resolve()),
        "--info-pkl",
        str(Path(args.infos_pkl).resolve()),
        "--output-dir",
        str(prepared_dir),
    ]
    check_env_command = [
        sys.executable,
        "-m",
        "opendrivevla_v5.cli.check_env",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--qa-json",
        str(Path(resolved["qa_json"]).resolve()),
        "--infos-pkl",
        str(Path(args.infos_pkl).resolve()),
        "--prepared-dir",
        str(prepared_dir),
        "--work-dir",
        str(work_dir),
        "--evaluator",
        str(Path(resolved["evaluator"]).resolve()),
        "--require-prepared",
        "--require-evaluate",
        "--model-path",
        str(Path(args.model_path).resolve()),
    ]
    extract_command = torchrun_command("opendrivevla_v5.cli.extract", args.nproc_per_node) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--model-path",
        str(Path(args.model_path).resolve()),
        "--prepared-dir",
        str(prepared_dir),
        "--work-dir",
        str(work_dir),
        "--split",
        "all",
        "--device",
        args.device,
        "--attn-implementation",
        args.attn_implementation,
    ]
    if args.limit is not None:
        extract_command.extend(["--limit", str(args.limit)])
    if args.overwrite_features:
        extract_command.append("--overwrite")

    train_command = torchrun_command("opendrivevla_v5.cli.train", args.nproc_per_node) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--model-path",
        str(Path(args.model_path).resolve()),
        "--prepared-dir",
        str(prepared_dir),
        "--feature-train-dir",
        str(work_dir / DEFAULT_FEATURE_TRAIN_DIR.name),
        "--feature-val-dir",
        str(work_dir / DEFAULT_FEATURE_VAL_DIR.name),
        "--work-dir",
        str(work_dir),
        "--device",
        args.device,
        "--attn-implementation",
        args.attn_implementation,
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--max-steps",
        str(args.max_steps),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
    ]
    if args.limit is not None:
        train_command.extend(["--max-train-samples", str(args.limit), "--max-eval-samples", str(args.limit)])
    if args.disable_lora:
        train_command.append("--disable-lora")
    if args.gradient_checkpointing:
        train_command.append("--gradient-checkpointing")
    if args.train_mm_projector:
        train_command.append("--train-mm-projector")

    forward_command = torchrun_command("opendrivevla_v5.cli.forward", args.nproc_per_node) + [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--prepared-dir",
        str(prepared_dir),
        "--uniad-pth-dir",
        str(work_dir / DEFAULT_FEATURE_VAL_DIR.name),
        "--output",
        str(prediction_file),
        "--work-dir",
        str(work_dir),
        "--device",
        args.device,
        "--attn-implementation",
        args.attn_implementation,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.limit is not None:
        forward_command.extend(["--limit", str(args.limit)])

    evaluate_command = [
        sys.executable,
        "-m",
        "opendrivevla_v5.cli.evaluate",
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
    if args.stage in {"check_env", "all"}:
        run_command(check_env_command)
    if args.stage in {"extract", "all"}:
        run_command(extract_command)
    if args.stage in {"train", "all"}:
        run_command(train_command)
    if args.stage in {"forward", "all"}:
        run_command(forward_command)
    if args.stage in {"evaluate", "all"}:
        run_command(evaluate_command)


if __name__ == "__main__":
    main()
