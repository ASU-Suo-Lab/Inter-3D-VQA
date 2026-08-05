from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from geovlm_intersection.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_LION_QUALITY,
    PACKAGE_NAME,
    REPO_ROOT,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GeoVLM v5 structured-subset pipeline.")
    parser.add_argument(
        "--stage",
        choices=["prepare", "check_env", "extract", "train", "forward", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--source-prepared-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--info-pkl", default=None)
    parser.add_argument("--lion-quality", choices=["low", "mid", "high"], default=DEFAULT_LION_QUALITY)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--lion-device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--extract-gpus", type=int, default=4, help="Number of local GPUs to use for torchrun feature extraction.")
    parser.add_argument("--train-gpus", type=int, default=4, help="Number of local GPUs to use for torchrun training.")
    parser.add_argument("--master-port", type=int, default=29531, help="torchrun master port for local DDP training.")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def append_if_set(command: list[str], flag: str, value: object) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        source_prepared_dir=args.source_prepared_dir,
        prepared_dir=args.prepared_dir,
        evaluator=args.evaluator,
        work_dir=args.work_dir,
    )
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    python_bin = sys.executable
    common = [
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--prepared-dir",
        str(resolved["prepared_dir"]),
        "--work-dir",
        str(worktree["work_dir"]),
    ]

    if args.stage in {"prepare", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.prepare",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--qa-json",
            str(resolved["qa_json"]),
            "--source-prepared-dir",
            str(resolved["source_prepared_dir"]),
            "--prepared-dir",
            str(resolved["prepared_dir"]),
        ]
        run_command(command)

    if args.stage in {"check_env", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.check_env",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(resolved["prepared_dir"]),
            "--lion-quality",
            args.lion_quality,
        ]
        append_if_set(command, "--info-pkl", args.info_pkl)
        run_command(command)

    if args.stage in {"extract", "all"}:
        extract_script_args = [
            str(REPO_ROOT / "geovlm_intersection" / "cli" / "extract.py"),
            *common,
            "--lion-quality",
            args.lion_quality,
            "--max-objects",
            str(args.max_objects),
            "--qwen-device",
            args.qwen_device,
            "--lion-device",
            args.lion_device,
        ]
        append_if_set(extract_script_args, "--info-pkl", args.info_pkl)
        if args.extract_gpus > 1:
            command = [
                python_bin,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(args.extract_gpus),
                "--nnodes",
                "1",
                "--node_rank",
                "0",
                "--master_addr",
                "127.0.0.1",
                "--master_port",
                str(args.master_port),
                *extract_script_args,
            ]
        else:
            command = [python_bin, *extract_script_args]
        run_command(command)

    if args.stage in {"train", "all"}:
        train_script_args = [
            str(REPO_ROOT / "geovlm_intersection" / "cli" / "train.py"),
            *common,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            str(args.device),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.train_gpus > 1:
            command = [
                python_bin,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(args.train_gpus),
                "--nnodes",
                "1",
                "--node_rank",
                "0",
                "--master_addr",
                "127.0.0.1",
                "--master_port",
                str(args.master_port),
                *train_script_args,
            ]
        else:
            command = [python_bin, *train_script_args]
        run_command(command)

    if args.stage in {"forward", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.forward",
            *common,
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(args.device),
            "--num-workers",
            str(args.num_workers),
        ]
        run_command(command)

    if args.stage in {"evaluate", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.evaluate",
            *common,
            "--evaluator",
            str(resolved["evaluator"]),
        ]
        if args.skip_semantic_metrics:
            command.append("--skip-semantic-metrics")
        run_command(command)


if __name__ == "__main__":
    main()
