from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from nuscenesqa_v5.config.common import DEFAULT_DATASET_VERSION, REPO_ROOT, ensure_worktree_layout, resolve_dataset_version_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NuScenes-QA intersection pipeline.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--stage", choices=["prepare", "check_env", "extract", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29571)
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())

    if args.stage in {"prepare", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "nuscenesqa_v5.cli.prepare",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--qa-json",
                str(resolved["qa_json"]),
                "--output-dir",
                str(prepared_dir),
            ]
        )

    if args.stage in {"check_env", "all"}:
        check_args = [
            python_bin,
            "-m",
            "nuscenesqa_v5.cli.check_env",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--qa-json",
            str(resolved["qa_json"]),
            "--prepared-dir",
            str(prepared_dir),
            "--work-dir",
            str(worktree["work_dir"]),
            "--evaluator",
            str(resolved["evaluator"]),
        ]
        if args.stage == "all":
            check_args.extend(["--require-prepared", "--require-evaluate"])
        run_command(check_args)

    if args.stage in {"extract", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "nuscenesqa_v5.cli.extract",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(worktree["work_dir"]),
            ]
        )

    if args.stage in {"train", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "nuscenesqa_v5.cli.check_env",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--qa-json",
                str(resolved["qa_json"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(worktree["work_dir"]),
                "--evaluator",
                str(resolved["evaluator"]),
                "--require-prepared",
                "--require-features",
                "--require-evaluate",
            ]
        )
        run_command(
            [
                "torchrun",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                "--node_rank=0",
                f"--master_port={args.master_port}",
                "-m",
                "nuscenesqa_v5.cli.train",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(worktree["work_dir"]),
            ]
        )

    if args.stage in {"forward", "all"}:
        prediction_dir = worktree["predictions"]
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "torchrun",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                "--node_rank=0",
                f"--master_port={args.master_port + 1}",
                "-m",
                "nuscenesqa_v5.cli.forward",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(worktree["work_dir"]),
            ]
        )

    if args.stage in {"evaluate", "all"}:
        command = [
            python_bin,
            "-m",
            "nuscenesqa_v5.cli.evaluate",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(prepared_dir),
            "--work-dir",
            str(worktree["work_dir"]),
            "--evaluator",
            str(resolved["evaluator"]),
        ]
        if args.skip_semantic_metrics:
            command.append("--skip-semantic-metrics")
        run_command(command)


if __name__ == "__main__":
    main()
