from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from claude46sonnet_vlm.config.common import DEFAULT_DATASET_VERSION, PACKAGE_NAME, REPO_ROOT, ensure_worktree_layout, resolve_dataset_version_paths
from claude46sonnet_vlm.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Claude intersection pipeline.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--stage", choices=["check_env", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-image-edge", type=int, default=None)
    parser.add_argument("--request-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def append_if_set(command: list[str], flag: str, value: object) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def load_forward_run_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
        evaluator=args.evaluator,
    )
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())

    if args.stage in {"check_env", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.check_env",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(resolved["prepared_dir"]),
            "--work-dir",
            str(worktree["work_dir"]),
            "--evaluator",
            str(resolved["evaluator"]),
        ]
        run_command(command)

    if args.stage in {"forward", "all"}:
        prediction_dir = worktree["predictions"]
        prediction_dir.mkdir(parents=True, exist_ok=True)
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.forward",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(resolved["prepared_dir"]),
            "--work-dir",
            str(worktree["work_dir"]),
        ]
        append_if_set(command, "--model", args.model)
        append_if_set(command, "--limit", args.limit)
        append_if_set(command, "--temperature", args.temperature)
        append_if_set(command, "--max-output-tokens", args.max_output_tokens)
        append_if_set(command, "--max-image-edge", args.max_image_edge)
        append_if_set(command, "--request-timeout-seconds", args.request_timeout_seconds)
        append_if_set(command, "--max-retries", args.max_retries)
        append_if_set(command, "--retry-backoff-seconds", args.retry_backoff_seconds)
        if args.stage == "forward":
            run_command(command)
        else:
            try:
                run_command(command)
            except subprocess.CalledProcessError as exc:
                forward_state = load_forward_run_state(worktree["forward_run_json"])
                success_count = int(forward_state.get("success_count", 0) or 0)
                requested_count = int(forward_state.get("requested_count", 0) or 0)
                if success_count > 0:
                    evaluate_command = [
                        python_bin,
                        "-m",
                        f"{PACKAGE_NAME}.cli.evaluate",
                        "--dataset-version",
                        str(resolved["dataset_version"]),
                        "--prepared-dir",
                        str(resolved["prepared_dir"]),
                        "--work-dir",
                        str(worktree["work_dir"]),
                        "--evaluator",
                        str(resolved["evaluator"]),
                        "--allow-partial",
                    ]
                    if args.skip_semantic_metrics:
                        evaluate_command.append("--skip-semantic-metrics")
                    run_command(evaluate_command)
                    raise RuntimeError(
                        f"Forward interrupted after {success_count}/{requested_count} samples; partial evaluation completed. See {worktree['forward_run_json']} and {worktree['metrics_run_json']}."
                    ) from exc
                raise

    if args.stage in {"evaluate", "all"}:
        command = [
            python_bin,
            "-m",
            f"{PACKAGE_NAME}.cli.evaluate",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(resolved["prepared_dir"]),
            "--work-dir",
            str(worktree["work_dir"]),
            "--evaluator",
            str(resolved["evaluator"]),
        ]
        if args.allow_partial:
            command.append("--allow-partial")
        if args.skip_semantic_metrics:
            command.append("--skip-semantic-metrics")
        run_command(command)


if __name__ == "__main__":
    main()
