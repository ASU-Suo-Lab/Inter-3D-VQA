from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from drivelm_v5.config.common import (
    DEFAULT_ADAPTER_PRETRAIN,
    DEFAULT_DATASET_VERSION,
    INFERENCE_DEFAULTS,
    DEFAULT_LLAMAA_DIR,
    TRAINING_DEFAULTS,
    REPO_ROOT,
    resolve_dataset_version_paths,
    worktree_paths,
)
from drivelm_v5.utils.io import ensure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DriveLM Intersection V5 pipeline.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--stage", choices=["prepare", "check_env", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--llama-dir", default=str(DEFAULT_LLAMAA_DIR))
    parser.add_argument("--adapter-checkpoint", default=str(DEFAULT_ADAPTER_PRETRAIN))
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29543)
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size_per_gpu"])
    parser.add_argument("--accum-iter", type=int, default=TRAINING_DEFAULTS["accum_iter"])
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=TRAINING_DEFAULTS["warmup_epochs"])
    parser.add_argument("--max-words", type=int, default=TRAINING_DEFAULTS["max_words"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--log-every", type=int, default=TRAINING_DEFAULTS["log_every"])
    parser.add_argument("--forward-batch-size", type=int, default=INFERENCE_DEFAULTS["batch_size"])
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
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
    work_dir = Path(resolved["work_dir"]).resolve()
    worktree = worktree_paths(work_dir)
    prediction_dir = Path(args.prediction_dir).resolve() if args.prediction_dir else worktree["predictions"]
    eval_output_dir = Path(args.eval_output_dir).resolve() if args.eval_output_dir else worktree["metrics"]
    llama_dir = Path(args.llama_dir).resolve()
    adapter_checkpoint = Path(args.adapter_checkpoint).resolve()

    if args.stage in {"prepare", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "drivelm_v5.cli.prepare",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--qa-json",
                str(resolved["qa_json"]),
                "--output-dir",
                str(prepared_dir),
            ]
        )

    if args.stage in {"check_env", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "drivelm_v5.cli.check_env",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--llama-dir",
                str(llama_dir),
                "--adapter-checkpoint",
                str(adapter_checkpoint),
                "--qa-json",
                str(resolved["qa_json"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--evaluator",
                str(resolved["evaluator"]),
                "--require-prepared",
                "--require-evaluate",
            ]
        )

    if args.stage in {"train", "all"}:
        run_command(
            [
                "torchrun",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                "--node_rank=0",
                f"--master_port={args.master_port}",
                "-m",
                "drivelm_v5.cli.train",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--llama-dir",
                str(llama_dir),
                "--pretrained-path",
                str(adapter_checkpoint),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--batch-size",
                str(args.batch_size),
                "--accum-iter",
                str(args.accum_iter),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--warmup-epochs",
                str(args.warmup_epochs),
                "--max-words",
                str(args.max_words),
                "--num-workers",
                str(args.num_workers),
                "--log-every",
                str(args.log_every),
            ]
        )

    if args.stage in {"forward", "all"}:
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
                "drivelm_v5.cli.forward",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--llama-dir",
                str(llama_dir),
                "--checkpoint",
                str(worktree["best_checkpoint"]),
                "--data-path",
                str(prepared_dir / "val_eval.json"),
                "--output-dir",
                str(prediction_dir),
                "--batch-size",
                str(args.forward_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
            ]
        )

    if args.stage in {"evaluate", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "drivelm_v5.cli.evaluate",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--predictions",
                str(prediction_dir / "merged_predictions.jsonl"),
                "--sidecar-jsonl",
                str(prepared_dir / "sidecar_val.jsonl"),
                "--output-dir",
                str(eval_output_dir),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--evaluator",
                str(resolved["evaluator"]),
                "--split",
                "val",
                *(["--skip-semantic-metrics"] if args.skip_semantic_metrics else []),
            ]
        )


if __name__ == "__main__":
    main()
