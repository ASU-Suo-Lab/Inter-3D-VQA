from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from senna_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_VISION_TOWER,
    INFERENCE_DEFAULTS,
    TRAINING_DEFAULTS,
    REPO_ROOT,
    resolve_dataset_version_paths,
    worktree_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Senna Intersection V5 pipeline.")
    parser.add_argument("--stage", choices=["prepare", "check_env", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_NAME_OR_PATH))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--mode", choices=["full", "lora"], default=TRAINING_DEFAULTS["mode"])
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=TRAINING_DEFAULTS["eval_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=TRAINING_DEFAULTS["grad_accum"])
    parser.add_argument("--learning-rate", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--mm-projector-lr", type=float, default=TRAINING_DEFAULTS["mm_projector_lr"])
    parser.add_argument("--save-steps", type=int, default=TRAINING_DEFAULTS["save_steps"])
    parser.add_argument("--eval-steps", type=int, default=TRAINING_DEFAULTS["eval_steps"])
    parser.add_argument("--logging-steps", type=int, default=TRAINING_DEFAULTS["logging_steps"])
    parser.add_argument("--warmup-ratio", type=float, default=TRAINING_DEFAULTS["warmup_ratio"])
    parser.add_argument("--lr-scheduler-type", default=TRAINING_DEFAULTS["lr_scheduler_type"])
    parser.add_argument("--save-total-limit", type=int, default=TRAINING_DEFAULTS["save_total_limit"])
    parser.add_argument("--model-max-length", type=int, default=TRAINING_DEFAULTS["model_max_length"])
    parser.add_argument("--workers", type=int, default=TRAINING_DEFAULTS["workers"])
    parser.add_argument("--lora-r", type=int, default=TRAINING_DEFAULTS["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=TRAINING_DEFAULTS["lora_alpha"])
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=INFERENCE_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=INFERENCE_DEFAULTS["top_p"])
    parser.add_argument("--num-beams", type=int, default=INFERENCE_DEFAULTS["num_beams"])
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29553)
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

    if args.stage in {"prepare", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "senna_v5.cli.prepare",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--qa-json",
                str(Path(resolved["qa_json"]).resolve()),
                "--output-dir",
                str(prepared_dir),
            ]
        )

    if args.stage in {"check_env", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "senna_v5.cli.check_env",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--qa-json",
                str(Path(resolved["qa_json"]).resolve()),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--evaluator",
                str(Path(resolved["evaluator"]).resolve()),
                "--require-prepared",
                "--model-path",
                str(Path(args.model_path).resolve()),
                "--vision-tower",
                str(Path(args.vision_tower).resolve()),
                "--expected-gpus",
                str(args.num_gpus),
            ]
        )

    if args.stage in {"train", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "senna_v5.cli.train",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--mode",
                args.mode,
                "--model-path",
                str(Path(args.model_path).resolve()),
                "--vision-tower",
                str(Path(args.vision_tower).resolve()),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--learning-rate",
                str(args.learning_rate),
                "--mm-projector-lr",
                str(args.mm_projector_lr),
                "--save-steps",
                str(args.save_steps),
                "--eval-steps",
                str(args.eval_steps),
                "--logging-steps",
                str(args.logging_steps),
                "--warmup-ratio",
                str(args.warmup_ratio),
                "--lr-scheduler-type",
                str(args.lr_scheduler_type),
                "--save-total-limit",
                str(args.save_total_limit),
                "--model-max-length",
                str(args.model_max_length),
                "--workers",
                str(args.workers),
                "--lora-r",
                str(args.lora_r),
                "--lora-alpha",
                str(args.lora_alpha),
                "--num-gpus",
                str(args.num_gpus),
                "--master-port",
                str(args.master_port),
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
                "senna_v5.cli.forward",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--model-path",
                str(worktree["best_checkpoint"]),
                "--data-path",
                str(prepared_dir / "val_eval.json"),
                "--output-dir",
                str(prediction_dir),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--num-beams",
                str(args.num_beams),
                *(["--top-p", str(args.top_p)] if args.top_p is not None else []),
            ]
        )

    if args.stage in {"evaluate", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "senna_v5.cli.evaluate",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--predictions",
                str(prediction_dir / "merged_predictions.jsonl"),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--evaluator",
                str(Path(resolved["evaluator"]).resolve()),
                "--output-dir",
                str(eval_output_dir),
                "--split",
                "val",
                *(["--skip-semantic-metrics"] if args.skip_semantic_metrics else []),
            ]
        )


if __name__ == "__main__":
    main()
