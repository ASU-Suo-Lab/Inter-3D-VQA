from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from traffixqwen_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_EVAL_DIR,
    DEFAULT_EVAL_STEPS,
    DEFAULT_FINETUNE_MODE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_IMAGE_TOKEN_COST,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BIAS,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_MAX_LENGTH,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_PREPARED_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SAVE_STEPS,
    DEFAULT_SAVE_TOTAL_LIMIT,
    DEFAULT_TRAIN_ATTN_IMPLEMENTATION,
    DEFAULT_WARMUP_RATIO,
    REPO_ROOT,
    resolve_dataset_version_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict TraffiX-Qwen Intersection V5 pipeline.")
    parser.add_argument("--stage", choices=["prepare", "check_env", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--model-max-length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--image-token-cost", type=int, default=DEFAULT_IMAGE_TOKEN_COST)
    parser.add_argument(
        "--finetune-mode",
        default=DEFAULT_FINETUNE_MODE,
        choices=["adapter_only", "adapter_lm_lora", "full_lm"],
    )
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-bias", default=DEFAULT_LORA_BIAS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-train-epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument("--dataloader-num-workers", type=int, default=DEFAULT_DATALOADER_NUM_WORKERS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--forward-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-attn-implementation",
        default=DEFAULT_TRAIN_ATTN_IMPLEMENTATION,
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--debug-forward", action="store_true")
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
    work_dir = Path(resolved["work_dir"]).resolve()
    model_output_dir = Path(args.model_output_dir).resolve() if args.model_output_dir else work_dir / "checkpoints" / "best"
    prediction_dir = Path(args.prediction_dir).resolve() if args.prediction_dir else work_dir / "predictions"
    eval_output_dir = Path(args.eval_output_dir).resolve() if args.eval_output_dir else work_dir / "metrics"

    if args.stage in {"prepare", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "traffixqwen_v5.cli.prepare",
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
                "traffixqwen_v5.cli.check_env",
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
            ]
        )

    if args.stage in {"train", "all"}:
        run_command(
            [
                python_bin,
                "-m",
                "traffixqwen_v5.cli.train",
                "--dataset-version",
                str(resolved["dataset_version"]),
                "--prepared-dir",
                str(prepared_dir),
                "--work-dir",
                str(work_dir),
                "--data-path",
                str(prepared_dir / "train.json"),
                "--output-dir",
                str(model_output_dir),
                "--model-max-length",
                str(args.model_max_length),
                "--image-token-cost",
                str(args.image_token_cost),
                "--finetune-mode",
                args.finetune_mode,
                "--lora-r",
                str(args.lora_r),
                "--lora-alpha",
                str(args.lora_alpha),
                "--lora-dropout",
                str(args.lora_dropout),
                "--lora-bias",
                args.lora_bias,
                "--per-device-train-batch-size",
                str(args.per_device_train_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--num-train-epochs",
                str(args.num_train_epochs),
                "--save-total-limit",
                str(args.save_total_limit),
                "--dataloader-num-workers",
                str(args.dataloader_num_workers),
                "--logging-steps",
                str(args.logging_steps),
                "--eval-steps",
                str(args.eval_steps),
                "--save-steps",
                str(args.save_steps),
                "--warmup-ratio",
                str(args.warmup_ratio),
                "--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing",
                "--attn-implementation",
                args.train_attn_implementation,
            ]
        )

    if args.stage in {"forward", "all"}:
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        command = [
            python_bin,
            "-m",
            "traffixqwen_v5.cli.forward",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(prepared_dir),
            "--work-dir",
            str(work_dir),
            "--model-path",
            str(model_output_dir),
            "--data-path",
            str(prepared_dir / "val.json"),
            "--output-dir",
            str(prediction_dir),
            "--num-gpus",
            "4",
            "--model-max-length",
            str(args.model_max_length),
            "--attn-implementation",
            args.attn_implementation,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.forward_batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.debug_forward:
            command.append("--debug-forward")
        run_command(command)

    if args.stage in {"evaluate", "all"}:
        command = [
            python_bin,
            "-m",
            "traffixqwen_v5.cli.evaluate",
            "--dataset-version",
            str(resolved["dataset_version"]),
            "--prepared-dir",
            str(prepared_dir),
            "--evaluator",
            str(Path(resolved["evaluator"]).resolve()),
            "--predictions",
            str(prediction_dir / "merged_predictions.jsonl"),
            "--output-dir",
            str(eval_output_dir),
            "--split",
            "val",
        ]
        if args.skip_semantic_metrics:
            command.append("--skip-semantic-metrics")
        run_command(command)


if __name__ == "__main__":
    main()
