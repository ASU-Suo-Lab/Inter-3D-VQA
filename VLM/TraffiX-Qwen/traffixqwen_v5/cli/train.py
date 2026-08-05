from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from traffixqwen_v5.config.common import DEFAULT_DATASET_VERSION, resolve_dataset_version_paths
from traffixqwen_v5.config.common import (
    DEFAULT_FINETUNE_MODE,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_IMAGE_TOKEN_COST,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BIAS,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MODEL_MAX_LENGTH,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_SAVE_TOTAL_LIMIT,
    DEFAULT_TRAIN_ATTN_IMPLEMENTATION,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_WARMUP_RATIO,
    REPO_ROOT,
)
from traffixqwen_v5.data.common import (
    DEFAULT_MM_PROJECTOR,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_PREPARED_DIR,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_VISION_TOWER,
    dump_json,
    ensure,
    load_json,
)
from traffixqwen_v5.utils.training_artifacts import (
    build_layout,
    cleanup_trainer_run_dir,
    copy_trainer_state,
    finalize_best_checkpoint,
    load_trainer_state,
    write_loss_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch strict TraffiX-Qwen Intersection V5 training.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL_NAME_OR_PATH))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--pretrain-mm-mlp-adapter", default=str(DEFAULT_MM_PROJECTOR))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="traffix-qwen-intersection-v5")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-train-epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument("--dataloader-num-workers", type=int, default=DEFAULT_DATALOADER_NUM_WORKERS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--master-port", type=int, default=29541)
    parser.add_argument("--model-max-length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--image-token-cost", type=int, default=DEFAULT_IMAGE_TOKEN_COST)
    parser.add_argument(
        "--finetune-mode",
        default=DEFAULT_FINETUNE_MODE,
        choices=["adapter_only", "adapter_lm_lora", "full_lm"],
    )
    parser.add_argument("--mm-tunable-parts", default=None)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-bias", default=DEFAULT_LORA_BIAS)
    parser.add_argument("--debug-loss", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_TRAIN_ATTN_IMPLEMENTATION,
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def resolve_finetune_settings(args: argparse.Namespace) -> tuple[str, bool]:
    mode_defaults = {
        "adapter_only": ("mm_mlp_adapter", False),
        "adapter_lm_lora": ("mm_mlp_adapter", True),
        "full_lm": ("mm_mlp_adapter,mm_language_model", False),
    }
    default_parts, default_lora_enable = mode_defaults[args.finetune_mode]
    mm_tunable_parts = args.mm_tunable_parts or default_parts
    tunable_parts = {part.strip() for part in mm_tunable_parts.split(",") if part.strip()}

    if args.finetune_mode == "full_lm":
        ensure(
            "mm_language_model" in tunable_parts,
            "--finetune-mode full_lm requires mm_language_model to remain trainable.",
        )
    else:
        ensure(
            "mm_language_model" not in tunable_parts,
            f"--finetune-mode {args.finetune_mode} does not allow mm_language_model in --mm-tunable-parts. "
            "Use --finetune-mode full_lm for full language-model finetuning.",
        )
    if default_lora_enable:
        ensure(args.lora_r > 0, "--lora-r must be positive when LoRA is enabled.")
        ensure(args.lora_alpha > 0, "--lora-alpha must be positive when LoRA is enabled.")
        ensure(0.0 <= args.lora_dropout < 1.0, "--lora-dropout must be in [0.0, 1.0).")
    return mm_tunable_parts, default_lora_enable


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    data_path = Path(args.data_path).resolve() if args.data_path is not None else prepared_dir / "train.json"
    eval_data_path = Path(args.eval_data_path).resolve() if args.eval_data_path is not None else prepared_dir / "val.json"
    model_name_or_path = Path(args.model_name_or_path).resolve()
    vision_tower = Path(args.vision_tower).resolve()
    mm_projector = Path(args.pretrain_mm_mlp_adapter).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else Path(resolved["work_dir"]) / "checkpoints" / "best"
    layout = build_layout(output_dir)
    trainer_output_dir = layout["trainer_run_dir"]
    logs_dir = layout["logs_dir"]
    plots_dir = layout["plots_dir"]
    manifest_path = prepared_dir / "split_manifest.json"

    ensure(data_path.is_file(), f"Training data not found: {data_path}")
    ensure(eval_data_path.is_file(), f"Evaluation data not found: {eval_data_path}")
    ensure(manifest_path.is_file(), f"Prepared split manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )
    ensure(model_name_or_path.exists(), f"Base model path not found: {model_name_or_path}")
    ensure(vision_tower.exists(), f"Vision tower path not found: {vision_tower}")
    ensure(mm_projector.is_file(), f"MM projector checkpoint not found: {mm_projector}")
    ensure(args.model_max_length > 0, "--model-max-length must be a positive integer.")
    ensure(args.image_token_cost > 0, "--image-token-cost must be a positive integer.")
    mm_tunable_parts, lora_enable = resolve_finetune_settings(args)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer_output_dir.parent.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = sorted(trainer_output_dir.glob("checkpoint-*")) if trainer_output_dir.exists() else []
    ensure(
        args.resume or not existing_checkpoints,
        f"Refusing to resume existing TraffiX-Qwen V5 checkpoints implicitly. "
        f"Found {len(existing_checkpoints)} checkpoints under {trainer_output_dir}. "
        f"Use a fresh --output-dir or pass --resume explicitly.",
    )
    if trainer_output_dir.exists() and not args.resume:
        shutil.rmtree(trainer_output_dir)
    logging_steps = 1 if args.debug_loss else args.logging_steps
    print(
        f"[train] finetune_mode={args.finetune_mode} mm_tunable_parts={mm_tunable_parts} "
        f"lora_enable={lora_enable} output_dir={output_dir} "
        f"image_token_cost={args.image_token_cost}"
    )
    if lora_enable:
        print(
            f"[train] lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
            f"lora_dropout={args.lora_dropout} lora_bias={args.lora_bias}"
        )

    command = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_port={args.master_port}",
        "-m",
        "llava.train.train_traffixqwen_v5",
        "--model_name_or_path",
        str(model_name_or_path),
        "--version",
        DEFAULT_PROMPT_VERSION,
        "--data_path",
        str(data_path),
        "--image_folder",
        "",
        f"--mm_tunable_parts={mm_tunable_parts}",
        "--mm_vision_tower_lr=2e-6",
        "--vision_tower",
        str(vision_tower),
        "--pretrain_mm_mlp_adapter",
        str(mm_projector),
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--mm_newline_position",
        "one_token",
        "--group_by_length",
        "True",
        "--mm_patch_merge_type",
        "spatial_unpad",
        "--bf16",
        "True",
        "--run_name",
        args.run_name,
        "--output_dir",
        str(trainer_output_dir),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--evaluation_strategy",
        "steps",
        "--eval_steps",
        str(args.eval_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--load_best_model_at_end",
        "True",
        "--metric_for_best_model",
        "eval_loss",
        "--greater_is_better",
        "False",
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        "0.0",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        str(logging_steps),
        "--logging_nan_inf_filter",
        "False",
        "--tf32",
        "True",
        "--model_max_length",
        str(args.model_max_length),
        "--gradient_checkpointing",
        "True" if args.gradient_checkpointing else "False",
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--ddp_find_unused_parameters",
        "False",
        "--lazy_preprocess",
        "True",
        "--torch_compile",
        "True" if args.torch_compile else "False",
        "--torch_compile_backend",
        "inductor",
        "--dataloader_drop_last",
        "False",
        "--report_to",
        args.report_to,
        "--attn_implementation",
        args.attn_implementation,
        "--lora_enable",
        "True" if lora_enable else "False",
    ]
    if lora_enable:
        command.extend(
            [
                "--lora_r",
                str(args.lora_r),
                "--lora_alpha",
                str(args.lora_alpha),
                "--lora_dropout",
                str(args.lora_dropout),
                "--lora_bias",
                args.lora_bias,
            ]
        )
    if args.max_steps > 0:
        command.extend(["--max_steps", str(args.max_steps)])

    if args.print_command:
        print(" ".join(command))
        return

    env = os.environ.copy()
    env["TRAFFIXQWEN_V5_EVAL_DATA_PATH"] = str(eval_data_path)
    env["TRAFFIXQWEN_V5_DEBUG_LOSS"] = "1" if args.debug_loss else "0"
    env["TRAFFIXQWEN_V5_DEBUG_BATCH_SIZE"] = str(args.per_device_train_batch_size)
    env["TRAFFIXQWEN_V5_IMAGE_TOKEN_COST"] = str(args.image_token_cost)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    trainer_state = load_trainer_state(trainer_output_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    copy_trainer_state(trainer_output_dir, logs_dir)
    train_summary = write_loss_artifacts(trainer_state, logs_dir, plots_dir)
    best_summary = finalize_best_checkpoint(trainer_output_dir, output_dir)
    dump_json(logs_dir / "best_checkpoint.json", best_summary)
    dump_json(
        logs_dir / "pipeline_summary.json",
        {
            "dataset_version": str(resolved["dataset_version"]),
            "train": train_summary,
            "best_checkpoint": best_summary,
            "trainer_output_dir": str(trainer_output_dir),
            "public_output_dir": str(output_dir),
        },
    )
    cleanup_trainer_run_dir(trainer_output_dir)
    print(f"[train] finalized best checkpoint to {output_dir}")
    print(f"[train] wrote loss history to {logs_dir / 'loss_history.jsonl'}")
    print(f"[train] wrote loss plot to {plots_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
