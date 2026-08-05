from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from senna_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_PREPARED_DIR,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_VISION_TOWER,
    DEFAULT_WORK_DIR,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from senna_v5.utils.io import ensure, dump_json, load_json
from senna_v5.utils.training_artifacts import (
    build_layout,
    cleanup_trainer_run_dir,
    copy_trainer_state,
    finalize_best_checkpoint,
    load_trainer_state,
    write_loss_artifacts,
)


TRAIN_SCRIPT = Path("/home/suolab/LLM/VLM/Senna/llava/senna/train_senna_llava_multi_img.py")
TRAIN_MODULE = "llava.senna.train_senna_llava_multi_img"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Senna V5 on prepared intersection data.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--mode", choices=["full", "lora"], default=TRAINING_DEFAULTS["mode"])
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_NAME_OR_PATH))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
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
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29553)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, trainer_run_dir: Path, train_json: Path, val_json: Path) -> list[str]:
    command = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_port={args.master_port}",
        "--module",
        TRAIN_MODULE,
        "--freeze_img_adapter",
        "False",
        "--init_weight_img_adapter",
        "False",
        "--model_name_or_path",
        str(Path(args.model_path).resolve()),
        "--version",
        DEFAULT_PROMPT_VERSION,
        "--data_path",
        str(train_json),
        "--eval_data_path",
        str(val_json),
        "--vision_tower",
        str(Path(args.vision_tower).resolve()),
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "pad",
        "--bf16",
        "True",
        "--output_dir",
        str(trainer_run_dir),
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--per_device_eval_batch_size",
        str(args.eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--ddp_find_unused_parameters",
        "False",
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
        str(args.lr_scheduler_type),
        "--logging_steps",
        str(args.logging_steps),
        "--model_max_length",
        str(args.model_max_length),
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        str(args.workers),
        "--lazy_preprocess",
        "True",
        "--report_to",
        "none",
    ]
    if args.mode == "lora":
        command.extend(
            [
                "--lora_enable",
                "True",
                "--lora_r",
                str(args.lora_r),
                "--lora_alpha",
                str(args.lora_alpha),
                "--mm_projector_lr",
                str(args.mm_projector_lr),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    train_json = prepared_dir / "train.json"
    val_json = prepared_dir / "val.json"
    ensure(train_json.is_file(), f"Training data not found: {train_json}")
    ensure(val_json.is_file(), f"Validation data not found: {val_json}")
    ensure(TRAIN_SCRIPT.is_file(), f"Training script not found: {TRAIN_SCRIPT}")
    manifest_path = prepared_dir / "split_manifest.json"
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )

    paths = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    layout = build_layout(paths["best_checkpoint"])
    trainer_run_dir = layout["trainer_run_dir"]
    if trainer_run_dir.exists():
        shutil.rmtree(trainer_run_dir)

    command = build_command(args, trainer_run_dir, train_json, val_json)
    print("[senna_v5.train] command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if args.print_command:
        return

    subprocess.run(command, cwd=Path(args.model_path).resolve().parents[1], check=True)

    trainer_state = load_trainer_state(trainer_run_dir)
    best_metrics = finalize_best_checkpoint(trainer_run_dir, paths["best_checkpoint"])
    copy_trainer_state(trainer_run_dir, paths["logs"])
    write_loss_artifacts(trainer_state, paths["logs"], paths["plots"])
    dump_json(
        paths["logs"] / "dataset_version.json",
        {
            "dataset_version": str(resolved["dataset_version"]),
            "prepared_dir": str(prepared_dir),
            "work_dir": str(Path(resolved["work_dir"]).resolve()),
        },
    )
    cleanup_trainer_run_dir(trainer_run_dir)

    print(f"[senna_v5.train] finalized best checkpoint to {paths['best_checkpoint']}")
    print(f"[senna_v5.train] wrote loss history to {paths['loss_history_jsonl']}")
    print(f"[senna_v5.train] wrote loss plot to {paths['loss_curve_png']}")
    print(f"[senna_v5.train] best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
