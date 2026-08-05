from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from bevllm_v5.config.common import (
    DEFAULT_BEST_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_FEATURE_TRAIN_DIR,
    DEFAULT_FEATURE_VAL_DIR,
    DEFAULT_LOSS_CURVE_PNG,
    DEFAULT_LOSS_HISTORY_JSON,
    DEFAULT_LOSS_HISTORY_JSONL,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_PREPARED_DIR,
    DEFAULT_QFORMER_MODEL_ID,
    DEFAULT_TRAIN_SUMMARY_JSON,
    DEFAULT_WORK_DIR,
    MODEL_DEFAULTS,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from bevllm_v5.data.dataset import StrictIntersectionV5TrainCollator, StrictIntersectionV5TrainDataset
from bevllm_v5.engine.training import (
    append_jsonl,
    evaluate_one_epoch,
    finalize_training_artifacts,
    is_main_process,
    materialize_best_checkpoint,
    train_one_epoch,
)
from bevllm_v5.utils.dist import cleanup_distributed, init_distributed, shard_sequence, synchronize_distributed
from bevllm_v5.utils.hf import ensure_hf_runtime_ready, resolve_access_token
from bevllm_v5.utils.io import ensure, load_json
from bevllm_v5.utils.modeling import build_runtime_model, runtime_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BEV-LLM on the strict Intersection V5 BEV QA dataset.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--feature-train-dir", default=None)
    parser.add_argument("--feature-val-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--qformer-model-id", default=DEFAULT_QFORMER_MODEL_ID)
    parser.add_argument("--tokenizer-model-max-length", type=int, default=MODEL_DEFAULTS["tokenizer_model_max_length"])
    parser.add_argument("--tokenizer-padding-side", default="right")
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(args: argparse.Namespace):
    resolved_access_token = resolve_access_token(args.access_token)
    model_config = runtime_model_config(
        model_id=args.model_id,
        access_token=resolved_access_token,
        cache_dir=args.cache_dir,
        tokenizer_model_max_length=args.tokenizer_model_max_length,
        tokenizer_padding_side=args.tokenizer_padding_side,
        use_lora=not args.disable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model_config["qformer_model_id"] = args.qformer_model_id
    model, tokenizer = build_runtime_model(model_config)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer, model_config


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    primary_error: Exception | None = None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BEV-LLM V5 training.")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    try:
        resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        work_dir = Path(resolved["work_dir"]).resolve()
        args.prepared_dir = str(prepared_dir)
        args.work_dir = str(work_dir)
        if args.feature_train_dir is None:
            args.feature_train_dir = str(work_dir / "features" / "train")
        if args.feature_val_dir is None:
            args.feature_val_dir = str(work_dir / "features" / "val")
        manifest_path = prepared_dir / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )
        feature_manifest_path = work_dir / "features" / "feature_manifest.json"
        if feature_manifest_path.is_file():
            feature_manifest = load_json(feature_manifest_path)
            ensure(
                str(feature_manifest.get("dataset_version")) == str(resolved["dataset_version"]),
                f"Feature version mismatch: expected {resolved['dataset_version']}, found {feature_manifest.get('dataset_version')}",
            )
        ensure_worktree_layout(work_dir)
        paths = ensure_worktree_layout(work_dir)
        if is_main_process():
            paths["loss_history_jsonl"].unlink(missing_ok=True)

        set_seed(args.seed + rank)
        cache_root = ensure_hf_runtime_ready(args.model_id, args.access_token, args.cache_dir)
        args.cache_dir = str(cache_root)
        args.access_token = resolve_access_token(args.access_token)
        model, tokenizer, model_config = build_model_and_tokenizer(args)
        model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        train_json = Path(args.train_json).resolve() if args.train_json else prepared_dir / "train.json"
        val_json = Path(args.val_json).resolve() if args.val_json else prepared_dir / "val.json"
        train_dataset = StrictIntersectionV5TrainDataset(
            tokenizer=tokenizer,
            data_path=str(train_json),
            feature_dir=str(Path(args.feature_train_dir).resolve()),
            max_length=args.tokenizer_model_max_length,
        )
        eval_dataset = StrictIntersectionV5TrainDataset(
            tokenizer=tokenizer,
            data_path=str(val_json),
            feature_dir=str(Path(args.feature_val_dir).resolve()),
            max_length=args.tokenizer_model_max_length,
        )
        eval_dataset_total = len(eval_dataset)
        if world_size > 1:
            eval_dataset.samples = shard_sequence(eval_dataset.samples, rank, world_size)
        collator = StrictIntersectionV5TrainCollator(tokenizer.pad_token_id)

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=TRAINING_DEFAULTS["betas"],
        )
        updates_per_epoch = max(math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)), 1)
        total_updates = args.max_steps if args.max_steps > 0 else updates_per_epoch * max(args.epochs, 1)
        warmup_steps = int(total_updates * max(args.warmup_ratio, 0.0))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_updates, 1),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not torch.cuda.is_bf16_supported()))

        history_rows = []
        best_metrics = {"best_eval_loss": None, "best_epoch": None, "best_step": 0}
        global_step = 0

        train_args = vars(args).copy()
        train_args["world_size"] = world_size
        train_args["checkpoint_dir"] = str(paths["best"])

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_stats, global_step, stop_training = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_accum_steps=args.gradient_accumulation_steps,
                scaler=scaler,
                scheduler=scheduler,
                history_jsonl=paths["loss_history_jsonl"],
                global_step=global_step,
                log_every=args.log_every,
                max_steps=args.max_steps,
            )
            eval_stats = evaluate_one_epoch(model, eval_loader, device)
            if is_main_process():
                eval_row = {
                    "kind": "eval",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "eval_loss": eval_stats["eval_loss"],
                    "eval_samples": eval_stats["eval_samples"],
                }
                append_jsonl(paths["loss_history_jsonl"], eval_row)
                history_rows.append(eval_row)
                if best_metrics["best_eval_loss"] is None or eval_stats["eval_loss"] < best_metrics["best_eval_loss"]:
                    best_metrics = {
                        "best_eval_loss": eval_stats["eval_loss"],
                        "best_epoch": epoch + 1,
                        "best_step": global_step,
                        "train_loss_at_best": train_stats["loss"],
                    }
                    materialize_best_checkpoint(
                        paths=paths,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        model_config=model_config,
                        train_args=train_args,
                        best_metrics=best_metrics,
                    )
            synchronize_distributed("epoch end")
            if stop_training:
                break

        if is_main_process():
            with paths["loss_history_jsonl"].open("r", encoding="utf-8") as file:
                history_rows = [json.loads(line) for line in file if line.strip()]
            summary = {
                "epochs_requested": args.epochs,
                "epochs_completed": min(args.epochs, epoch + 1),
                "global_step": global_step,
                "train_samples": len(train_dataset),
                "eval_samples": eval_dataset_total,
                "best_checkpoint": str(paths["best"]),
                "best_metrics": best_metrics,
            }
            finalize_training_artifacts(
                history_rows=history_rows,
                best_metrics=best_metrics,
                summary=summary,
                loss_history_json=paths["loss_history_json"],
                loss_curve_png=paths["loss_curve_png"],
                train_summary_json=paths["train_summary_json"],
            )
            print(f"[train] finalized best checkpoint to {paths['best']}", flush=True)
            print(f"[train] wrote loss history to {paths['loss_history_jsonl']}", flush=True)
            print(f"[train] wrote loss plot to {paths['loss_curve_png']}", flush=True)
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        cleanup_distributed(suppress_errors=primary_error is not None)


if __name__ == "__main__":
    main()
