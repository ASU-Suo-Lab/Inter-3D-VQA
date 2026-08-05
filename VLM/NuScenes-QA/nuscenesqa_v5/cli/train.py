from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nuscenesqa_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    MODEL_DEFAULTS,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from nuscenesqa_v5.data.dataset import IntersectionNuScenesQATrainDataset, collate_train
from nuscenesqa_v5.engine.training import append_jsonl, evaluate_one_epoch, finalize_training_artifacts, is_main_process, save_checkpoint, train_one_epoch
from nuscenesqa_v5.utils.dist import barrier, destroy_process_group, init_distributed
from nuscenesqa_v5.utils.io import ensure, load_json
from nuscenesqa_v5.utils.modeling import build_model_config, load_feature_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NuScenes-QA on the prepared intersection split.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size_per_gpu"])
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--log-every", type=int, default=TRAINING_DEFAULTS["log_every"])
    parser.add_argument("--eval-every", type=int, default=TRAINING_DEFAULTS["eval_every"])
    parser.add_argument("--max-question-chars", type=int, default=TRAINING_DEFAULTS["max_question_chars"])
    parser.add_argument("--max-answer-chars", type=int, default=TRAINING_DEFAULTS["max_answer_chars"])
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    ensure(torch.cuda.is_available(), "CUDA is required for training.")
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    set_seed(args.seed + rank)

    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    split_manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(split_manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared split_manifest dataset_version={split_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    feature_manifest = load_feature_manifest(worktree["feature_manifest"])
    ensure(feature_manifest.get("dataset_version") == resolved["dataset_version"], f"Feature manifest dataset_version={feature_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    train_records = prepared_dir / "train.json"
    val_records = prepared_dir / "val.json"
    ensure(train_records.is_file(), f"Training records not found: {train_records}")
    ensure(val_records.is_file(), f"Validation records not found: {val_records}")

    from nuscenesqa_v5.models.mcan_generative import MCANGenerativeQA

    model_config = build_model_config(feature_manifest, MODEL_DEFAULTS)
    model = MCANGenerativeQA(model_config).to(device)
    model_without_ddp = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model_without_ddp = model.module

    train_dataset = IntersectionNuScenesQATrainDataset(
        records_path=train_records,
        feature_root=worktree["features"] / "train",
        max_question_chars=args.max_question_chars,
        max_answer_chars=args.max_answer_chars,
    )
    val_dataset = IntersectionNuScenesQATrainDataset(
        records_path=val_records,
        feature_root=worktree["features"] / "val",
        max_question_chars=args.max_question_chars,
        max_answer_chars=args.max_answer_chars,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_train,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history_jsonl = worktree["loss_history_jsonl"]
    if is_main_process():
        history_jsonl.write_text("", encoding="utf-8")

    best_eval_loss = float("inf")
    best_epoch = -1
    best_step = 0
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_stats, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            history_jsonl=history_jsonl,
            log_every=args.log_every,
            step_offset=global_step,
        )
        if (epoch + 1) % args.eval_every == 0:
            eval_stats = evaluate_one_epoch(model, val_loader, device, epoch)
            if is_main_process():
                append_jsonl(
                    history_jsonl,
                    {
                        "kind": "eval",
                        "step": global_step,
                        "epoch": epoch + 1,
                        "eval_loss": eval_stats["eval_loss"],
                        "eval_samples": eval_stats["eval_samples"],
                    },
                )
                if eval_stats["eval_loss"] <= best_eval_loss:
                    best_eval_loss = eval_stats["eval_loss"]
                    best_epoch = epoch + 1
                    best_step = global_step
                    save_checkpoint(
                        worktree["best_checkpoint"],
                        model_without_ddp,
                        optimizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                        train_args=vars(args),
                        model_config=model_config.to_dict(),
                    )
                print(
                    f"[eval] epoch={epoch + 1} step={global_step} "
                    f"train_loss={train_stats['loss']:.4f} eval_loss={eval_stats['eval_loss']:.4f} "
                    f"best_eval_loss={best_eval_loss:.4f}"
                )
            barrier()

    if is_main_process():
        with history_jsonl.open("r", encoding="utf-8") as file:
            history_rows = [json.loads(line) for line in file if line.strip()]
        best_metrics = {
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_eval_loss": best_eval_loss,
            "best_checkpoint": str(worktree["best_checkpoint"]),
        }
        summary = {
            "num_train_samples": len(train_dataset),
            "num_eval_samples": len(val_dataset),
            "batch_size_per_gpu": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "best_checkpoint": str(worktree["best_checkpoint"]),
            "best_metrics": best_metrics,
            "feature_manifest": str(worktree["feature_manifest"]),
            "model_config": model_config.to_dict(),
            "dataset_version": str(resolved["dataset_version"]),
        }
        finalize_training_artifacts(
            history_rows,
            best_metrics,
            summary,
            loss_history_json=worktree["loss_history_json"],
            loss_curve_png=worktree["loss_curve_png"],
            best_metrics_json=worktree["best_metrics_json"],
            train_summary_json=worktree["train_summary_json"],
        )

    destroy_process_group()


if __name__ == "__main__":
    main()
