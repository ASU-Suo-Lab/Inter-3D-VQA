from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from drivelm_v5.config.common import (
    DEFAULT_ADAPTER_PRETRAIN,
    DEFAULT_DATASET_VERSION,
    DEFAULT_LLAMAA_DIR,
    DEFAULT_PREPARED_DIR,
    DEFAULT_WORK_DIR,
    TRAINING_DEFAULTS,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from drivelm_v5.engine.training import (
    append_jsonl,
    evaluate_one_epoch_v5,
    finalize_training_artifacts,
    is_main_process,
    save_checkpoint,
    train_one_epoch_v5,
)
from drivelm_v5.utils.imports import add_llama_adapter_to_path
from drivelm_v5.utils.io import ensure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DriveLM V5 on the prepared SunLakes split.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--llama-dir", default=os.environ.get("DRIVELM_LLAMA_DIR", str(DEFAULT_LLAMAA_DIR)))
    parser.add_argument("--pretrained-path", default=str(DEFAULT_ADAPTER_PRETRAIN))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size_per_gpu"])
    parser.add_argument("--accum-iter", type=int, default=TRAINING_DEFAULTS["accum_iter"])
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--blr", type=float, default=TRAINING_DEFAULTS["base_learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--min-lr", type=float, default=TRAINING_DEFAULTS["min_lr"])
    parser.add_argument("--warmup-epochs", type=int, default=TRAINING_DEFAULTS["warmup_epochs"])
    parser.add_argument("--max-words", type=int, default=TRAINING_DEFAULTS["max_words"])
    parser.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--log-every", type=int, default=TRAINING_DEFAULTS["log_every"])
    parser.add_argument("--dist-on-itp", action="store_true")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def ensure_llama_dir(path: Path) -> None:
    ensure(path.is_dir(), f"LLaMA base directory not found: {path}")
    ensure((path / "tokenizer.model").is_file(), f"Missing tokenizer.model under {path}")
    ensure((path / "7B").is_dir(), f"Missing 7B directory under {path}")
    ensure((path / "7B" / "params.json").is_file(), f"Missing params.json under {path / '7B'}")
    ensure(list((path / "7B").glob("*.pth")), f"No LLaMA checkpoint shards found under {path / '7B'}")


def main() -> None:
    args = parse_args()
    add_llama_adapter_to_path()

    import util.lr_sched as lr_sched
    import util.misc as misc
    from data.dataset import FinetuneDataset, transform_train
    from llama.llama_adapter import LLaMA_adapter

    prepared_dir = Path(args.prepared_dir).resolve()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    train_config = prepared_dir / "train_data_config.yaml"
    val_config = prepared_dir / "val_data_config.yaml"
    manifest_path = prepared_dir / "split_manifest.json"
    work_dir = Path(resolved["work_dir"]).resolve()
    ensure(train_config.is_file(), f"Training data config not found: {train_config}")
    ensure(val_config.is_file(), f"Validation data config not found: {val_config}")
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    ensure(Path(args.pretrained_path).resolve().is_file(), f"Adapter checkpoint not found: {args.pretrained_path}")
    ensure_llama_dir(Path(args.llama_dir).resolve())
    ensure(torch.cuda.is_available(), "CUDA is required for DriveLM V5 training.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )

    paths = ensure_worktree_layout(work_dir)
    args.output_dir = str(paths["checkpoints"])
    args.log_dir = str(paths["logs"])
    misc.init_distributed_mode(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    llama_dir = Path(args.llama_dir).resolve()
    llama_ckpt_dir = str(llama_dir / "7B")
    llama_tokenizer_path = str(llama_dir / "tokenizer.model")
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenizer_path, max_seq_len=args.max_words, max_batch_size=max(args.batch_size, 32))
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount()
    misc.load_model(model_without_ddp, str(Path(args.pretrained_path).resolve()))

    dataset_train = FinetuneDataset(str(train_config), transform=transform_train, max_words=args.max_words, tokenizer_path=llama_tokenizer_path)
    dataset_val = FinetuneDataset(str(val_config), transform=transform_train, max_words=args.max_words, tokenizer_path=llama_tokenizer_path)

    global_rank = misc.get_rank()
    num_tasks = misc.get_world_size()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    history_jsonl = paths["loss_history_jsonl"]
    if is_main_process():
        history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        history_jsonl.write_text("", encoding="utf-8")

    best_eval_loss = float("inf")
    best_epoch = -1
    best_step = 0
    global_step = 0
    history_rows: list[dict] = []

    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)

        train_stats, global_step = train_one_epoch_v5(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            lr_sched_module=lr_sched,
            args=args,
            step_offset=global_step,
            history_jsonl=history_jsonl,
            log_every=args.log_every,
        )
        eval_stats = evaluate_one_epoch_v5(model, data_loader_val, device)
        eval_row = {
            "kind": "eval",
            "step": global_step,
            "epoch": epoch + 1,
            "eval_loss": eval_stats["eval_loss"],
            "eval_closs": eval_stats["eval_closs"],
            "eval_samples": eval_stats["eval_samples"],
        }

        if is_main_process():
            append_jsonl(history_jsonl, eval_row)
            if eval_stats["eval_loss"] <= best_eval_loss:
                best_eval_loss = eval_stats["eval_loss"]
                best_epoch = epoch + 1
                best_step = global_step
                save_checkpoint(
                    paths["best_checkpoint"],
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    epoch=epoch,
                    global_step=global_step,
                    train_args=vars(args),
                )
            print(
                f"[eval] epoch={epoch + 1} step={global_step} "
                f"eval_loss={eval_stats['eval_loss']:.4f} best_eval_loss={best_eval_loss:.4f}"
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main_process():
        with history_jsonl.open("r", encoding="utf-8") as file:
            history_rows = [json.loads(line) for line in file if line.strip()]
        best_metrics = {
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_eval_loss": best_eval_loss,
            "best_checkpoint": str(paths["best_checkpoint"]),
        }
        summary = {
            "num_train_samples": len(dataset_train),
            "num_eval_samples": len(dataset_val),
            "train_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.accum_iter,
            "epochs": args.epochs,
            "effective_batch_size": eff_batch_size,
            "best_checkpoint": str(paths["best_checkpoint"]),
            "best_metrics": best_metrics,
            "llama_dir": str(llama_dir),
            "pretrained_path": str(Path(args.pretrained_path).resolve()),
        }
        finalize_training_artifacts(
            history_rows,
            best_metrics,
            summary,
            loss_history_json=paths["loss_history_json"],
            loss_curve_png=paths["loss_curve_png"],
            best_metrics_json=paths["best_metrics_json"],
            train_summary_json=paths["train_summary_json"],
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
