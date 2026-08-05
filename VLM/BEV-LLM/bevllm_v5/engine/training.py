from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bevLLM")

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from bevllm_v5.utils.io import dump_json, ensure
from bevllm_v5.utils.modeling import (
    sanitize_model_config,
    sanitize_train_args,
    save_checkpoint_bundle,
    write_checkpoint_config,
)


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return tensor.item()


def reduce_sum(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.item()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    *,
    model,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler,
    history_jsonl: Path,
    global_step: int,
    log_every: int,
    max_steps: int,
) -> tuple[Dict[str, float], int, bool]:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = amp_dtype_for_device(device)

    total_loss = 0.0
    step_count = 0
    stop_training = False

    for data_iter_step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        bevs = batch["bev"].to(device, non_blocking=True)
        view = batch["view"]

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bevs=bevs,
                view=view,
                labels=labels,
            )
            loss = outputs.loss

        loss_value = float(loss.item())
        ensure(math.isfinite(loss_value), f"Loss became non-finite at epoch={epoch} step={data_iter_step}: {loss_value}")
        loss = loss / grad_accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (data_iter_step + 1) % grad_accum_steps != 0:
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        step_count += 1
        reduced_loss = reduce_mean(loss_value, device)
        total_loss += reduced_loss
        lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            append_jsonl(
                history_jsonl,
                {
                    "kind": "train",
                    "step": global_step,
                    "epoch": epoch + (data_iter_step + 1) / len(data_loader),
                    "loss": reduced_loss,
                    "learning_rate": lr,
                },
            )
            if global_step % log_every == 0:
                print(f"[train] epoch={epoch + 1} step={global_step} loss={reduced_loss:.4f} lr={lr:.6e}", flush=True)

        if max_steps > 0 and global_step >= max_steps:
            stop_training = True
            break

    avg_loss = total_loss / max(step_count, 1)
    return {"loss": avg_loss}, global_step, stop_training


@torch.no_grad()
def evaluate_one_epoch(model, data_loader: Iterable, device: torch.device) -> Dict[str, float]:
    model.eval()
    amp_dtype = amp_dtype_for_device(device)

    loss_sum = 0.0
    sample_count = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        bevs = batch["bev"].to(device, non_blocking=True)
        view = batch["view"]

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bevs=bevs,
                view=view,
                labels=labels,
            )
            loss = float(outputs.loss.item())

        batch_size = input_ids.shape[0]
        loss_sum += loss * batch_size
        sample_count += batch_size

    total_loss = reduce_sum(loss_sum, device)
    total_samples = int(reduce_sum(float(sample_count), device))
    return {
        "eval_loss": total_loss / max(total_samples, 1),
        "eval_samples": total_samples,
    }


def plot_loss_curves(history_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    train_rows = [row for row in history_rows if row.get("kind") == "train"]
    eval_rows = [row for row in history_rows if row.get("kind") == "eval"]
    plt.figure(figsize=(10, 6))
    if train_rows:
        plt.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], label="train_loss", linewidth=1.3)
    if eval_rows:
        plt.plot([row["step"] for row in eval_rows], [row["eval_loss"] for row in eval_rows], label="eval_loss", linewidth=1.8, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("BEV-LLM V5 Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def finalize_training_artifacts(
    *,
    history_rows: List[Dict[str, Any]],
    best_metrics: Dict[str, Any],
    summary: Dict[str, Any],
    loss_history_json: Path,
    loss_curve_png: Path,
    train_summary_json: Path,
) -> None:
    dump_json(loss_history_json, history_rows)
    plot_loss_curves(history_rows, loss_curve_png)
    dump_json(train_summary_json, summary)


def materialize_best_checkpoint(
    *,
    paths: Dict[str, Path],
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    model_config: Dict[str, Any],
    train_args: Dict[str, Any],
    best_metrics: Dict[str, Any],
) -> None:
    if paths["best"].exists():
        shutil.rmtree(paths["best"])
    paths["best"].mkdir(parents=True, exist_ok=True)
    save_checkpoint_bundle(
        checkpoint_path=paths["best_checkpoint"],
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        model_config=model_config,
        train_args=train_args,
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        best_metrics=best_metrics,
    )
    write_checkpoint_config(
        paths["best_config_json"],
        {
            "model_config": sanitize_model_config(model_config),
            "train_args": sanitize_train_args(train_args),
            "best_metrics": best_metrics,
        },
    )
    dump_json(paths["best_metrics_json"], best_metrics)
