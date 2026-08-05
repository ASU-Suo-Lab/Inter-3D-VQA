from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-driveLM")

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from drivelm_v5.utils.io import dump_json, ensure


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


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


def save_checkpoint(
    path: Path,
    model_without_ddp,
    optimizer: torch.optim.Optimizer,
    loss_scaler,
    epoch: int,
    global_step: int,
    train_args: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "train_args": train_args,
    }
    torch.save(payload, path)


def train_one_epoch_v5(
    model,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    lr_sched_module,
    args,
    step_offset: int,
    history_jsonl: Path,
    log_every: int,
) -> tuple[Dict[str, float], int]:
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    total_loss = 0.0
    total_closs = 0.0
    step_count = 0
    global_step = step_offset

    for data_iter_step, (examples, labels, _example_mask, imgs) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            lr_sched_module.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        examples = examples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            c_loss, m_loss = model(examples, labels, imgs)
        loss = c_loss + m_loss * 0
        loss_value = float(loss.item())
        c_loss_value = float(c_loss.item())

        ensure(math.isfinite(loss_value), f"Loss became non-finite at epoch={epoch} step={data_iter_step}: {loss_value}")

        loss = loss / accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            global_step += 1
            step_count += 1
            reduced_loss = reduce_mean(loss_value, device)
            reduced_closs = reduce_mean(c_loss_value, device)
            grad_norm_value = None if grad_norm is None else reduce_mean(float(grad_norm), device)
            lr = optimizer.param_groups[0]["lr"]
            total_loss += reduced_loss
            total_closs += reduced_closs
            if is_main_process():
                append_jsonl(
                    history_jsonl,
                    {
                        "kind": "train",
                        "step": global_step,
                        "epoch": epoch + (data_iter_step + 1) / len(data_loader),
                        "loss": reduced_loss,
                        "closs": reduced_closs,
                        "grad_norm": grad_norm_value,
                        "learning_rate": lr,
                    },
                )
                if global_step % log_every == 0:
                    print(
                        f"[train] epoch={epoch + 1} step={global_step} "
                        f"loss={reduced_loss:.4f} closs={reduced_closs:.4f} lr={lr:.6e}"
                    )

    avg_loss = total_loss / max(step_count, 1)
    avg_closs = total_closs / max(step_count, 1)
    return {"loss": avg_loss, "closs": avg_closs}, global_step


@torch.no_grad()
def evaluate_one_epoch_v5(model, data_loader: Iterable, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    closs_sum = 0.0
    sample_count = 0
    for examples, labels, _example_mask, imgs in data_loader:
        examples = examples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            c_loss, m_loss = model(examples, labels, imgs)
        batch_size = examples.shape[0]
        loss_sum += float(c_loss.item()) * batch_size
        closs_sum += float(c_loss.item()) * batch_size
        sample_count += batch_size

    total_loss = reduce_sum(loss_sum, device)
    total_closs = reduce_sum(closs_sum, device)
    total_samples = int(reduce_sum(float(sample_count), device))
    return {
        "eval_loss": total_loss / max(total_samples, 1),
        "eval_closs": total_closs / max(total_samples, 1),
        "eval_samples": total_samples,
    }


def plot_loss_curves(history_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    train_rows = [row for row in history_rows if row.get("kind") == "train"]
    eval_rows = [row for row in history_rows if row.get("kind") == "eval"]
    plt.figure(figsize=(10, 6))
    if train_rows:
        plt.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], label="train_loss", linewidth=1.4)
    if eval_rows:
        plt.plot([row["step"] for row in eval_rows], [row["eval_loss"] for row in eval_rows], label="eval_loss", linewidth=1.8, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DriveLM V5 Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def finalize_training_artifacts(
    history_rows: List[Dict[str, Any]],
    best_metrics: Dict[str, Any],
    summary: Dict[str, Any],
    loss_history_json: Path,
    loss_curve_png: Path,
    best_metrics_json: Path,
    train_summary_json: Path,
) -> None:
    dump_json(loss_history_json, history_rows)
    plot_loss_curves(history_rows, loss_curve_png)
    dump_json(best_metrics_json, best_metrics)
    dump_json(train_summary_json, summary)

__all__ = [
    "append_jsonl",
    "evaluate_one_epoch_v5",
    "finalize_training_artifacts",
    "is_main_process",
    "save_checkpoint",
    "train_one_epoch_v5",
]
