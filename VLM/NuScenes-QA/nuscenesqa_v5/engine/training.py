from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nuscenesqa")

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from nuscenesqa_v5.utils.io import dump_json, ensure


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


def save_checkpoint(path: Path, model_without_ddp, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, train_args: Dict[str, Any], model_config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_args": train_args,
            "model_config": model_config,
        },
        path,
    )


def train_one_epoch(
    model,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    history_jsonl: Path,
    log_every: int,
    step_offset: int,
) -> tuple[Dict[str, float], int]:
    model.train(True)
    total_loss = 0.0
    step_count = 0
    global_step = step_offset
    progress = tqdm(data_loader, disable=not is_main_process(), desc=f"Train {epoch + 1}", leave=False)
    for batch in progress:
        object_features = batch["object_features"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        question_ids = batch["question_ids"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        decoder_target_ids = batch["decoder_target_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                object_features=object_features,
                bbox_features=bbox_features,
                question_ids=question_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_target_ids=decoder_target_ids,
            )
            loss = outputs["loss"]
        loss_value = float(loss.item())
        ensure(math.isfinite(loss_value), f"Training loss became non-finite at epoch={epoch}: {loss_value}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
                    "epoch": epoch + step_count / max(len(data_loader), 1),
                    "loss": reduced_loss,
                    "learning_rate": lr,
                },
            )
            progress.set_postfix(step=global_step, loss=f"{reduced_loss:.4f}", lr=f"{lr:.2e}")
            if global_step % log_every == 0:
                print(f"[train] epoch={epoch + 1} step={global_step} loss={reduced_loss:.4f} lr={lr:.6e}")

    return {"loss": total_loss / max(step_count, 1)}, global_step


@torch.no_grad()
def evaluate_one_epoch(model, data_loader: Iterable, device: torch.device, epoch: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(data_loader, disable=not is_main_process(), desc=f"Eval {epoch + 1}", leave=False)
    for batch in progress:
        object_features = batch["object_features"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        question_ids = batch["question_ids"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        decoder_target_ids = batch["decoder_target_ids"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                object_features=object_features,
                bbox_features=bbox_features,
                question_ids=question_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_target_ids=decoder_target_ids,
            )
        batch_size = int(question_ids.shape[0])
        total_loss += float(outputs["loss"].item()) * batch_size
        total_samples += batch_size
    reduced_loss = reduce_sum(total_loss, device)
    reduced_samples = int(reduce_sum(float(total_samples), device))
    return {
        "eval_loss": reduced_loss / max(reduced_samples, 1),
        "eval_samples": reduced_samples,
    }


def plot_loss_curves(history_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    train_rows = [row for row in history_rows if row.get("kind") == "train"]
    eval_rows = [row for row in history_rows if row.get("kind") == "eval"]
    plt.figure(figsize=(10, 6))
    if train_rows:
        plt.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], label="train_loss", linewidth=1.2)
    if eval_rows:
        plt.plot([row["step"] for row in eval_rows], [row["eval_loss"] for row in eval_rows], label="eval_loss", linewidth=1.6, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("NuScenes-QA V5 Loss Curves")
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
