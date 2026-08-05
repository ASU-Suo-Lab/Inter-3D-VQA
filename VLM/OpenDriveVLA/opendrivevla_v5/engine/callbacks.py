from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openDriveVLA")

import matplotlib.pyplot as plt
from transformers import TrainerCallback

from opendrivevla_v5.config.common import (
    DEFAULT_BEST_METRICS_JSON,
    DEFAULT_BEST_DIR,
    DEFAULT_LAST_DIR,
    DEFAULT_LOSS_CURVE_PNG,
    DEFAULT_LOSS_HISTORY_JSON,
    DEFAULT_LOSS_HISTORY_JSONL,
)


class LossHistoryCallback(TrainerCallback):
    def __init__(self, logs_dir: Path, plots_dir: Path):
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.history: List[Dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        row = {"step": int(state.global_step)}
        if state.epoch is not None:
            row["epoch"] = float(state.epoch)
        row.update({key: value for key, value in logs.items() if isinstance(value, (int, float))})
        self.history.append(row)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.logs_dir / DEFAULT_LOSS_HISTORY_JSON.name
        jsonl_path = self.logs_dir / DEFAULT_LOSS_HISTORY_JSONL.name
        png_path = self.plots_dir / DEFAULT_LOSS_CURVE_PNG.name

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(self.history, file, ensure_ascii=False, indent=2)
        with jsonl_path.open("w", encoding="utf-8") as file:
            for row in self.history:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._plot(png_path)

    def _plot(self, output_path: Path) -> None:
        train_steps = [row["step"] for row in self.history if "loss" in row]
        train_losses = [row["loss"] for row in self.history if "loss" in row]
        eval_steps = [row["step"] for row in self.history if "eval_loss" in row]
        eval_losses = [row["eval_loss"] for row in self.history if "eval_loss" in row]
        lr_steps = [row["step"] for row in self.history if "learning_rate" in row]
        lrs = [row["learning_rate"] for row in self.history if "learning_rate" in row]

        plt.figure(figsize=(10, 6))
        if train_steps:
            plt.plot(train_steps, train_losses, label="train_loss", linewidth=1.5)
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="eval_loss", linewidth=1.5)
        if lr_steps:
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(lr_steps, lrs, label="lr", color="tab:green", alpha=0.5)
            ax2.set_ylabel("learning_rate")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            plt.legend(loc="upper right")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("OpenDriveVLA V5 Loss Curves")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def _copy_checkpoint_dir(src: Path, dst: Path, tokenizer) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tokenizer.save_pretrained(dst)


def materialize_best_and_last(
    trainer,
    tokenizer,
    trainer_output_dir: Path,
    checkpoint_dir: Path,
    log_history: List[Dict],
) -> Dict[str, Optional[float]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_dir = checkpoint_dir / DEFAULT_LAST_DIR.name
    best_dir = checkpoint_dir / DEFAULT_BEST_DIR.name
    best_metrics_path = checkpoint_dir / DEFAULT_BEST_METRICS_JSON.name

    final_dir = trainer_output_dir / "final-model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    eval_rows = [row for row in log_history if "eval_loss" in row]
    best_step = None
    best_eval_loss = None
    best_src = final_dir
    if eval_rows:
        best_row = min(eval_rows, key=lambda row: row["eval_loss"])
        best_step = int(best_row["step"])
        best_eval_loss = float(best_row["eval_loss"])
        candidate = trainer_output_dir / f"checkpoint-{best_step}"
        if candidate.is_dir():
            best_src = candidate

    _copy_checkpoint_dir(final_dir, last_dir, tokenizer)
    _copy_checkpoint_dir(best_src, best_dir, tokenizer)

    best_payload = {
        "best_step": best_step,
        "best_eval_loss": best_eval_loss,
        "best_checkpoint": str(best_dir),
        "last_checkpoint": str(last_dir),
    }
    with best_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(best_payload, file, ensure_ascii=False, indent=2)

    for path in trainer_output_dir.iterdir():
        if path.name.startswith("checkpoint-") or path.name == "final-model":
            shutil.rmtree(path)
    return best_payload
