from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from senna_v5.utils.io import dump_json, ensure, load_json


def resolve_work_root(best_output_dir: Path) -> Path:
    if best_output_dir.parent.name == "checkpoints":
        return best_output_dir.parent.parent
    if best_output_dir.name == "checkpoints":
        return best_output_dir.parent
    return best_output_dir.parent


def build_layout(best_output_dir: Path) -> Dict[str, Path]:
    best_output_dir = best_output_dir.resolve()
    checkpoint_dir = best_output_dir.parent if best_output_dir.parent != best_output_dir else best_output_dir
    work_root = resolve_work_root(best_output_dir)
    return {
        "work_root": work_root,
        "checkpoint_dir": checkpoint_dir,
        "best_dir": best_output_dir,
        "trainer_run_dir": checkpoint_dir / f".{best_output_dir.name}_trainer",
        "logs_dir": work_root / "logs",
        "plots_dir": work_root / "plots",
        "predictions_dir": work_root / "predictions",
        "metrics_dir": work_root / "metrics",
    }


def iter_checkpoint_dirs(run_dir: Path) -> List[Path]:
    checkpoints: List[tuple[int, Path]] = []
    for candidate in run_dir.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, candidate))
    checkpoints.sort(key=lambda item: item[0])
    return [candidate for _, candidate in checkpoints]


def load_trainer_state(run_dir: Path) -> Dict[str, Any]:
    trainer_state_path = run_dir / "trainer_state.json"
    ensure(trainer_state_path.is_file(), f"Trainer state not found: {trainer_state_path}")
    trainer_state = load_json(trainer_state_path)
    ensure(isinstance(trainer_state, dict), f"Invalid trainer state payload: {trainer_state_path}")
    return trainer_state


def extract_history(trainer_state: Dict[str, Any]) -> Dict[str, Any]:
    raw_events = trainer_state.get("log_history", [])
    ensure(isinstance(raw_events, list), "trainer_state.json does not contain a valid log_history list.")
    history: List[Dict[str, Any]] = []
    train_events: List[Dict[str, Any]] = []
    eval_events: List[Dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        step = raw_event.get("step")
        if not isinstance(step, int):
            continue
        epoch = raw_event.get("epoch")
        epoch_value = float(epoch) if isinstance(epoch, (int, float)) else None
        if isinstance(raw_event.get("loss"), (int, float)):
            event = {
                "kind": "train",
                "step": step,
                "epoch": epoch_value,
                "train_loss": float(raw_event["loss"]),
                "lr": float(raw_event.get("learning_rate", 0.0)),
            }
            history.append(event)
            train_events.append(event)
        if isinstance(raw_event.get("eval_loss"), (int, float)):
            event = {
                "kind": "eval",
                "step": step,
                "epoch": epoch_value,
                "eval_loss": float(raw_event["eval_loss"]),
            }
            history.append(event)
            eval_events.append(event)
    history.sort(key=lambda item: (item["step"], 0 if item["kind"] == "train" else 1))
    best_eval = min(eval_events, key=lambda item: item["eval_loss"]) if eval_events else None
    train_summary = next(
        (
            raw_event
            for raw_event in reversed(raw_events)
            if isinstance(raw_event, dict) and isinstance(raw_event.get("train_runtime"), (int, float))
        ),
        None,
    )
    return {
        "history": history,
        "train_events": train_events,
        "eval_events": eval_events,
        "best_eval": best_eval,
        "train_summary": train_summary,
    }


def write_loss_artifacts(trainer_state: Dict[str, Any], logs_dir: Path, plots_dir: Path) -> Dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_history(trainer_state)
    history = extracted["history"]
    history_jsonl = logs_dir / "loss_history.jsonl"
    with history_jsonl.open("w", encoding="utf-8") as file:
        for event in history:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    best_eval = extracted["best_eval"]
    train_summary = extracted["train_summary"] or {}
    payload = {
        "summary": {
            "global_step": trainer_state.get("global_step"),
            "num_train_epochs": trainer_state.get("num_train_epochs"),
            "num_train_points": len(extracted["train_events"]),
            "num_eval_points": len(extracted["eval_events"]),
            "best_eval_loss": best_eval["eval_loss"] if best_eval else None,
            "best_step": best_eval["step"] if best_eval else None,
            "train_loss": float(train_summary["train_loss"]) if isinstance(train_summary.get("train_loss"), (int, float)) else None,
            "train_runtime": float(train_summary["train_runtime"]) if isinstance(train_summary.get("train_runtime"), (int, float)) else None,
        },
        "history": history,
    }
    dump_json(logs_dir / "loss_history.json", payload)
    dump_json(logs_dir / "train_summary.json", payload["summary"])
    _plot_history(extracted["train_events"], extracted["eval_events"], plots_dir / "loss_curves.png")
    return payload["summary"]


def _plot_history(train_events: List[Dict[str, Any]], eval_events: List[Dict[str, Any]], plot_path: Path) -> None:
    if not train_events and not eval_events:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 6))
    if train_events:
        ax1.plot([event["step"] for event in train_events], [event["train_loss"] for event in train_events], label="train_loss", color="#1f77b4", linewidth=1.6)
    if eval_events:
        ax1.plot([event["step"] for event in eval_events], [event["eval_loss"] for event in eval_events], label="eval_loss", color="#d62728", linewidth=1.8, marker="o", markersize=4)
    ax1.set_xlabel("global step")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.25)
    if train_events:
        ax2 = ax1.twinx()
        ax2.plot([event["step"] for event in train_events], [event.get("lr", 0.0) for event in train_events], label="lr", color="#2ca02c", linewidth=1.0, alpha=0.7)
        ax2.set_ylabel("learning rate")
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")
    else:
        ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def resolve_best_source(run_dir: Path, trainer_state: Dict[str, Any]) -> Dict[str, Any]:
    extracted = extract_history(trainer_state)
    best_eval = extracted["best_eval"]
    if best_eval is not None:
        checkpoint_dir = run_dir / f"checkpoint-{best_eval['step']}"
        ensure(checkpoint_dir.is_dir(), f"Best checkpoint directory not found: {checkpoint_dir}")
        return {
            "source_dir": checkpoint_dir,
            "best_step": best_eval["step"],
            "best_eval_loss": best_eval["eval_loss"],
            "selection": "eval_loss",
        }
    best_model_checkpoint = trainer_state.get("best_model_checkpoint")
    if isinstance(best_model_checkpoint, str) and best_model_checkpoint:
        checkpoint_dir = Path(best_model_checkpoint).resolve()
        ensure(checkpoint_dir.is_dir(), f"best_model_checkpoint does not exist: {checkpoint_dir}")
        return {
            "source_dir": checkpoint_dir,
            "best_step": _parse_step(checkpoint_dir),
            "best_eval_loss": trainer_state.get("best_metric"),
            "selection": "trainer_state.best_model_checkpoint",
        }
    checkpoint_dirs = iter_checkpoint_dirs(run_dir)
    if checkpoint_dirs:
        checkpoint_dir = checkpoint_dirs[-1]
        return {
            "source_dir": checkpoint_dir,
            "best_step": _parse_step(checkpoint_dir),
            "best_eval_loss": None,
            "selection": "latest_checkpoint",
        }
    return {
        "source_dir": run_dir,
        "best_step": int(trainer_state.get("global_step") or 0),
        "best_eval_loss": None,
        "selection": "final_output",
    }


def finalize_best_checkpoint(run_dir: Path, best_output_dir: Path) -> Dict[str, Any]:
    trainer_state = load_trainer_state(run_dir)
    selection = resolve_best_source(run_dir, trainer_state)
    source_dir = selection["source_dir"]
    ensure(source_dir.is_dir(), f"Best checkpoint source directory not found: {source_dir}")
    if best_output_dir.exists():
        shutil.rmtree(best_output_dir)
    best_output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, best_output_dir)
    _overlay_runtime_artifacts(run_dir, best_output_dir)
    _prune_public_checkpoint(best_output_dir)
    summary = {
        "source_dir": str(source_dir),
        "best_dir": str(best_output_dir),
        "best_step": selection["best_step"],
        "best_eval_loss": selection["best_eval_loss"],
        "selection": selection["selection"],
    }
    dump_json(best_output_dir.parent / "best_metrics.json", summary)
    return summary


def cleanup_trainer_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)


def copy_trainer_state(run_dir: Path, logs_dir: Path) -> None:
    trainer_state_path = run_dir / "trainer_state.json"
    if trainer_state_path.is_file():
        logs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trainer_state_path, logs_dir / "trainer_state.json")


def _parse_step(checkpoint_dir: Path) -> int:
    try:
        return int(checkpoint_dir.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _prune_public_checkpoint(best_output_dir: Path) -> None:
    removable_files = (
        "optimizer.pt",
        "optimizer.bin",
        "scheduler.pt",
        "scaler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    removable_patterns = ("rng_state*.pth", "rng_state*.pt")
    for filename in removable_files:
        candidate = best_output_dir / filename
        if candidate.exists():
            candidate.unlink()
    for pattern in removable_patterns:
        for candidate in best_output_dir.glob(pattern):
            if candidate.is_file():
                candidate.unlink()


def _overlay_runtime_artifacts(run_dir: Path, best_output_dir: Path) -> None:
    runtime_files = (
        "config.json",
        "generation_config.json",
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    for filename in runtime_files:
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, best_output_dir / filename)
