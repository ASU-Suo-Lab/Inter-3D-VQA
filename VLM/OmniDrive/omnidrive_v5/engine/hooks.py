import json
from numbers import Number
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rank(runner) -> int:
    return getattr(runner, "rank", 0)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_state(runner):
    state = getattr(runner, "_omnidrive_v5_state", None)
    if state is None:
        raise RuntimeError("OmniDrive V5 loss history state is not initialized.")
    return state


def _hook_msgs(runner):
    if runner.meta is None:
        runner.meta = {}
    return runner.meta.setdefault("hook_msgs", {})


def _ensure_state(runner, log_subdir: str, plot_subdir: str):
    if _rank(runner) != 0:
        return None
    logs_dir = Path(runner.work_dir) / log_subdir
    plots_dir = Path(runner.work_dir) / plot_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = logs_dir / "loss_history.jsonl"
    events = []
    if history_jsonl.exists():
        with history_jsonl.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    runner._omnidrive_v5_state = {
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "history_jsonl": history_jsonl,
        "history_json": logs_dir / "loss_history.json",
        "plot_path": plots_dir / "loss_curves.png",
        "events": events,
    }
    return runner._omnidrive_v5_state


def append_history_event(runner, event):
    if _rank(runner) != 0:
        return
    state = _get_state(runner)
    state["events"].append(event)
    with state["history_jsonl"].open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_history_payload(runner):
    if _rank(runner) != 0:
        return
    state = _get_state(runner)
    train_events = [event for event in state["events"] if event["kind"] == "train"]
    eval_events = [event for event in state["events"] if event["kind"] == "eval"]
    hook_msgs = _hook_msgs(runner)
    payload = {
        "summary": {
            "num_train_points": len(train_events),
            "num_eval_points": len(eval_events),
            "best_eval_loss": hook_msgs.get("best_eval_loss"),
            "best_step": hook_msgs.get("best_step"),
            "last_step": train_events[-1]["step"] if train_events else None,
        },
        "history": state["events"],
    }
    with state["history_json"].open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _plot_history(runner):
    if _rank(runner) != 0:
        return
    state = _get_state(runner)
    train_events = [event for event in state["events"] if event["kind"] == "train"]
    eval_events = [event for event in state["events"] if event["kind"] == "eval"]
    if not train_events and not eval_events:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    if train_events:
        ax1.plot(
            [event["step"] for event in train_events],
            [event["train_loss"] for event in train_events],
            label="train_loss",
            color="#1f77b4",
            linewidth=1.6,
        )
    if eval_events:
        ax1.plot(
            [event["step"] for event in eval_events],
            [event["eval_loss"] for event in eval_events],
            label="eval_loss",
            color="#d62728",
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
    ax1.set_xlabel("global step")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.25)

    if train_events:
        ax2 = ax1.twinx()
        ax2.plot(
            [event["step"] for event in train_events],
            [event["lr"] for event in train_events],
            label="lr",
            color="#2ca02c",
            linewidth=1.0,
            alpha=0.7,
        )
        ax2.set_ylabel("learning rate")
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(state["plot_path"], dpi=200)
    plt.close(fig)


@HOOKS.register_module()
class LossHistoryHook(Hook):
    def __init__(self, logging_steps=100, log_subdir="logs", plot_subdir="plots", iters_per_epoch=None):
        self.logging_steps = max(1, int(logging_steps))
        self.log_subdir = log_subdir
        self.plot_subdir = plot_subdir
        self.iters_per_epoch = int(iters_per_epoch) if iters_per_epoch else None

    def before_run(self, runner):
        _ensure_state(runner, self.log_subdir, self.plot_subdir)

    def after_train_iter(self, runner):
        if _rank(runner) != 0:
            return
        step = runner.iter + 1
        if step % self.logging_steps != 0 and step != runner.max_iters:
            return
        append_history_event(
            runner,
            {
                "kind": "train",
                "step": step,
                "epoch": self._epoch_float(runner),
                "iter": step,
                "train_loss": self._extract_loss(runner),
                "lr": self._extract_lr(runner),
            },
        )

    def after_run(self, runner):
        _write_history_payload(runner)
        _plot_history(runner)

    def _extract_loss(self, runner):
        log_vars = runner.outputs.get("log_vars", {}) if isinstance(runner.outputs, dict) else {}
        if "loss" in log_vars:
            return self._to_float(log_vars["loss"])
        for key, value in log_vars.items():
            if "loss" in key:
                return self._to_float(value)
        loss = runner.outputs.get("loss") if isinstance(runner.outputs, dict) else None
        return self._to_float(loss)

    def _extract_lr(self, runner):
        current_lr = runner.current_lr()
        if isinstance(current_lr, dict):
            current_lr = next(iter(current_lr.values()), [0.0])
        if isinstance(current_lr, (list, tuple)):
            return float(current_lr[0]) if current_lr else 0.0
        return float(current_lr)

    def _epoch_float(self, runner):
        if self.iters_per_epoch:
            return float(runner.iter + 1) / float(self.iters_per_epoch)
        epoch = getattr(runner, "epoch", 0)
        max_iters = max(int(getattr(runner, "max_iters", 1)), 1)
        return float(epoch) + float(runner.iter + 1) / float(max_iters)

    def _to_float(self, value):
        if value is None:
            return 0.0
        if isinstance(value, Number):
            return float(value)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)


class ValidationLossHook(Hook):
    def __init__(
        self,
        dataloader,
        interval=400,
        checkpoint_subdir="checkpoints",
        save_best=True,
        save_last=True,
        metric_key="eval_loss",
        rule="less",
        iters_per_epoch=None,
    ):
        self.dataloader = dataloader
        self.interval = max(1, int(interval))
        self.checkpoint_subdir = checkpoint_subdir
        self.save_best = bool(save_best)
        self.save_last = bool(save_last)
        self.metric_key = metric_key
        self.iters_per_epoch = int(iters_per_epoch) if iters_per_epoch else None
        if rule not in {"less", "greater"}:
            raise ValueError(f"Unsupported validation-loss rule: {rule}")
        self.rule = rule
        self.best_score = None
        self.best_step = None
        self.checkpoint_dir = None
        self.best_metrics_path = None

    def before_run(self, runner):
        self.checkpoint_dir = Path(runner.work_dir) / self.checkpoint_subdir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics_path = self.checkpoint_dir / "best_metrics.json"
        hook_msgs = _hook_msgs(runner)
        if hook_msgs.get("best_eval_loss") is not None:
            self.best_score = float(hook_msgs["best_eval_loss"])
            self.best_step = int(hook_msgs.get("best_step", 0))
        elif self.best_metrics_path.exists():
            with self.best_metrics_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if payload.get(self.metric_key) is not None:
                self.best_score = float(payload[self.metric_key])
                self.best_step = int(payload.get("step", 0))
                hook_msgs["best_eval_loss"] = self.best_score
                hook_msgs["best_step"] = self.best_step
                hook_msgs["best_ckpt"] = str(self.checkpoint_dir / "best.pth")

    def after_train_iter(self, runner):
        step = runner.iter + 1
        if step % self.interval != 0 and step != runner.max_iters:
            return
        eval_loss = self._evaluate_loss(runner)
        hook_msgs = _hook_msgs(runner)
        hook_msgs["last_eval_loss"] = eval_loss
        append_history_event(
            runner,
            {
                "kind": "eval",
                "step": step,
                "epoch": self._epoch_float(step),
                "iter": step,
                "eval_loss": eval_loss,
            },
        )
        if _rank(runner) == 0:
            runner.logger.info("[val_loss] step=%d eval_loss=%.6f", step, eval_loss)
            if self.save_last:
                runner.save_checkpoint(
                    str(self.checkpoint_dir),
                    filename_tmpl="last.pth",
                    create_symlink=False,
                )
                hook_msgs["last_ckpt"] = str(self.checkpoint_dir / "last.pth")
            if self.save_best and self._is_better(eval_loss):
                self.best_score = float(eval_loss)
                self.best_step = step
                hook_msgs["best_eval_loss"] = self.best_score
                hook_msgs["best_step"] = step
                runner.save_checkpoint(
                    str(self.checkpoint_dir),
                    filename_tmpl="best.pth",
                    create_symlink=False,
                )
                with self.best_metrics_path.open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "step": step,
                            "epoch": self._epoch_float(step),
                            self.metric_key: self.best_score,
                            "checkpoint": str(self.checkpoint_dir / "best.pth"),
                            "rule": self.rule,
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                hook_msgs["best_ckpt"] = str(self.checkpoint_dir / "best.pth")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _evaluate_loss(self, runner) -> float:
        if _rank(runner) == 0 and getattr(runner, "_train_progress_bar_active", False):
            print("")
            runner._train_progress_bar_active = False

        model = runner.model
        inner_model = _unwrap_model(model)
        was_training = model.training
        if hasattr(inner_model, "_reset_temporal_memory"):
            inner_model._reset_temporal_memory()
            inner_model.test_flag = False

        model.eval()
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        stats = torch.zeros(2, dtype=torch.float64, device=device)
        try:
            for data in self.dataloader:
                with torch.no_grad():
                    loss_dict = model(return_loss=True, **data)
                    loss, _ = inner_model._parse_losses(loss_dict)
                batch_size = self._infer_batch_size(data)
                stats[0] += float(loss.item()) * batch_size
                stats[1] += batch_size
        finally:
            if hasattr(inner_model, "_reset_temporal_memory"):
                inner_model._reset_temporal_memory()
                inner_model.test_flag = False
            if was_training:
                model.train()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() <= 0:
            raise RuntimeError("ValidationLossHook received an empty validation set.")
        return float((stats[0] / stats[1]).item())

    def _infer_batch_size(self, data) -> int:
        img_metas = data.get("img_metas")
        if img_metas is not None and hasattr(img_metas, "data"):
            payload = img_metas.data[0]
            if isinstance(payload, list):
                return len(payload)
        input_ids = data.get("input_ids")
        if isinstance(input_ids, list):
            return len(input_ids)
        return 1

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.rule == "less":
            return score < self.best_score
        return score > self.best_score

    def _epoch_float(self, step: int) -> float:
        if self.iters_per_epoch:
            return float(step) / float(self.iters_per_epoch)
        return float(step)
