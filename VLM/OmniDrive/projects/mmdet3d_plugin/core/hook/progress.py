import datetime
import shutil
import sys
import time
from numbers import Number

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TrainingProgressBarHook(Hook):
    def __init__(self, iters_per_epoch, total_epochs, refresh_interval=1):
        self.iters_per_epoch = max(1, int(iters_per_epoch))
        self.total_epochs = max(1, int(total_epochs))
        self.refresh_interval = max(1, int(refresh_interval))
        self.start_time = None

    def before_run(self, runner):
        self.start_time = time.time()
        runner._train_progress_bar_active = False

    def after_train_iter(self, runner):
        if runner.rank != 0:
            return

        completed = runner.iter + 1
        if completed % self.refresh_interval != 0 and completed != runner.max_iters:
            return

        elapsed = max(time.time() - self.start_time, 1e-6)
        remaining = max(runner.max_iters - completed, 0)
        eta_seconds = int(elapsed / completed * remaining)
        epoch = min(self.total_epochs, runner.epoch + 1)
        iter_in_epoch = ((completed - 1) % self.iters_per_epoch) + 1
        loss = self._extract_loss(runner)
        lr = self._extract_lr(runner)

        message = (
            f"Train [{epoch}/{self.total_epochs}] "
            f"[{iter_in_epoch}/{self.iters_per_epoch}] "
            f"[{completed}/{runner.max_iters}] "
            f"loss: {loss:.4f} "
            f"lr: {lr:.3e} "
            f"eta: {str(datetime.timedelta(seconds=eta_seconds))}"
        )
        self._write_single_line(message)
        runner._train_progress_bar_active = True

    def after_run(self, runner):
        self._finish_line(runner)

    def _finish_line(self, runner):
        if runner.rank == 0 and getattr(runner, "_train_progress_bar_active", False):
            sys.stdout.write("\n")
            sys.stdout.flush()
            runner._train_progress_bar_active = False

    def _write_single_line(self, message):
        width = shutil.get_terminal_size((120, 20)).columns
        clipped = message[: max(1, width - 1)]
        sys.stdout.write("\r\033[K" + clipped)
        sys.stdout.flush()

    def _extract_loss(self, runner):
        log_vars = runner.outputs.get("log_vars", {}) if isinstance(runner.outputs, dict) else {}
        if "loss" in log_vars:
            return self._to_float(log_vars["loss"])

        for key, value in log_vars.items():
            if key.startswith("loss") or key.endswith("loss") or "loss" in key:
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

    def _to_float(self, value):
        if value is None:
            return 0.0
        if isinstance(value, Number):
            return float(value)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        return float(value)
