from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from bevllm.loader import build_model, load_checkpoint
from bevllm_v5.config.common import MODEL_DEFAULTS
from bevllm_v5.utils.io import dump_json, ensure, load_json


def sanitize_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = copy.deepcopy(model_config)
    sanitized["access_token"] = None
    return sanitized


def sanitize_train_args(train_args: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = copy.deepcopy(train_args)
    sanitized["access_token"] = None
    return sanitized


def runtime_model_config(
    *,
    model_id: str | None = None,
    access_token: str | None = None,
    cache_dir: str | None = None,
    tokenizer_model_max_length: int | None = None,
    tokenizer_padding_side: str | None = None,
    use_lora: bool | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(MODEL_DEFAULTS)
    if model_id is not None:
        config["model_id"] = model_id
    if access_token is not None:
        config["access_token"] = access_token
    if cache_dir is not None:
        config["cache_dir"] = cache_dir
    if tokenizer_model_max_length is not None:
        config["tokenizer_model_max_length"] = int(tokenizer_model_max_length)
    if tokenizer_padding_side is not None:
        config["tokenizer_padding_side"] = tokenizer_padding_side
    if use_lora is not None:
        config["use_lora"] = bool(use_lora)
    if lora_r is not None:
        config["lora_config"]["r"] = int(lora_r)
    if lora_alpha is not None:
        config["lora_config"]["lora_alpha"] = int(lora_alpha)
    if lora_dropout is not None:
        config["lora_config"]["lora_dropout"] = float(lora_dropout)
    return config


def build_runtime_model(model_config: Dict[str, Any]):
    return build_model(model_config)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint_bundle(
    checkpoint_path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_config: Dict[str, Any],
    train_args: Dict[str, Any],
    scheduler_state: Dict[str, Any] | None = None,
    scaler_state: Dict[str, Any] | None = None,
    best_metrics: Dict[str, Any] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_without_ddp = unwrap_model(model)
    sanitized_model_config = sanitize_model_config(model_config)
    sanitized_train_args = sanitize_train_args(train_args)
    payload = {
        "MODEL_STATE": model_without_ddp.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "SCHEDULER_STATE": scheduler_state,
        "SCALER_STATE": scaler_state,
        "EPOCHS_RUN": epoch,
        "GLOBAL_STEP": global_step,
        "MODEL_CONFIG": sanitized_model_config,
        "TRAIN_ARGS": sanitized_train_args,
        "BEST_METRICS": best_metrics or {},
    }
    torch.save(payload, checkpoint_path)


def load_runtime_checkpoint(model, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    info = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    info["global_step"] = checkpoint.get("GLOBAL_STEP")
    info["model_config"] = checkpoint.get("MODEL_CONFIG")
    info["train_args"] = checkpoint.get("TRAIN_ARGS")
    info["best_metrics"] = checkpoint.get("BEST_METRICS", {})
    return info


def resolve_checkpoint_artifacts(path_like: str | Path) -> Tuple[Path, Path]:
    path = Path(path_like).resolve()
    if path.is_dir():
        checkpoint_path = path / "checkpoint.pt"
        config_path = path / "config.json"
        ensure(checkpoint_path.is_file(), f"Missing checkpoint.pt under {path}")
        ensure(config_path.is_file(), f"Missing config.json under {path}")
        return checkpoint_path, config_path
    ensure(path.is_file(), f"Checkpoint not found: {path}")
    config_path = path.with_name("config.json")
    ensure(config_path.is_file(), f"Missing config.json next to checkpoint file: {config_path}")
    return path, config_path


def load_checkpoint_config(path_like: str | Path) -> Dict[str, Any]:
    _checkpoint_path, config_path = resolve_checkpoint_artifacts(path_like)
    payload = load_json(config_path)
    ensure(isinstance(payload, dict) and "model_config" in payload, f"Invalid checkpoint config payload: {config_path}")
    return payload


def write_checkpoint_config(path: Path, payload: Dict[str, Any]) -> None:
    dump_json(path, payload)
