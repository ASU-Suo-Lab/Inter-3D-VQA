from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from geovlm_intersection.data import GeoVLMSupervision, supervision_to_device_dict
from geovlm_intersection.models import GeoVLMConfig, GeoVLMModel, GeoVLMLossOutput, compute_structured_losses
from geovlm_intersection.models.semantic_decoder import filter_structured_targets_for_training


def build_model_config_from_dims(
    *,
    image_token_dim: int,
    bev_token_dim: int,
    object_token_dim: int,
    question_token_dim: int,
) -> GeoVLMConfig:
    return GeoVLMConfig(
        image_token_dim=image_token_dim,
        bev_token_dim=bev_token_dim,
        object_token_dim=object_token_dim,
        question_token_dim=question_token_dim,
    )


def build_model_from_feature_dims(
    *,
    image_token_dim: int,
    bev_token_dim: int,
    object_token_dim: int,
    question_token_dim: int,
    device: str | torch.device,
) -> GeoVLMModel:
    model = GeoVLMModel(
        build_model_config_from_dims(
            image_token_dim=image_token_dim,
            bev_token_dim=bev_token_dim,
            object_token_dim=object_token_dim,
            question_token_dim=question_token_dim,
        )
    )
    return model.to(device)


def compute_batch_supervision_loss(
    outputs: dict[str, torch.Tensor],
    supervisions: list[GeoVLMSupervision],
    *,
    device: str | torch.device,
    component_weights: dict[str, float] | None = None,
    subtemplate_loss_weights: dict[str, float] | None = None,
) -> GeoVLMLossOutput:
    batch_loss_sums: list[torch.Tensor] = []
    component_sums: dict[str, torch.Tensor] = {}
    active_count = 0
    batch_size = len(supervisions)
    for index, supervision in enumerate(supervisions):
        sample_outputs = {
            key: value[index : index + 1] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
            for key, value in outputs.items()
        }
        sample_targets = filter_structured_targets_for_training(
            supervision.subtemplate,
            supervision_to_device_dict(supervision, device=device),
        )
        try:
            sample_loss = compute_structured_losses(
                sample_outputs,
                sample_targets,
                component_weights=component_weights,
            )
        except ValueError:
            continue
        sample_weight = 1.0
        if subtemplate_loss_weights:
            sample_weight = float(subtemplate_loss_weights.get(supervision.subtemplate, 1.0))
        batch_loss_sums.append(sample_loss.total_loss_sum * sample_weight)
        active_count += int(sample_loss.active_count)
        for name, value in sample_loss.components.items():
            component_sums[name] = component_sums.get(name, torch.zeros_like(value)) + (value * sample_weight)
    if not batch_loss_sums:
        zero_ref = None
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                zero_ref = value.sum() * 0.0
                break
        if zero_ref is None:
            raise ValueError("No tensor outputs available to construct zero structured loss.")
        return GeoVLMLossOutput(total_loss=zero_ref, total_loss_sum=zero_ref, active_count=0, components={})
    total_loss_sum = torch.stack(batch_loss_sums).sum()
    total_loss = total_loss_sum / float(max(1, batch_size))
    component_means = {name: value / float(max(1, batch_size)) for name, value in component_sums.items()}
    return GeoVLMLossOutput(
        total_loss=total_loss,
        total_loss_sum=total_loss_sum,
        active_count=active_count,
        components=component_means,
    )


def save_checkpoint(
    path: Path,
    *,
    model: GeoVLMModel,
    config: GeoVLMConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    dataset_version: str,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "dataset_version": dataset_version,
    }
    if extra_state:
        payload["extra_state"] = extra_state
    torch.save(payload, path)


def load_checkpoint(path: Path, *, device: str | torch.device) -> tuple[GeoVLMModel, dict[str, Any]]:
    checkpoint = torch.load(path.resolve(), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a dict: {path}")
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"Checkpoint is missing model config: {path}")
    config = GeoVLMConfig(**config_payload)
    model = GeoVLMModel(config).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint is missing model_state_dict: {path}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is structurally incompatible with the current GeoVLM model: {path}. "
            "Re-train with the current code, or load a checkpoint produced by the same model revision."
        ) from exc
    return model, checkpoint
