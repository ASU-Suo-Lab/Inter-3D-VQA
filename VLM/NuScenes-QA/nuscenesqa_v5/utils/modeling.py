from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from nuscenesqa_v5.data.dataset import CHAR_VOCAB
from nuscenesqa_v5.models.mcan_generative import MCANGenerativeConfig, MCANGenerativeQA
from nuscenesqa_v5.utils.io import ensure, load_json


def build_model_config(feature_manifest: Mapping[str, int | list[str]], model_defaults: Mapping[str, int | float]) -> MCANGenerativeConfig:
    return MCANGenerativeConfig(
        object_feature_dim=int(feature_manifest["object_feature_dim"]),
        bbox_feature_dim=int(feature_manifest["bbox_feature_dim"]),
        hidden_size=int(model_defaults["hidden_size"]),
        char_embed_size=int(model_defaults["char_embed_size"]),
        bbox_embed_size=int(model_defaults["bbox_embed_size"]),
        layers=int(model_defaults["layers"]),
        multi_head=int(model_defaults["multi_head"]),
        ff_size=int(model_defaults["ff_size"]),
        dropout=float(model_defaults["dropout"]),
        flat_mlp_size=int(model_defaults["flat_mlp_size"]),
        flat_glimpses=int(model_defaults["flat_glimpses"]),
        vocab_size=len(CHAR_VOCAB),
    )


def load_feature_manifest(path: Path) -> dict:
    manifest = load_json(path)
    ensure("object_feature_dim" in manifest and "bbox_feature_dim" in manifest, f"Invalid feature manifest: {path}")
    return manifest


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[MCANGenerativeQA, dict]:
    ensure(path.is_file(), f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    model_config = MCANGenerativeConfig(**payload["model_config"])
    model = MCANGenerativeQA(model_config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    return model, payload

