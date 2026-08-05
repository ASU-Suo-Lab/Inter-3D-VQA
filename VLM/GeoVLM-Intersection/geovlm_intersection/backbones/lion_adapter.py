from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from geovlm_intersection.config.common import DEFAULT_LION_CHECKPOINTS, DEFAULT_LION_CONFIG, DEFAULT_LION_QUALITY, OPENPCDET_ROOT


@dataclass(frozen=True)
class LionRuntime:
    config_path: Path
    checkpoint_path: Path
    backbone_class: type
    detector_class: type
    config_name: str


def patch_transformers_generation_for_openpcdet() -> None:
    """
    OpenPCDet's vendored mamba_ssm expects generation output classes that were
    removed in newer transformers releases. We provide a minimal runtime alias
    inside GeoVLM instead of modifying OpenPCDet itself.
    """

    generation_module = importlib.import_module("transformers.generation")
    utils_module = importlib.import_module("transformers.generation.utils")
    output_cls = getattr(utils_module, "GenerateDecoderOnlyOutput")
    if not hasattr(generation_module, "GreedySearchDecoderOnlyOutput"):
        setattr(generation_module, "GreedySearchDecoderOnlyOutput", output_cls)
    if not hasattr(generation_module, "SampleDecoderOnlyOutput"):
        setattr(generation_module, "SampleDecoderOnlyOutput", output_cls)


def _ensure_openpcdet_import_path() -> None:
    root = str(OPENPCDET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"LION config must be a YAML mapping: {path}")
    return data


def load_lion_runtime(quality: str = DEFAULT_LION_QUALITY) -> LionRuntime:
    quality_key = quality.strip().lower()
    if quality_key not in DEFAULT_LION_CHECKPOINTS:
        supported = ", ".join(sorted(DEFAULT_LION_CHECKPOINTS))
        raise ValueError(f"Unsupported LION checkpoint quality: {quality}. Expected one of: {supported}")

    config_path = DEFAULT_LION_CONFIG.resolve()
    checkpoint_path = DEFAULT_LION_CHECKPOINTS[quality_key].resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LION config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing LION checkpoint: {checkpoint_path}")

    patch_transformers_generation_for_openpcdet()
    _ensure_openpcdet_import_path()

    from pcdet.models.backbones_3d.lion import LION3DBackboneOneStride
    from pcdet.models.detectors.transfusion import TransFusion

    model_cfg = _load_yaml(config_path).get("MODEL")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"LION config missing MODEL section: {config_path}")

    return LionRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        backbone_class=LION3DBackboneOneStride,
        detector_class=TransFusion,
        config_name=str(model_cfg.get("BACKBONE_3D", {}).get("NAME", "unknown")),
    )
