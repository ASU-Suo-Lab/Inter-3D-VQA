from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from opendrivevla_v5.config.common import DEFAULT_UNIAD_CONFIG
from opendrivevla_v5.utils.dist import resolve_local_rank_device
from opendrivevla_v5.utils.io import ensure


def patch_torch_compat() -> None:
    named_parameters = torch.nn.Module.named_parameters
    if "remove_duplicate" not in inspect.signature(named_parameters).parameters:
        def _named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            return named_parameters(self, prefix=prefix, recurse=recurse)
        torch.nn.Module.named_parameters = _named_parameters


def infer_model_name(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        architectures = [item.lower() for item in config.get("architectures", [])]
        if any("qwen" in item for item in architectures):
            return "llava_qwen"
        if any("mistral" in item for item in architectures):
            return "llava_mistral"
        if any("mixtral" in item for item in architectures):
            return "llava_mixtral"
        if any("gemma" in item for item in architectures):
            return "llava_gemma"
        if any("llama" in item for item in architectures):
            return "llava_llama"
    return os.path.basename(os.path.abspath(model_path))


def read_adapter_base_model(model_path: str) -> str:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return ""
    with open(adapter_config_path, "r", encoding="utf-8") as file:
        adapter_config = json.load(file)
    return adapter_config.get("base_model_name_or_path", "")


def build_v5_overwrite_config(vision_tower_test_mode: bool = True) -> dict:
    return {
        "image_aspect_ratio": "pad",
        "vision_tower_test_mode": vision_tower_test_mode,
        "uniad_config_path": str(DEFAULT_UNIAD_CONFIG.resolve()),
    }


def resolve_extract_model_path(model_path: str) -> str:
    base_model_path = read_adapter_base_model(model_path)
    return base_model_path or model_path


def load_trainable_model(model_path: str, device: str, attn_implementation: str, vision_tower_test_mode: bool = True):
    disable_torch_init()
    patch_torch_compat()
    device_name = resolve_local_rank_device(device)
    if device_name.startswith("cuda"):
        torch.cuda.set_device(int(device_name.split(":")[1]))

    tokenizer, model, _, _ = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=infer_model_name(model_path),
        device_map=None,
        multimodal=True,
        attn_implementation=attn_implementation,
        overwrite_config=build_v5_overwrite_config(vision_tower_test_mode=vision_tower_test_mode),
    )
    model.to(device_name)
    return tokenizer, model


def load_inference_model(model_path: str, device: str, attn_implementation: str):
    from peft import PeftModel

    disable_torch_init()
    patch_torch_compat()
    device_name = resolve_local_rank_device(device)
    base_model_path = read_adapter_base_model(model_path)
    load_path = base_model_path or model_path

    tokenizer, model, _, _ = load_pretrained_model(
        load_path,
        model_base=None,
        model_name=infer_model_name(load_path),
        device_map=device_name,
        multimodal=True,
        attn_implementation=attn_implementation,
        overwrite_config=build_v5_overwrite_config(vision_tower_test_mode=True),
    )
    if base_model_path:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    model.to(device_name)
    model.eval()
    return tokenizer, model


def load_feature_extractor(model_path: str, device: str, attn_implementation: str):
    resolved_model_path = resolve_extract_model_path(model_path)
    _, model = load_trainable_model(
        model_path=resolved_model_path,
        device=device,
        attn_implementation=attn_implementation,
        vision_tower_test_mode=True,
    )
    model.eval()
    vision_tower = model.get_vision_tower()
    detector = vision_tower.vision_tower.vision_model
    detector.to(resolve_local_rank_device(device))
    detector.float()
    detector.eval()
    return model, detector


def infer_lora_target_modules(model) -> List[str]:
    preferred_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    discovered = set()
    for module_name, module in model.named_modules():
        if "vision_tower" in module_name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf_name = module_name.split(".")[-1]
        if leaf_name in preferred_suffixes:
            discovered.add(leaf_name)
    return sorted(discovered)


def enable_mm_projector_training(model) -> None:
    projector_prefixes = ("mm_projector_scene", "mm_projector_track", "mm_projector_map")
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in projector_prefixes):
            param.requires_grad = True


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    ensure(checkpoint_dir.is_dir(), f"Checkpoint directory not found: {checkpoint_dir}")
    ensure((checkpoint_dir / "config.json").is_file(), f"Missing config.json in {checkpoint_dir}")
