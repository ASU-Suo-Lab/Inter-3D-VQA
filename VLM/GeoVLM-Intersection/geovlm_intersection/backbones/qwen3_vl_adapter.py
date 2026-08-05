from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from geovlm_intersection.config.common import DEFAULT_QWEN3_VL_MODEL_DIR
from geovlm_intersection.data.v5_io import PreparedSample
from geovlm_intersection.prompting import build_full_prompt


@dataclass(frozen=True)
class Qwen3VLRuntime:
    model_dir: Path
    processor: object
    config: object
    processor_name: str
    config_name: str
    text_hidden_size: int
    vision_hidden_size: int
    embedding_weight_file: Path
    embedding_weight_key: str


@dataclass(frozen=True)
class Qwen3VLPreparedInputs:
    question_input_ids: torch.Tensor
    question_attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_patch_tokens: torch.Tensor
    image_patch_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_counts: tuple[int, ...]
    camera_names: tuple[str, ...]
    system_prompt: str
    user_prompt: str
    full_prompt: str
    prompt_mode: str | None
    prompt_version: str | None
    subtemplate_patch_style: str | None


@dataclass(frozen=True)
class Qwen3VLTextInputs:
    question_input_ids: torch.Tensor
    question_attention_mask: torch.Tensor
    system_prompt: str
    user_prompt: str
    full_prompt: str
    prompt_mode: str | None
    prompt_version: str | None
    subtemplate_patch_style: str | None


@dataclass(frozen=True)
class Qwen3VLQuestionEmbeddings:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    embeddings: torch.Tensor


@dataclass(frozen=True)
class Qwen3VLModelRuntime:
    base_runtime: Qwen3VLRuntime
    model: Qwen3VLForConditionalGeneration
    device: str
    torch_dtype: torch.dtype


@dataclass(frozen=True)
class Qwen3VLVisionFeatures:
    image_tokens: torch.Tensor
    image_mask: torch.Tensor
    image_token_counts: tuple[int, ...]
    per_image_feature_shapes: tuple[tuple[int, int], ...]
    hidden_size: int


def load_qwen3_vl_runtime(model_dir: Path | None = None) -> Qwen3VLRuntime:
    resolved_dir = (model_dir or DEFAULT_QWEN3_VL_MODEL_DIR).resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Missing Qwen3-VL model directory: {resolved_dir}")

    processor = AutoProcessor.from_pretrained(resolved_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(resolved_dir, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    if text_config is None or getattr(text_config, "hidden_size", None) is None:
        raise ValueError(f"Qwen3-VL config missing text_config.hidden_size: {resolved_dir}")
    if vision_config is None or getattr(vision_config, "out_hidden_size", None) is None:
        raise ValueError(f"Qwen3-VL config missing vision_config.out_hidden_size: {resolved_dir}")
    text_hidden_size = int(text_config.hidden_size)
    vision_hidden_size = int(vision_config.out_hidden_size)
    embedding_weight_key = "model.language_model.embed_tokens.weight"
    embedding_weight_file = resolved_dir / "model-00001-of-00004.safetensors"
    if not embedding_weight_file.is_file():
        raise FileNotFoundError(f"Missing Qwen3-VL embedding shard: {embedding_weight_file}")

    return Qwen3VLRuntime(
        model_dir=resolved_dir,
        processor=processor,
        config=config,
        processor_name=type(processor).__name__,
        config_name=type(config).__name__,
        text_hidden_size=text_hidden_size,
        vision_hidden_size=vision_hidden_size,
        embedding_weight_file=embedding_weight_file,
        embedding_weight_key=embedding_weight_key,
    )


def prepare_qwen3_vl_inputs(
    runtime: Qwen3VLRuntime,
    sample: PreparedSample,
    *,
    prompt_bundle: dict[str, Any] | None = None,
) -> Qwen3VLPreparedInputs:
    if prompt_bundle is None:
        system_prompt = ""
        user_prompt = sample.question
        full_prompt = sample.question
        prompt_mode = None
        prompt_version = None
        subtemplate_patch_style = None
    else:
        system_prompt, user_prompt, full_prompt = build_full_prompt(sample, prompt_bundle)
        prompt_mode = str(prompt_bundle.get("prompt_mode")) if prompt_bundle.get("prompt_mode") is not None else None
        prompt_version_raw = prompt_bundle.get("prompt_version")
        patch_style_raw = prompt_bundle.get("subtemplate_patch_style")
        prompt_version = str(prompt_version_raw) if prompt_version_raw is not None else None
        subtemplate_patch_style = str(patch_style_raw) if patch_style_raw is not None else None

    images = [Image.open(path).convert("RGB") for path in sample.image_paths]
    try:
        encoded = runtime.processor(images=images, text=[full_prompt], return_tensors="pt")
    finally:
        for image in images:
            image.close()

    required_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing = required_keys.difference(encoded.keys())
    if missing:
        raise KeyError(f"Qwen3-VL processor output missing keys: {sorted(missing)}")

    pixel_values = encoded["pixel_values"]
    image_grid_thw = encoded["image_grid_thw"]
    if pixel_values.ndim != 2:
        raise ValueError(f"Expected pixel_values to be rank-2 [tokens, dim], got shape={tuple(pixel_values.shape)}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"Expected image_grid_thw to be [num_images, 3], got shape={tuple(image_grid_thw.shape)}")

    token_counts = tuple(int(row[0] * row[1] * row[2]) for row in image_grid_thw.tolist())
    if len(token_counts) != len(sample.camera_views):
        raise ValueError(
            f"Processor returned {len(token_counts)} image grids for {len(sample.camera_views)} camera views."
        )

    max_tokens = max(token_counts)
    hidden_dim = int(pixel_values.shape[-1])
    image_patch_tokens = pixel_values.new_zeros((1, len(token_counts), max_tokens, hidden_dim))
    image_patch_mask = torch.zeros((1, len(token_counts), max_tokens), dtype=torch.bool)

    start = 0
    for image_idx, token_count in enumerate(token_counts):
        end = start + token_count
        image_slice = pixel_values[start:end]
        if image_slice.shape[0] != token_count:
            raise ValueError(
                f"Processor pixel_values split mismatch at image {image_idx}: expected {token_count}, got {image_slice.shape[0]}"
            )
        image_patch_tokens[0, image_idx, :token_count] = image_slice
        image_patch_mask[0, image_idx, :token_count] = True
        start = end
    if start != pixel_values.shape[0]:
        raise ValueError(f"Processor pixel_values length mismatch: consumed {start}, total {pixel_values.shape[0]}")

    return Qwen3VLPreparedInputs(
        question_input_ids=encoded["input_ids"],
        question_attention_mask=encoded["attention_mask"],
        pixel_values=encoded["pixel_values"],
        image_patch_tokens=image_patch_tokens,
        image_patch_mask=image_patch_mask,
        image_grid_thw=image_grid_thw,
        image_token_counts=token_counts,
        camera_names=tuple(view.image_name for view in sample.camera_views),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
        prompt_mode=prompt_mode,
        prompt_version=prompt_version,
        subtemplate_patch_style=subtemplate_patch_style,
    )


def prepare_qwen3_vl_text_inputs(
    runtime: Qwen3VLRuntime,
    sample: PreparedSample,
    *,
    prompt_bundle: dict[str, Any] | None = None,
) -> Qwen3VLTextInputs:
    if prompt_bundle is None:
        system_prompt = ""
        user_prompt = sample.question
        full_prompt = sample.question
        prompt_mode = None
        prompt_version = None
        subtemplate_patch_style = None
    else:
        system_prompt, user_prompt, full_prompt = build_full_prompt(sample, prompt_bundle)
        prompt_mode = str(prompt_bundle.get("prompt_mode")) if prompt_bundle.get("prompt_mode") is not None else None
        prompt_version_raw = prompt_bundle.get("prompt_version")
        patch_style_raw = prompt_bundle.get("subtemplate_patch_style")
        prompt_version = str(prompt_version_raw) if prompt_version_raw is not None else None
        subtemplate_patch_style = str(patch_style_raw) if patch_style_raw is not None else None

    encoded = runtime.processor(text=[full_prompt], return_tensors="pt")
    required_keys = {"input_ids", "attention_mask"}
    missing = required_keys.difference(encoded.keys())
    if missing:
        raise KeyError(f"Qwen3-VL text processor output missing keys: {sorted(missing)}")
    return Qwen3VLTextInputs(
        question_input_ids=encoded["input_ids"],
        question_attention_mask=encoded["attention_mask"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
        prompt_mode=prompt_mode,
        prompt_version=prompt_version,
        subtemplate_patch_style=subtemplate_patch_style,
    )


def embed_qwen3_vl_question_ids(
    runtime: Qwen3VLRuntime,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> Qwen3VLQuestionEmbeddings:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2 [B, T], got shape={tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    elif attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape must match input_ids shape, got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
        )

    flat_ids = input_ids.reshape(-1).to(torch.long).cpu()
    unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    with safe_open(runtime.embedding_weight_file, framework="pt", device="cpu") as shard:
        weight_slice = shard.get_slice(runtime.embedding_weight_key)
        unique_embeddings = weight_slice[unique_ids.tolist()]
    embeddings = unique_embeddings[inverse].view(*input_ids.shape, runtime.text_hidden_size)
    return Qwen3VLQuestionEmbeddings(
        input_ids=input_ids,
        attention_mask=attention_mask,
        embeddings=embeddings,
    )


def build_qwen3_vl_model_runtime(
    *,
    base_runtime: Qwen3VLRuntime | None = None,
    model_dir: Path | None = None,
    device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Qwen3VLModelRuntime:
    resolved_base = base_runtime or load_qwen3_vl_runtime(model_dir=model_dir)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL vision extraction requires CUDA, but torch.cuda.is_available() is False")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved_base.model_dir,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return Qwen3VLModelRuntime(
        base_runtime=resolved_base,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
    )


def extract_qwen3_vl_vision_features(
    runtime: Qwen3VLModelRuntime,
    prepared_inputs: Qwen3VLPreparedInputs,
) -> Qwen3VLVisionFeatures:
    pixel_values = prepared_inputs.pixel_values.to(device=runtime.device, dtype=runtime.torch_dtype)
    image_grid_thw = prepared_inputs.image_grid_thw.to(device=runtime.device)
    with torch.inference_mode():
        vision_output = runtime.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    image_embeds = vision_output.pooler_output
    if not isinstance(image_embeds, (tuple, list)):
        raise ValueError(f"Expected Qwen3-VL vision pooler_output to be list/tuple, got: {type(image_embeds).__name__}")
    if len(image_embeds) != len(prepared_inputs.camera_names):
        raise ValueError(
            f"Qwen3-VL returned {len(image_embeds)} image feature groups for {len(prepared_inputs.camera_names)} camera views"
        )

    per_image_feature_shapes = tuple(tuple(int(dim) for dim in tensor.shape) for tensor in image_embeds)
    image_token_counts = tuple(int(tensor.shape[0]) for tensor in image_embeds)
    hidden_size = int(image_embeds[0].shape[-1]) if image_embeds else runtime.base_runtime.vision_hidden_size
    max_tokens = max(image_token_counts) if image_token_counts else 0

    image_tokens = torch.zeros((1, len(image_embeds), max_tokens, hidden_size), dtype=image_embeds[0].dtype)
    image_mask = torch.zeros((1, len(image_embeds), max_tokens), dtype=torch.bool)
    for image_idx, image_tensor in enumerate(image_embeds):
        token_count = int(image_tensor.shape[0])
        image_tokens[0, image_idx, :token_count] = image_tensor.detach().cpu()
        image_mask[0, image_idx, :token_count] = True

    return Qwen3VLVisionFeatures(
        image_tokens=image_tokens,
        image_mask=image_mask,
        image_token_counts=image_token_counts,
        per_image_feature_shapes=per_image_feature_shapes,
        hidden_size=hidden_size,
    )
