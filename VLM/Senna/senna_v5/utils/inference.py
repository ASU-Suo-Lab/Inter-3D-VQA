from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token


def load_local_images(image_paths: Sequence[str]) -> list[Image.Image]:
    images = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def generate_multi_image_answer(
    *,
    prompt: str,
    image_paths: Sequence[str],
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    temperature: float,
    top_p: float | None,
    num_beams: int,
    max_new_tokens: int,
) -> str:
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    elif model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    conversation = conv.get_prompt()

    images = load_local_images(image_paths)
    image_sizes = [image.size for image in images]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    images_tensor = images_tensor.unsqueeze(0)
    input_ids = tokenizer_image_token(conversation, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
