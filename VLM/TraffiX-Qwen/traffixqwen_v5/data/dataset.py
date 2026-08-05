from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_anyres_image, process_highres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.train import train as base_train

from .common import (
    DEFAULT_IMAGE_TOKEN_COST,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TEMPORAL_WINDOW,
    DEFAULT_VIEWS,
    ensure,
    validate_prepared_sample,
)


def ensure_multimodal_data_args_defaults(data_args: base_train.DataArguments) -> None:
    if not hasattr(data_args, "mm_use_im_start_end"):
        data_args.mm_use_im_start_end = False
    if not hasattr(data_args, "mm_use_im_patch_token"):
        data_args.mm_use_im_patch_token = False


def apply_system_prompt(conv, system_prompt: str) -> None:
    system_text = system_prompt.strip()
    ensure(system_text, "Intersection V5 sample is missing a non-empty system prompt.")
    if conv.system.startswith("<|im_start|>system"):
        conv.system = f"<|im_start|>system\n{system_text}"
    else:
        conv.system = system_text


def preprocess_qwen_intersection_train(
    conversations: Sequence[Dict[str, str]],
    system_prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_mode: str,
) -> Dict[str, torch.Tensor]:
    ensure(len(conversations) >= 2, "Intersection V5 train sample must contain at least one user turn and one assistant turn.")

    user_turn = conversations[0]
    assistant_turn = conversations[1]
    ensure(user_turn["from"] == "human", "Intersection V5 train sample must start with a human turn.")
    ensure(assistant_turn["from"] == "gpt", "Intersection V5 train sample must contain an assistant answer as the second turn.")

    prompt_conv = copy.deepcopy(conv_templates[conv_mode])
    apply_system_prompt(prompt_conv, system_prompt)
    prompt_conv.append_message(prompt_conv.roles[0], user_turn["value"])
    prompt_conv.append_message(prompt_conv.roles[1], None)
    prompt = prompt_conv.get_prompt()

    answer_text = assistant_turn["value"]
    answer_prompt = prompt + answer_text

    full_conv = copy.deepcopy(conv_templates[conv_mode])
    apply_system_prompt(full_conv, system_prompt)
    full_conv.append_message(full_conv.roles[0], user_turn["value"])
    full_conv.append_message(full_conv.roles[1], answer_text)
    full_prompt = full_conv.get_prompt()

    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    answer_ids = tokenizer_image_token(answer_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    ensure(
        input_ids.shape[0] >= answer_ids.shape[0] >= prompt_ids.shape[0],
        "Intersection V5 tokenization produced an invalid prompt/answer length ordering.",
    )
    ensure(
        torch.equal(input_ids[: answer_ids.shape[0]], answer_ids),
        "Intersection V5 train tokenization mismatch between prompt+answer and full prompt.",
    )
    ensure(
        torch.equal(answer_ids[: prompt_ids.shape[0]], prompt_ids),
        "Intersection V5 train tokenization mismatch between prompt-only and prompt+answer.",
    )

    labels = input_ids.clone()
    labels[: prompt_ids.shape[0]] = IGNORE_INDEX
    labels[answer_ids.shape[0] :] = IGNORE_INDEX

    supervised_tokens = labels[labels != IGNORE_INDEX]
    ensure(supervised_tokens.numel() > 0, "Intersection V5 train sample produced zero supervised tokens.")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt": prompt,
        "answer_text": answer_text,
        "supervised_text": tokenizer.decode(supervised_tokens, skip_special_tokens=False),
        "supervised_token_count": int(supervised_tokens.numel()),
    }


class StrictIntersectionV5TrainDataset(base_train.LazySupervisedDataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: base_train.DataArguments, model):
        ensure_multimodal_data_args_defaults(data_args)
        super().__init__(data_path=data_path, tokenizer=tokenizer, data_args=data_args, model=model)
        image_token_cost = int(os.environ.get("TRAFFIXQWEN_V5_IMAGE_TOKEN_COST", str(DEFAULT_IMAGE_TOKEN_COST)))
        ensure(image_token_cost > 0, f"Invalid TRAFFIXQWEN_V5_IMAGE_TOKEN_COST: {image_token_cost}")
        self.image_token_cost = image_token_cost
        self._length_cache: List[int] | None = None
        for sample in self.list_data_dict:
            ensure("video" not in sample, f"Intersection V5 train sample {sample.get('id')} unexpectedly contains video input.")
            validate_prepared_sample(sample, temporal_window=DEFAULT_TEMPORAL_WINDOW, views=DEFAULT_VIEWS)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._get_item(i)

    def build_supervision_debug(self, i: int = 0) -> Dict[str, Any]:
        sample = self.list_data_dict[i]
        tensors = self._build_train_tensors(sample)
        input_length = int(tensors["input_ids"].shape[0])
        supervised_ratio = tensors["supervised_token_count"] / max(input_length, 1)
        return {
            "id": sample.get("id", i),
            "prompt": tensors["prompt"],
            "answer_text": tensors["answer_text"],
            "supervised_text": tensors["supervised_text"],
            "supervised_token_count": tensors["supervised_token_count"],
            "input_length": input_length,
            "supervised_ratio": supervised_ratio,
            "multimodal_cost_length": self._compute_sample_cost_length(sample, tensors),
        }

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        ensure("image" in sample, f"Intersection V5 sample {sample.get('id')} is missing images.")
        ensure("video" not in sample, f"Intersection V5 sample {sample.get('id')} unexpectedly contains video input.")
        image_file = sample["image"]
        ensure(isinstance(image_file, list) and image_file, f"Intersection V5 sample {sample.get('id')} must contain a non-empty image list.")
        image = [self.process_image(f, "pad") for f in image_file]
        image = [[im[0], im[1], "image"] for im in image]

        data_dict = self._build_train_tensors(sample)
        data_dict = {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "id": sample.get("id", i),
            "image": image,
        }
        return data_dict

    @property
    def lengths(self) -> List[int]:
        if self._length_cache is None:
            self._length_cache = [self._compute_sample_cost_length(sample) for sample in self.list_data_dict]
        return self._length_cache

    def _build_train_tensors(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        sources = base_train.preprocess_multimodal(copy.deepcopy([sample["conversations"]]), self.data_args)
        return preprocess_qwen_intersection_train(sources[0], sample["system"], self.tokenizer, DEFAULT_PROMPT_VERSION)

    def _compute_sample_cost_length(
        self,
        sample: Mapping[str, Any],
        tensors: Dict[str, torch.Tensor] | None = None,
    ) -> int:
        if tensors is None:
            tensors = self._build_train_tensors(sample)
        image_count = len(sample.get("image", []))
        ensure(image_count > 0, f"Intersection V5 length estimation requires images for sample {sample.get('id')}.")
        tokenized_length = int(tensors["input_ids"].shape[0])
        text_token_length = max(tokenized_length - image_count, 0)
        return text_token_length + image_count * self.image_token_cost


def preprocess_qwen_prompt(
    conversations: Sequence[Dict[str, str]],
    system_prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_mode: str,
) -> Dict[str, torch.Tensor]:
    question = conversations[0]["value"]
    conv = copy.deepcopy(conv_templates[conv_mode])
    apply_system_prompt(conv, system_prompt)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "prompt": prompt,
    }


@dataclass
class InferenceRecord:
    question_id: str
    scene_id: str
    frame_token: str
    question: str
    answer: str
    subtemplate: str
    chapter: str
    section: str


class StrictIntersectionV5EvalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: base_train.DataArguments,
        conv_mode: str = DEFAULT_PROMPT_VERSION,
    ):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.conv_mode = conv_mode
        ensure_multimodal_data_args_defaults(self.data_args)
        ensure(self.data_args.image_processor is not None, "image_processor must be initialized before building eval dataset.")
        for sample in self.samples:
            ensure("video" not in sample, f"Intersection V5 eval sample {sample.get('id')} unexpectedly contains video input.")
            validate_prepared_sample(sample, temporal_window=DEFAULT_TEMPORAL_WINDOW, views=DEFAULT_VIEWS)

    def __len__(self) -> int:
        return len(self.samples)

    def process_image(self, image_file: str, overwrite_image_aspect_ratio: str | None = None):
        processor = self.data_args.image_processor
        image = Image.open(os.path.join(self.data_args.image_folder or "", image_file)).convert("RGB")
        image_size = image.size
        image_aspect_ratio = overwrite_image_aspect_ratio or self.data_args.image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_files = sample["image"]
        images = [self.process_image(image_file, "pad") for image_file in image_files]
        packed_images = [[image_tensor, image_size, modality] for image_tensor, image_size, modality in images]
        prompt_dict = preprocess_qwen_prompt(sample["conversations"], sample["system"], self.tokenizer, self.conv_mode)
        prompt_dict["input_ids"] = prompt_dict["input_ids"][0]
        prompt_dict["labels"] = prompt_dict["labels"][0]
        prompt_dict["image"] = packed_images
        prompt_dict["id"] = sample["id"]
        prompt_dict["record"] = InferenceRecord(
            question_id=str(sample["id"]),
            scene_id=str(sample["scene_id"]),
            frame_token=str(sample["frame_token"]),
            question=str(sample["question"]),
            answer=str(sample["answer"]),
            subtemplate=str(sample["subtemplate"]),
            chapter=str(sample.get("chapter", sample.get("metadata", {}).get("chapter", ""))),
            section=str(sample.get("section", sample.get("metadata", {}).get("section", ""))),
        )
        return prompt_dict


class StrictIntersectionV5InferenceCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.base_collator = base_train.DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(instances)
        batch["records"] = [instance["record"] for instance in instances]
        batch["ids"] = [instance["id"] for instance in instances]
        return batch
