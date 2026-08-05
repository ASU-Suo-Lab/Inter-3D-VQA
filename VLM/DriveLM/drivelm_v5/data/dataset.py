from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from drivelm_v5.utils.io import ensure

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def build_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


@dataclass
class InferenceRecord:
    question_id: str
    scene_id: str
    frame_token: str
    question: str
    answer: str
    chapter: str
    section: str
    subtemplate: str


class DriveLMV5InferenceDataset(Dataset):
    def __init__(self, data_path: Path, prompt_formatter):
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, list) and payload, f"{data_path} does not contain a non-empty list.")
        self.samples = payload
        self.prompt_formatter = prompt_formatter
        self.transform = build_image_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_paths = sample.get("images") or sample.get("image")
        ensure(isinstance(image_paths, list) and image_paths, f"Sample {sample.get('id')} is missing images.")
        tensors: List[torch.Tensor] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            ensure(image is not None, f"Failed to read image: {image_path}")
            tensors.append(self.transform(Image.fromarray(image)))
        return {
            "images": torch.stack(tensors, dim=0),
            "prompt": self.prompt_formatter("<image>\n" + str(sample["question"]).strip()),
            "record": InferenceRecord(
                question_id=str(sample["id"]),
                scene_id=str(sample["scene_id"]),
                frame_token=str(sample["frame_token"]),
                question=str(sample["question"]).strip(),
                answer=str(sample["answer"]).strip(),
                chapter=str(sample["chapter"]),
                section=str(sample["section"]),
                subtemplate=str(sample["subtemplate"]),
            ),
        }


class DriveLMV5InferenceCollator:
    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "images": torch.stack([item["images"] for item in batch], dim=0),
            "prompts": [item["prompt"] for item in batch],
            "records": [item["record"] for item in batch],
        }

