from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from bevllm.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from bevllm_v5.config.common import DEFAULT_SYSTEM_PROMPT, DEFAULT_TOKENIZER_MODEL_MAX_LENGTH
from bevllm_v5.utils.io import ensure


def load_bev_tensor(feature_dir: Path, frame_token: str) -> torch.Tensor:
    tensor_path = feature_dir / f"{frame_token}.pt"
    ensure(tensor_path.is_file(), f"BEV feature tensor not found: {tensor_path}")
    bev = torch.load(tensor_path, map_location="cpu")
    if bev.ndim == 3:
        bev = bev.unsqueeze(0)
    ensure(bev.ndim == 4, f"Expected BEV tensor with 3 or 4 dims, got shape {tuple(bev.shape)}")
    return bev.detach().clone()


def build_messages(question: str, answer: str | None = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f" {DEFAULT_IMAGE_TOKEN}{question.strip()}"},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def apply_chat_template(tokenizer, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool = False) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    text = ""
    for message in messages:
        text += f"{message['role'].upper()}: {message['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    encoded = tokenizer(text, add_special_tokens=True)
    return list(encoded["input_ids"])


def build_train_features(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, torch.Tensor]:
    prefix_messages = build_messages(question, answer=None)
    full_messages = build_messages(question, answer=answer)
    prefix_ids = list(apply_chat_template(tokenizer, prefix_messages, add_generation_prompt=True))
    full_ids = list(apply_chat_template(tokenizer, full_messages, add_generation_prompt=False))
    ensure(
        full_ids[: len(prefix_ids)] == prefix_ids,
        "Chat template prefix mismatch while building supervised labels.",
    )

    if max_length is not None:
        full_ids = full_ids[:max_length]
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    prefix_len = min(len(prefix_ids), labels.shape[0])
    labels[:prefix_len] = IGNORE_INDEX
    ensure(
        bool((labels != IGNORE_INDEX).any()),
        "Sample lost all supervised answer tokens after truncation.",
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def build_inference_prompt(tokenizer, question: str, max_length: int) -> str:
    messages = build_messages(question, answer=None)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt = ""
    for message in messages:
        prompt += f"{message['role'].upper()}: {message['content']}\n"
    return prompt + "ASSISTANT: "


class StrictIntersectionV5TrainDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        feature_dir: str,
        max_length: int = DEFAULT_TOKENIZER_MODEL_MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_path = Path(data_path).resolve()
        self.feature_dir = Path(feature_dir).resolve()
        ensure(self.data_path.is_file(), f"Training data not found: {self.data_path}")
        ensure(self.feature_dir.is_dir(), f"BEV feature directory not found: {self.feature_dir}")
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, list) and payload, f"{self.data_path} does not contain a non-empty list.")
        self.samples = payload
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        encoded = build_train_features(
            self.tokenizer,
            question=str(sample["question"]),
            answer=str(sample["answer"]),
            max_length=self.max_length,
        )
        return {
            "id": str(sample["question_id"]),
            "question_id": str(sample["question_id"]),
            "frame_token": str(sample["frame_token"]),
            "scene_id": str(sample["scene_id"]),
            "chapter": str(sample["chapter"]),
            "section": str(sample["section"]),
            "subtemplate": str(sample["subtemplate"]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "bev": load_bev_tensor(self.feature_dir, str(sample["frame_token"])),
            "view": int(sample.get("view", 6)),
        }


class StrictIntersectionV5TrainCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "ids": [item["id"] for item in batch],
            "question_ids": [item["question_id"] for item in batch],
            "frame_tokens": [item["frame_token"] for item in batch],
            "scene_ids": [item["scene_id"] for item in batch],
            "subtemplates": [item["subtemplate"] for item in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "bev": torch.cat([item["bev"] for item in batch], dim=0),
            "view": [item["view"] for item in batch],
        }


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


class StrictIntersectionV5InferenceDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        feature_dir: str,
        max_length: int = DEFAULT_TOKENIZER_MODEL_MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_path = Path(data_path).resolve()
        self.feature_dir = Path(feature_dir).resolve()
        ensure(self.data_path.is_file(), f"Inference data not found: {self.data_path}")
        ensure(self.feature_dir.is_dir(), f"BEV feature directory not found: {self.feature_dir}")
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, list) and payload, f"{self.data_path} does not contain a non-empty list.")
        self.samples = payload
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "prompt": build_inference_prompt(self.tokenizer, str(sample["question"]), self.max_length),
            "bev": load_bev_tensor(self.feature_dir, str(sample["frame_token"])),
            "view": int(sample.get("view", 6)),
            "record": InferenceRecord(
                question_id=str(sample["question_id"]),
                scene_id=str(sample["scene_id"]),
                frame_token=str(sample["frame_token"]),
                question=str(sample["question"]).strip(),
                answer=str(sample["answer"]).strip(),
                chapter=str(sample["chapter"]),
                section=str(sample["section"]),
                subtemplate=str(sample["subtemplate"]),
            ),
        }


class StrictIntersectionV5InferenceCollator:
    def __init__(self, tokenizer, max_length: int = DEFAULT_TOKENIZER_MODEL_MAX_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [item["prompt"] for item in batch]
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "bev": torch.cat([item["bev"] for item in batch], dim=0),
            "view": [item["view"] for item in batch],
            "prompts": prompts,
            "records": [item["record"] for item in batch],
        }
