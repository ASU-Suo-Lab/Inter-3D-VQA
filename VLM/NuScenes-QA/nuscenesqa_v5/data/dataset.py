from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from nuscenesqa_v5.utils.io import ensure, load_json

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
ASCII_TOKENS = [chr(code) for code in range(32, 127)]
CHAR_VOCAB = SPECIAL_TOKENS + ASCII_TOKENS
CHAR_TO_ID = {char: idx for idx, char in enumerate(CHAR_VOCAB)}
PAD_ID = CHAR_TO_ID["<pad>"]
BOS_ID = CHAR_TO_ID["<bos>"]
EOS_ID = CHAR_TO_ID["<eos>"]
UNK_ID = CHAR_TO_ID["<unk>"]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def encode_chars(text: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_text(text)
    ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    mask = np.zeros((max_length,), dtype=np.bool_)
    for index, char in enumerate(clean[:max_length]):
        ids[index] = CHAR_TO_ID.get(char, UNK_ID)
        mask[index] = True
    return ids, mask


def encode_answer(text: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_text(text)
    input_ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    target_ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    input_ids[0] = BOS_ID
    limit = max_length - 1
    encoded = [CHAR_TO_ID.get(char, UNK_ID) for char in clean[:limit]]
    if encoded:
        input_ids[1 : 1 + len(encoded)] = np.asarray(encoded[: max_length - 1], dtype=np.int64)
        target_ids[: len(encoded)] = np.asarray(encoded[:max_length], dtype=np.int64)
    eos_index = min(len(encoded), max_length - 1)
    target_ids[eos_index] = EOS_ID
    return input_ids, target_ids


def encode_answer_with_prefix(text: str, prefix: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_text(text)
    clean_prefix = clean_text(prefix)
    encoded = [CHAR_TO_ID.get(char, UNK_ID) for char in clean[: max_length - 1]]
    prefix_ids = [CHAR_TO_ID.get(char, UNK_ID) for char in clean_prefix[: len(encoded)]]
    input_ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    target_ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    input_ids[0] = BOS_ID
    if encoded:
        input_ids[1 : 1 + len(encoded)] = np.asarray(encoded, dtype=np.int64)
        target_ids[: len(encoded)] = np.asarray(encoded, dtype=np.int64)
    if prefix_ids:
        target_ids[: len(prefix_ids)] = PAD_ID
    eos_index = min(len(encoded), max_length - 1)
    target_ids[eos_index] = EOS_ID
    return input_ids, target_ids


def encode_prefix(prefix: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_text(prefix)
    ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    mask = np.zeros((max_length,), dtype=np.bool_)
    encoded = [CHAR_TO_ID.get(char, UNK_ID) for char in clean[:max_length]]
    if encoded:
        ids[: len(encoded)] = np.asarray(encoded, dtype=np.int64)
        mask[: len(encoded)] = True
    return ids, mask


def decode_answer_ids(ids: torch.Tensor | np.ndarray) -> str:
    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    chars: list[str] = []
    for value in values:
        if value == EOS_ID:
            break
        if value in {PAD_ID, BOS_ID}:
            continue
        chars.append(CHAR_VOCAB[value] if 0 <= value < len(CHAR_VOCAB) else "")
    return "".join(chars).strip()


def load_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ensure(path.is_file(), f"Feature file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    return payload["object_features"].astype(np.float32), payload["bbox_features"].astype(np.float32)


class IntersectionNuScenesQATrainDataset(Dataset):
    def __init__(
        self,
        records_path: Path,
        feature_root: Path,
        max_question_chars: int,
        max_answer_chars: int,
    ) -> None:
        self.records = load_json(records_path)
        ensure(isinstance(self.records, list) and self.records, f"No records found in {records_path}")
        self.feature_root = feature_root
        self.max_question_chars = max_question_chars
        self.max_answer_chars = max_answer_chars

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.records[index]
        ensure("supervision_answer" in row and "decoder_prefix" in row, "Prepared training rows must include supervision_answer and decoder_prefix. Re-run prepare.")
        feature_path = self.feature_root / f"{row['frame_token']}.npz"
        object_features, bbox_features = load_feature(feature_path)
        question_ids, question_mask = encode_chars(row["question"], self.max_question_chars)
        decoder_input_ids, decoder_target_ids = encode_answer_with_prefix(
            row["supervision_answer"],
            row["decoder_prefix"],
            self.max_answer_chars,
        )
        return {
            "object_features": torch.from_numpy(object_features),
            "bbox_features": torch.from_numpy(bbox_features),
            "question_ids": torch.from_numpy(question_ids),
            "question_mask": torch.from_numpy(question_mask),
            "decoder_input_ids": torch.from_numpy(decoder_input_ids),
            "decoder_target_ids": torch.from_numpy(decoder_target_ids),
        }


class IntersectionNuScenesQAEvalDataset(Dataset):
    def __init__(self, records_path: Path, feature_root: Path, max_question_chars: int) -> None:
        self.records = load_json(records_path)
        ensure(isinstance(self.records, list) and self.records, f"No records found in {records_path}")
        self.feature_root = feature_root
        self.max_question_chars = max_question_chars

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.records[index]
        ensure("decoder_prefix" in row, "Prepared eval rows must include decoder_prefix. Re-run prepare.")
        feature_path = self.feature_root / f"{row['frame_token']}.npz"
        object_features, bbox_features = load_feature(feature_path)
        question_ids, question_mask = encode_chars(row["question"], self.max_question_chars)
        decoder_prefix_ids, decoder_prefix_mask = encode_prefix(row["decoder_prefix"], self.max_question_chars)
        metadata = {
            "question_id": str(row["question_id"]),
            "scene_id": str(row["scene_id"]),
            "frame_token": str(row["frame_token"]),
            "chapter": str(row["chapter"]),
            "section": str(row["section"]),
            "subtemplate": str(row["subtemplate"]),
            "decoder_prefix": str(row["decoder_prefix"]),
            "question": str(row["question"]).strip(),
            "reference_answer": str(row["answer"]).strip(),
        }
        return {
            "object_features": torch.from_numpy(object_features),
            "bbox_features": torch.from_numpy(bbox_features),
            "question_ids": torch.from_numpy(question_ids),
            "question_mask": torch.from_numpy(question_mask),
            "decoder_prefix_ids": torch.from_numpy(decoder_prefix_ids),
            "decoder_prefix_mask": torch.from_numpy(decoder_prefix_mask),
            "metadata": metadata,
        }


def collate_train(batch: list[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    object_features = torch.stack([item["object_features"] for item in batch], dim=0)
    bbox_features = torch.stack([item["bbox_features"] for item in batch], dim=0)
    question_ids = torch.stack([item["question_ids"] for item in batch], dim=0)
    question_mask = torch.stack([item["question_mask"] for item in batch], dim=0)
    decoder_input_ids = torch.stack([item["decoder_input_ids"] for item in batch], dim=0)
    decoder_target_ids = torch.stack([item["decoder_target_ids"] for item in batch], dim=0)
    return {
        "object_features": object_features,
        "bbox_features": bbox_features,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
    }


def collate_eval(batch: list[Mapping[str, Any]]) -> Dict[str, Any]:
    object_features = torch.stack([item["object_features"] for item in batch], dim=0)
    bbox_features = torch.stack([item["bbox_features"] for item in batch], dim=0)
    question_ids = torch.stack([item["question_ids"] for item in batch], dim=0)
    question_mask = torch.stack([item["question_mask"] for item in batch], dim=0)
    decoder_prefix_ids = torch.stack([item["decoder_prefix_ids"] for item in batch], dim=0)
    decoder_prefix_mask = torch.stack([item["decoder_prefix_mask"] for item in batch], dim=0)
    metadata = [dict(item["metadata"]) for item in batch]
    return {
        "object_features": object_features,
        "bbox_features": bbox_features,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "decoder_prefix_ids": decoder_prefix_ids,
        "decoder_prefix_mask": decoder_prefix_mask,
        "metadata": metadata,
    }
