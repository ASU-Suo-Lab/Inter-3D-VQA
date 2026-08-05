from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from geovlm_intersection.backbones.qwen3_vl_adapter import (
    embed_qwen3_vl_question_ids,
    load_qwen3_vl_runtime,
    prepare_qwen3_vl_text_inputs,
)
from geovlm_intersection.config.common import (
    DATASET_VERSION_DEFAULTS,
    FEATURE_LAYOUT_VERSION,
    FRAME_ONLY_FEATURE_STORAGE,
    resolve_feature_split_alias,
    validate_features_manifest_payload,
)
from geovlm_intersection.data.targets import (
    OBJECT_CENTRIC_SUBTEMPLATES,
    OBJECT_SELECTION_MATCH_MAX_DISTANCE_M,
    SUBTEMPLATE_TO_INDEX,
    GeoVLMSupervision,
    build_structured_supervision,
    map_lion_label_to_object_type,
)
from geovlm_intersection.models.semantic_decoder import build_decoder_chat_prompt_text
from geovlm_intersection.data.v5_io import PreparedSample, build_info_index, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.prompting import load_prompt_bundle
from geovlm_intersection.rendering import render_canonical_answer_from_supervision
from geovlm_intersection.utils.io import load_json


DEFAULT_RUNTIME_QUESTION_TOKEN_BUDGET = 256


def _resolve_object_selection_index(
    *,
    supervision: GeoVLMSupervision,
    object_tokens: torch.Tensor,
    raw_object_token_count: int,
) -> int | None:
    if supervision.subtemplate not in OBJECT_CENTRIC_SUBTEMPLATES:
        return None
    if supervision.position_3d is None:
        return None
    valid_count = max(0, min(int(raw_object_token_count), int(object_tokens.shape[0])))
    if valid_count <= 0:
        return None
    centers = object_tokens[:valid_count, :2]
    labels = object_tokens[:valid_count, -1]
    target_position = centers.new_tensor([supervision.position_3d[0], supervision.position_3d[1]])
    candidate_indices = list(range(valid_count))
    if supervision.target_object_type_name:
        typed_indices = [
            index
            for index in candidate_indices
            if map_lion_label_to_object_type(float(labels[index].item())) == supervision.target_object_type_name
        ]
        if typed_indices:
            candidate_indices = typed_indices
    if not candidate_indices:
        return None
    candidate_centers = centers[candidate_indices]
    distances = torch.norm(candidate_centers - target_position.unsqueeze(0), dim=-1)
    best_local = int(distances.argmin().item())
    if float(distances[best_local].item()) > OBJECT_SELECTION_MATCH_MAX_DISTANCE_M:
        return None
    return int(candidate_indices[best_local])


def compress_sequence_tokens(tokens: torch.Tensor, token_budget: int) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected rank-2 tokens [T, D], got shape={tuple(tokens.shape)}")
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got: {token_budget}")
    if tokens.shape[0] <= token_budget:
        return tokens
    pooled = F.adaptive_avg_pool1d(tokens.transpose(0, 1).unsqueeze(0), token_budget)
    return pooled.squeeze(0).transpose(0, 1)


def compress_image_tokens(image_tokens: torch.Tensor, token_budget_per_camera: int) -> torch.Tensor:
    if image_tokens.ndim != 3:
        raise ValueError(f"Expected image_tokens [C, T, D], got shape={tuple(image_tokens.shape)}")
    compressed = [compress_sequence_tokens(camera_tokens, token_budget_per_camera) for camera_tokens in image_tokens]
    return torch.stack(compressed, dim=0)


def pad_or_truncate_tokens(tokens: torch.Tensor, token_budget: int) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected rank-2 tokens [T, D], got shape={tuple(tokens.shape)}")
    token_count, hidden = tokens.shape
    if token_count == token_budget:
        return tokens
    if token_count > token_budget:
        return tokens[:token_budget]
    padded = tokens.new_zeros((token_budget, hidden))
    padded[:token_count] = tokens
    return padded


def build_frame_storage_tensors(
    *,
    image_tokens: torch.Tensor,
    bev_tokens: torch.Tensor,
    object_tokens: torch.Tensor,
    raw_object_tokens: torch.Tensor,
    image_token_budget_per_camera: int,
    bev_token_budget: int,
    object_token_budget: int,
) -> dict[str, torch.Tensor]:
    image_tokens_comp = compress_image_tokens(image_tokens, image_token_budget_per_camera).to(dtype=torch.float16).cpu()
    bev_tokens_comp = pad_or_truncate_tokens(
        compress_sequence_tokens(bev_tokens, bev_token_budget), bev_token_budget
    ).to(dtype=torch.float16).cpu()
    object_tokens_comp = pad_or_truncate_tokens(object_tokens, object_token_budget).to(dtype=torch.float16).cpu()
    raw_object_tokens_comp = pad_or_truncate_tokens(raw_object_tokens, object_token_budget).to(dtype=torch.float32).cpu()
    return {
        "image_tokens": image_tokens_comp,
        "bev_tokens": bev_tokens_comp,
        "object_tokens": object_tokens_comp,
        "raw_object_tokens": raw_object_tokens_comp,
    }


def save_feature_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_feature_payload(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing feature payload: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Feature payload must be a dict: {resolved}")
    return payload


@dataclass(frozen=True)
class GeoVLMFeatureRecord:
    question_id: str
    subtemplate: str
    frame_feature_path: Path
    prepared_record: dict[str, Any]


class GeoVLMFeatureDataset(Dataset):
    def __init__(
        self,
        *,
        prepared_dir: Path,
        work_dir: Path,
        split: str,
        info_index: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.prepared_dir = prepared_dir.resolve()
        self.work_dir = work_dir.resolve()
        self.split = split
        self.info_index = info_index or build_info_index()
        self.qa_json_path = Path(DATASET_VERSION_DEFAULTS["v5"]["qa_json"]).resolve()
        self._prompt_bundle: dict[str, Any] | None = None
        self._qwen_runtime = None
        self.records = load_prepared_records(self.prepared_dir, split=split)
        features_manifest_path = self.work_dir / "features" / "feature_manifest.json"
        features_manifest = validate_features_manifest_payload(
            load_json(features_manifest_path),
            dataset_version="v5",
            features_dir=self.work_dir / "features",
            required_splits=(split,),
        )
        physical_split = resolve_feature_split_alias(features_manifest, split)
        self.feature_index = load_json(self.work_dir / "features" / f"{physical_split}_index.json")
        if not isinstance(self.feature_index, list):
            raise ValueError(
                f"Feature index must be a JSON list: {self.work_dir / 'features' / f'{physical_split}_index.json'}"
            )
        self.by_question_id = {str(item["question_id"]): item for item in self.feature_index}
        self.samples: list[GeoVLMFeatureRecord] = []
        for record in self.records:
            question_id = str(record["question_id"])
            feature_meta = self.by_question_id.get(question_id)
            if feature_meta is None:
                raise KeyError(f"Missing extracted feature for question_id={question_id} in split={split}")
            frame_feature_path = feature_meta.get("frame_feature_path")
            if not frame_feature_path:
                raise ValueError(
                    f"GeoVLM now requires frame-only features. Missing frame_feature_path for question_id={question_id}. "
                    "Re-run extract with the current code."
                )
            feature_storage = feature_meta.get("feature_storage")
            if feature_storage is not None and feature_storage != FRAME_ONLY_FEATURE_STORAGE:
                raise ValueError(
                    f"Feature index row for question_id={question_id} has unsupported feature_storage={feature_storage}. "
                    "Re-run extract with the current code."
                )
            if feature_meta.get("feature_layout_version") is not None and feature_meta.get("feature_layout_version") != FEATURE_LAYOUT_VERSION:
                raise ValueError(
                    f"Feature index row for question_id={question_id} has stale feature_layout_version="
                    f"{feature_meta.get('feature_layout_version')}. Re-run extract with the current code."
                )
            if feature_meta.get("feature_path") or feature_meta.get("question_feature_path"):
                raise ValueError(
                    f"GeoVLM no longer supports legacy per-question feature payloads for question_id={question_id}. "
                    "Re-run extract with the current code."
                )
            self.samples.append(
                GeoVLMFeatureRecord(
                    question_id=question_id,
                    subtemplate=str(record["subtemplate"]),
                    frame_feature_path=Path(frame_feature_path).resolve(),
                    prepared_record=record,
                )
            )

    def _ensure_prompt_bundle(self) -> dict[str, Any]:
        if self._prompt_bundle is None:
            self._prompt_bundle = load_prompt_bundle(self.qa_json_path)
        return self._prompt_bundle

    def _ensure_qwen_runtime(self):
        if self._qwen_runtime is None:
            self._qwen_runtime = load_qwen3_vl_runtime()
        return self._qwen_runtime

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_record = self.samples[index]
        feature_payload = load_feature_payload(sample_record.frame_feature_path)
        prepared_sample = resolve_prepared_sample(
            sample_record.prepared_record,
            self.info_index,
            dataset_version="v5",
            prepared_split=self.split,
            prepared_index=index,
        )
        prompt_bundle = self._ensure_prompt_bundle()
        qwen_runtime = self._ensure_qwen_runtime()
        qwen_text_inputs = prepare_qwen3_vl_text_inputs(qwen_runtime, prepared_sample, prompt_bundle=prompt_bundle)
        qwen_question = embed_qwen3_vl_question_ids(
            qwen_runtime,
            qwen_text_inputs.question_input_ids,
            qwen_text_inputs.question_attention_mask,
        )
        question_tokens = pad_or_truncate_tokens(
            compress_sequence_tokens(qwen_question.embeddings.squeeze(0), DEFAULT_RUNTIME_QUESTION_TOKEN_BUDGET),
            DEFAULT_RUNTIME_QUESTION_TOKEN_BUDGET,
        ).to(dtype=torch.float16)
        supervision = build_structured_supervision(prepared_sample)
        if supervision is None:
            raise ValueError(f"No structured supervision for question_id={sample_record.question_id}")
        object_selection_index = _resolve_object_selection_index(
            supervision=supervision,
            object_tokens=feature_payload["raw_object_tokens"].float(),
            raw_object_token_count=int(feature_payload.get("raw_object_token_count", feature_payload["raw_object_tokens"].shape[0])),
        )
        if object_selection_index is not None:
            supervision = replace(supervision, object_selection_index=object_selection_index)
        decoder_prompt_text = build_decoder_chat_prompt_text(
            tokenizer=qwen_runtime.processor.tokenizer,
            system_prompt=qwen_text_inputs.system_prompt,
            user_prompt=qwen_text_inputs.user_prompt,
        )
        canonical_answer_text = render_canonical_answer_from_supervision(
            supervision,
            fallback_answer=str(sample_record.prepared_record.get("answer", "")),
        )
        return {
            "question_id": sample_record.question_id,
            "subtemplate": sample_record.subtemplate,
            "subtemplate_index": SUBTEMPLATE_TO_INDEX[sample_record.subtemplate],
            "selection_target_index": supervision.object_selection_index if supervision.object_selection_index is not None else -1,
            "frame_feature_path": str(sample_record.frame_feature_path),
            "image_tokens": feature_payload["image_tokens"].float(),
            "bev_tokens": feature_payload["bev_tokens"].float(),
            "object_tokens": feature_payload["object_tokens"].float(),
            "raw_object_tokens": feature_payload["raw_object_tokens"].float(),
            "question_tokens": question_tokens.float(),
            "decoder_prompt_text": decoder_prompt_text,
            "answer_text": canonical_answer_text,
            "supervision": supervision,
            "prepared_record": sample_record.prepared_record,
        }


def collate_feature_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty feature batch.")
    return {
        "question_id": [item["question_id"] for item in batch],
        "subtemplate": [item["subtemplate"] for item in batch],
        "subtemplate_index": torch.tensor([item["subtemplate_index"] for item in batch], dtype=torch.long),
        "selection_target_index": torch.tensor([item["selection_target_index"] for item in batch], dtype=torch.long),
        "frame_feature_path": [item["frame_feature_path"] for item in batch],
        "prepared_record": [item["prepared_record"] for item in batch],
        "image_tokens": torch.stack([item["image_tokens"] for item in batch], dim=0),
        "bev_tokens": torch.stack([item["bev_tokens"] for item in batch], dim=0),
        "object_tokens": torch.stack([item["object_tokens"] for item in batch], dim=0),
        "raw_object_tokens": torch.stack([item["raw_object_tokens"] for item in batch], dim=0),
        "question_tokens": torch.stack([item["question_tokens"] for item in batch], dim=0),
        "decoder_prompt_text": [item["decoder_prompt_text"] for item in batch],
        "answer_text": [item["answer_text"] for item in batch],
        "supervision": [item["supervision"] for item in batch],
    }
