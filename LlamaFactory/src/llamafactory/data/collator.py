# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


INTERSECTION_CENTER_X = 17.7
INTERSECTION_CENTER_Y = 13.2


def _load_single_lidar_object_feature(object_feature_path: str) -> np.ndarray:
    suffix = os.path.splitext(object_feature_path)[1].lower()
    if suffix == ".npy":
        object_tokens = np.load(object_feature_path)
    elif suffix == ".npz":
        archive = np.load(object_feature_path)
        if "lidar_object_tokens" in archive:
            object_tokens = archive["lidar_object_tokens"]
        else:
            first_key = next(iter(archive.files))
            object_tokens = archive[first_key]
    elif suffix == ".pt":
        object_tokens = torch.load(object_feature_path, map_location="cpu")
        if isinstance(object_tokens, dict):
            if "lidar_object_tokens" in object_tokens:
                object_tokens = object_tokens["lidar_object_tokens"]
            else:
                first_key = next(iter(object_tokens))
                object_tokens = object_tokens[first_key]
        if torch.is_tensor(object_tokens):
            object_tokens = object_tokens.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported LiDAR object feature file type: {object_feature_path}")

    object_tokens = np.asarray(object_tokens, dtype=np.float32)
    if object_tokens.ndim != 2:
        raise ValueError(
            f"LiDAR object features must be 2D [tokens, dim], got shape {object_tokens.shape} from {object_feature_path}"
        )
    return object_tokens


def _prepare_lidar_object_feature_batch(
    batch_lidar_object_features: list[list[str] | None],
    fixed_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if len(batch_lidar_object_features) == 0:
        return None, None

    if all(
        sample_object_features is None or len(sample_object_features) == 0
        for sample_object_features in batch_lidar_object_features
    ):
        return None, None

    loaded_features = []
    feature_dims = set()
    for sample_object_features in batch_lidar_object_features:
        if sample_object_features is None or len(sample_object_features) == 0:
            loaded_features.append(None)
            continue

        if len(sample_object_features) != 1:
            raise ValueError("Each sample currently supports at most one LiDAR object feature file.")

        object_tokens = _load_single_lidar_object_feature(sample_object_features[0])
        loaded_features.append(object_tokens)
        feature_dims.add(int(object_tokens.shape[1]))

    if not feature_dims:
        return None, None
    if len(feature_dims) != 1:
        raise ValueError(f"LiDAR object feature dimensions must match across the batch, got: {sorted(feature_dims)}")

    token_dim = feature_dims.pop()
    object_tensors = []
    object_masks = []
    for object_tokens in loaded_features:
        padded = np.zeros((fixed_tokens, token_dim), dtype=np.float32)
        mask = np.zeros((fixed_tokens,), dtype=np.bool_)
        if object_tokens is not None:
            token_count = int(min(object_tokens.shape[0], fixed_tokens))
            padded[:token_count] = object_tokens[:token_count]
            mask[:token_count] = True
        object_tensors.append(padded)
        object_masks.append(mask)

    return torch.from_numpy(np.stack(object_tensors, axis=0)), torch.from_numpy(np.stack(object_masks, axis=0))


def _load_single_lidar_object_geometry_feature(object_geometry_feature_path: str) -> np.ndarray:
    suffix = os.path.splitext(object_geometry_feature_path)[1].lower()
    if suffix == ".npy":
        geometry_tokens = np.load(object_geometry_feature_path)
    elif suffix == ".npz":
        archive = np.load(object_geometry_feature_path)
        if "lidar_object_geometry_tokens" in archive:
            geometry_tokens = archive["lidar_object_geometry_tokens"]
        else:
            first_key = next(iter(archive.files))
            geometry_tokens = archive[first_key]
    elif suffix == ".pt":
        geometry_tokens = torch.load(object_geometry_feature_path, map_location="cpu")
        if isinstance(geometry_tokens, dict):
            if "lidar_object_geometry_tokens" in geometry_tokens:
                geometry_tokens = geometry_tokens["lidar_object_geometry_tokens"]
            else:
                first_key = next(iter(geometry_tokens))
                geometry_tokens = geometry_tokens[first_key]
        if torch.is_tensor(geometry_tokens):
            geometry_tokens = geometry_tokens.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported LiDAR object geometry feature file type: {object_geometry_feature_path}")

    geometry_tokens = np.asarray(geometry_tokens, dtype=np.float32)
    if geometry_tokens.ndim != 2:
        raise ValueError(
            "LiDAR object geometry features must be 2D [objects, dim], "
            f"got shape {geometry_tokens.shape} from {object_geometry_feature_path}"
        )
    return geometry_tokens


def _prepare_lidar_object_geometry_feature_batch(
    batch_lidar_object_geometry_features: list[list[str] | None],
    fixed_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if len(batch_lidar_object_geometry_features) == 0:
        return None, None

    if all(
        sample_object_geometry_features is None or len(sample_object_geometry_features) == 0
        for sample_object_geometry_features in batch_lidar_object_geometry_features
    ):
        return None, None

    loaded_features = []
    feature_dims = set()
    for sample_object_geometry_features in batch_lidar_object_geometry_features:
        if sample_object_geometry_features is None or len(sample_object_geometry_features) == 0:
            loaded_features.append(None)
            continue

        if len(sample_object_geometry_features) != 1:
            raise ValueError("Each sample currently supports at most one LiDAR object geometry feature file.")

        geometry_tokens = _load_single_lidar_object_geometry_feature(sample_object_geometry_features[0])
        loaded_features.append(geometry_tokens)
        feature_dims.add(int(geometry_tokens.shape[1]))

    if not feature_dims:
        return None, None
    if len(feature_dims) != 1:
        raise ValueError(
            f"LiDAR object geometry feature dimensions must match across the batch, got: {sorted(feature_dims)}"
        )

    token_dim = feature_dims.pop()
    object_geometry_tensors = []
    object_geometry_masks = []
    for geometry_tokens in loaded_features:
        padded = np.zeros((fixed_tokens, token_dim), dtype=np.float32)
        mask = np.zeros((fixed_tokens,), dtype=np.bool_)
        if geometry_tokens is not None:
            token_count = int(min(geometry_tokens.shape[0], fixed_tokens))
            padded[:token_count] = geometry_tokens[:token_count]
            mask[:token_count] = True
        object_geometry_tensors.append(padded)
        object_geometry_masks.append(mask)

    return torch.from_numpy(np.stack(object_geometry_tensors, axis=0)), torch.from_numpy(
        np.stack(object_geometry_masks, axis=0)
    )


def _load_single_lidar_object_numeric_feature(
    object_feature_path: str,
    feature_dim: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_dim not in (10, 13):
        raise ValueError(f"LiDAR object numeric feature dim must be 10 or 13, got: {feature_dim}")

    suffix = os.path.splitext(object_feature_path)[1].lower()
    if suffix != ".pt":
        raise ValueError(
            f"Explicit LiDAR object numeric features require .pt object payloads, got: {object_feature_path}"
        )

    object_payload = torch.load(object_feature_path, map_location="cpu")
    if not isinstance(object_payload, dict):
        raise ValueError(f"LiDAR object payload must be a dict for numeric feature loading: {object_feature_path}")

    pred_boxes = object_payload.get("pred_boxes")
    pred_scores = object_payload.get("pred_scores")
    pred_labels = object_payload.get("pred_labels")
    if pred_boxes is None or pred_scores is None or pred_labels is None:
        raise ValueError(
            f"LiDAR object payload must contain pred_boxes/pred_scores/pred_labels for numeric features: {object_feature_path}"
        )

    if torch.is_tensor(pred_boxes):
        pred_boxes = pred_boxes.detach().cpu().numpy()
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.detach().cpu().numpy()
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] < 7:
        raise ValueError(
            f"LiDAR pred_boxes must be 2D [objects, >=7], got shape {pred_boxes.shape} from {object_feature_path}"
        )
    if pred_scores.ndim != 1 or pred_scores.shape[0] != pred_boxes.shape[0]:
        raise ValueError(
            f"LiDAR pred_scores must be 1D with len={pred_boxes.shape[0]}, got shape {pred_scores.shape} "
            f"from {object_feature_path}"
        )
    if pred_labels.ndim != 1 or pred_labels.shape[0] != pred_boxes.shape[0]:
        raise ValueError(
            f"LiDAR pred_labels must be 1D with len={pred_boxes.shape[0]}, got shape {pred_labels.shape} "
            f"from {object_feature_path}"
        )

    numeric_features = np.zeros((pred_boxes.shape[0], feature_dim), dtype=np.float32)
    numeric_features[:, 0:3] = pred_boxes[:, 0:3]
    numeric_features[:, 3:6] = pred_boxes[:, 3:6]
    numeric_features[:, 6] = np.sin(pred_boxes[:, 6])
    numeric_features[:, 7] = np.cos(pred_boxes[:, 6])
    numeric_features[:, 8] = pred_scores
    numeric_features[:, 9] = np.sqrt(
        np.square(pred_boxes[:, 0] - INTERSECTION_CENTER_X) + np.square(pred_boxes[:, 1] - INTERSECTION_CENTER_Y)
    )
    if feature_dim == 13 and pred_boxes.shape[1] >= 9:
        numeric_features[:, 10] = pred_boxes[:, 7]
        numeric_features[:, 11] = pred_boxes[:, 8]
        numeric_features[:, 12] = np.sqrt(np.square(pred_boxes[:, 7]) + np.square(pred_boxes[:, 8]))
    return numeric_features, pred_labels


def _prepare_lidar_object_numeric_feature_batch(
    batch_lidar_object_features: list[list[str] | None],
    fixed_tokens: int,
    feature_dim: int = 10,
) -> dict[str, torch.Tensor] | None:
    if len(batch_lidar_object_features) == 0:
        return None

    if all(
        sample_object_features is None or len(sample_object_features) == 0
        for sample_object_features in batch_lidar_object_features
    ):
        return None

    loaded_features = []
    feature_dims = set()
    for sample_object_features in batch_lidar_object_features:
        if sample_object_features is None or len(sample_object_features) == 0:
            loaded_features.append(None)
            continue
        if len(sample_object_features) != 1:
            raise ValueError("Each sample currently supports at most one LiDAR object feature file.")

        numeric_features, label_ids = _load_single_lidar_object_numeric_feature(
            sample_object_features[0],
            feature_dim=feature_dim,
        )
        loaded_features.append((numeric_features, label_ids))
        feature_dims.add(int(numeric_features.shape[1]))

    if not feature_dims:
        return None
    if len(feature_dims) != 1:
        raise ValueError(
            f"LiDAR object numeric feature dimensions must match across the batch, got: {sorted(feature_dims)}"
        )

    token_dim = feature_dims.pop()
    numeric_tensors = []
    numeric_masks = []
    numeric_label_ids = []
    for numeric_pair in loaded_features:
        padded = np.zeros((fixed_tokens, token_dim), dtype=np.float32)
        mask = np.zeros((fixed_tokens,), dtype=np.bool_)
        label_ids = np.zeros((fixed_tokens,), dtype=np.int64)
        if numeric_pair is not None:
            numeric_features, raw_label_ids = numeric_pair
            token_count = int(min(numeric_features.shape[0], fixed_tokens))
            padded[:token_count] = numeric_features[:token_count]
            label_ids[:token_count] = raw_label_ids[:token_count]
            mask[:token_count] = True
        numeric_tensors.append(padded)
        numeric_masks.append(mask)
        numeric_label_ids.append(label_ids)

    return {
        "lidar_object_numeric_features": torch.from_numpy(np.stack(numeric_tensors, axis=0)),
        "lidar_object_numeric_feature_mask": torch.from_numpy(np.stack(numeric_masks, axis=0)),
        "lidar_object_numeric_label_ids": torch.from_numpy(np.stack(numeric_label_ids, axis=0)),
    }


def _load_single_lidar_scene_feature(scene_feature_path: str) -> np.ndarray:
    suffix = os.path.splitext(scene_feature_path)[1].lower()
    if suffix == ".npy":
        scene_tokens = np.load(scene_feature_path)
    elif suffix == ".npz":
        archive = np.load(scene_feature_path)
        if "lidar_scene_tokens" in archive:
            scene_tokens = archive["lidar_scene_tokens"]
        else:
            first_key = next(iter(archive.files))
            scene_tokens = archive[first_key]
    elif suffix == ".pt":
        scene_tokens = torch.load(scene_feature_path, map_location="cpu")
        if isinstance(scene_tokens, dict):
            if "lidar_scene_tokens" in scene_tokens:
                scene_tokens = scene_tokens["lidar_scene_tokens"]
            else:
                first_key = next(iter(scene_tokens))
                scene_tokens = scene_tokens[first_key]
        if torch.is_tensor(scene_tokens):
            scene_tokens = scene_tokens.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported LiDAR scene feature file type: {scene_feature_path}")

    scene_tokens = np.asarray(scene_tokens, dtype=np.float32)
    if scene_tokens.ndim != 2:
        raise ValueError(
            f"LiDAR scene features must be 2D [tokens, dim], got shape {scene_tokens.shape} from {scene_feature_path}"
        )
    return scene_tokens


def _prepare_lidar_scene_feature_batch(
    batch_lidar_scene_features: list[list[str] | None],
    fixed_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if len(batch_lidar_scene_features) == 0:
        return None, None

    if all(
        sample_scene_features is None or len(sample_scene_features) == 0
        for sample_scene_features in batch_lidar_scene_features
    ):
        return None, None

    loaded_features = []
    feature_dims = set()
    for sample_scene_features in batch_lidar_scene_features:
        if sample_scene_features is None or len(sample_scene_features) == 0:
            loaded_features.append(None)
            continue
        if len(sample_scene_features) != 1:
            raise ValueError("Each sample currently supports at most one LiDAR scene feature file.")

        scene_tokens = _load_single_lidar_scene_feature(sample_scene_features[0])
        loaded_features.append(scene_tokens)
        feature_dims.add(int(scene_tokens.shape[1]))

    if not feature_dims:
        return None, None
    if len(feature_dims) != 1:
        raise ValueError(f"LiDAR scene feature dimensions must match across the batch, got: {sorted(feature_dims)}")

    token_dim = feature_dims.pop()
    scene_tensors = []
    scene_masks = []
    for scene_tokens in loaded_features:
        padded = np.zeros((fixed_tokens, token_dim), dtype=np.float32)
        mask = np.zeros((fixed_tokens,), dtype=np.bool_)
        if scene_tokens is not None:
            token_count = int(min(scene_tokens.shape[0], fixed_tokens))
            padded[:token_count] = scene_tokens[:token_count]
            mask[:token_count] = True
        scene_tensors.append(padded)
        scene_masks.append(mask)

    return torch.from_numpy(np.stack(scene_tensors, axis=0)), torch.from_numpy(np.stack(scene_masks, axis=0))


def _slice_mm_inputs_for_sample(
    mm_inputs: dict[str, Any],
    batch_imglens: list[int],
    batch_vidlens: list[int],
    batch_idx: int,
    images_per_subseq: Optional[list[int]] = None,
    videos_per_subseq: Optional[list[int]] = None,
    subseq_idx: Optional[int] = None,
) -> dict[str, Any]:
    r"""Slice mm_inputs for one batch sample, optionally for a single sub-sequence when packing.

    image_grid_thw / video_grid_thw have shape [num_items, 3]. Indices for sample batch_idx
    are batch_imglens[batch_idx] images and batch_vidlens[batch_idx] videos. When subseq_idx
    is given, further restrict to that sub-seq's counts via packed_*_counts.
    has_dummy_image=True means only batch[0] will be concated with fake image and no multimodal data.
    """
    image_start_idx = sum(batch_imglens[:batch_idx])
    image_end_idx = sum(batch_imglens[: batch_idx + 1])
    video_start_idx = sum(batch_vidlens[:batch_idx])
    video_end_idx = sum(batch_vidlens[: batch_idx + 1])

    if subseq_idx is not None and images_per_subseq is not None:
        image_start_idx += sum(images_per_subseq[:subseq_idx])
        image_end_idx = image_start_idx + images_per_subseq[subseq_idx]

    if subseq_idx is not None and videos_per_subseq is not None:
        video_start_idx += sum(videos_per_subseq[:subseq_idx])
        video_end_idx = video_start_idx + videos_per_subseq[subseq_idx]

    sliced_mm_inputs: dict[str, Any] = {}
    key_to_slice_meta = {
        "image_grid_thw": (image_start_idx, image_end_idx, True),
        "video_grid_thw": (video_start_idx, video_end_idx, True),
        "second_per_grid_ts": (video_start_idx, video_end_idx, False),  # qwen2.5vl
        "video_second_per_grid": (video_start_idx, video_end_idx, False),  # qwen omni
    }

    for key, (start_idx, end_idx, assign_none_when_empty) in key_to_slice_meta.items():
        if key not in mm_inputs:
            continue

        mm_value = mm_inputs[key]
        if mm_value is not None and end_idx > start_idx:
            sliced_mm_inputs[key] = mm_value[start_idx:end_idx]
        elif assign_none_when_empty:
            sliced_mm_inputs[key] = None

    return sliced_mm_inputs


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def _compute_rope_position_ids(
        self, features: dict[str, "torch.Tensor"], mm_inputs: dict[str, Any]
    ) -> None:
        r"""Compute position_ids and rope_deltas via get_rope_func for VLMs."""
        rope_index_kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "video_grid_thw": mm_inputs.get("video_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        if features["attention_mask"].sum() == 0:
            features["position_ids"] = torch.zeros((3, *features["input_ids"].shape))
            features["rope_deltas"] = torch.zeros(features["input_ids"].shape[0])
            return

        if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            video_token_id = getattr(self.model.config, "video_token_id", None)
            if image_token_id is not None or video_token_id is not None:
                mm_token_type_ids = torch.zeros_like(features["input_ids"])
                if image_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == image_token_id] = 1
                if video_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == video_token_id] = 2
                rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids

        if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
            rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
        elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
            rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

        if getattr(self.model.config, "model_type", None) in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
            rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
            feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
            if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

            features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
            features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                dim=-1
            ).unsqueeze(-1)
        else:  # for qwen vl
            features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

    def _compute_rope_position_ids_with_packing(
        self,
        features: dict[str, "torch.Tensor"],
        mm_inputs: dict[str, Any],
        packing_params_list: list[dict[str, Any] | None],
        batch_imglens: list[int],
        batch_vidlens: list[int],
        batch_audlens: list[int],
        has_dummy_image: bool,
    ) -> None:
        r"""Compute position_ids and rope_deltas per sample (or per sub-sequence when packed), then merge and validate."""
        bsz = features["input_ids"].size(0)
        seq_len = features["input_ids"].size(1)
        all_position_ids: list[torch.Tensor] = []
        all_rope_deltas: list[torch.Tensor] = []

        if has_dummy_image:
            # for [0, seq_len] = [0, unpadded_length + right_padding_length + fake_input_ids_len + collator_padding_length]
            # FIXME: maybe right_padding_length is large, with improper max_cutoff_len
            unpadded_length = int(features["attention_mask"][0].bool().sum().item())
            right_padding_length = int((packing_params_list[0] or {}).get("right_padding_length") or 0)
            fake_input_padding_length = max(0, seq_len - unpadded_length - right_padding_length)
            dummy_image_right_padding_mrope = torch.zeros((3, bsz, fake_input_padding_length))
            dummy_image_right_padding_attention_mask = torch.zeros((bsz, fake_input_padding_length))
            assert self.tokenizer.padding_side == "right", "padding_side should be right when fake image is injected"
            dummy_mm_inputs = copy.deepcopy(mm_inputs)

        for sample_idx in range(bsz):
            sample_packing = (packing_params_list[sample_idx] or {}) if sample_idx < len(packing_params_list) else {}
            sequence_boundaries = sample_packing.get("sequence_boundaries")
            num_sub_seqs = (len(sequence_boundaries) - 1) if sequence_boundaries and len(sequence_boundaries) > 1 else 1
            image_subseq_ids = sample_packing.get("image_subseq_ids") or []
            video_subseq_ids = sample_packing.get("video_subseq_ids") or []
            images_per_subseq = (
                [image_subseq_ids.count(i) for i in range(num_sub_seqs)] if image_subseq_ids and num_sub_seqs > 1 else None
            )
            videos_per_subseq = (
                [video_subseq_ids.count(i) for i in range(num_sub_seqs)] if video_subseq_ids and num_sub_seqs > 1 else None
            )
            if has_dummy_image:
                mm_inputs = {}

            if num_sub_seqs <= 1:
                sample_features = {
                    "input_ids": features["input_ids"],
                    "attention_mask": features["attention_mask"][sample_idx : sample_idx + 1],
                }
                mm_inputs_for_sample = _slice_mm_inputs_for_sample(
                    mm_inputs, batch_imglens, batch_vidlens, sample_idx=sample_idx
                )
                self._compute_rope_position_ids(sample_features, mm_inputs_for_sample)
                all_position_ids.append(sample_features["position_ids"])
                all_rope_deltas.append(sample_features["rope_deltas"])
            else:
                # when we do packing, don't need rope_deltas when training.
                sample_position_ids: list[torch.Tensor] = []
                for subseq_idx in range(num_sub_seqs):
                    subseq_start = sequence_boundaries[subseq_idx]
                    subseq_end = sequence_boundaries[subseq_idx + 1]
                    subseq_features = {
                        "input_ids": features["input_ids"][sample_idx : sample_idx + 1, subseq_start:subseq_end],
                        "attention_mask": features["attention_mask"][sample_idx : sample_idx + 1, subseq_start:subseq_end],
                    }
                    mm_inputs_for_subseq = _slice_mm_inputs_for_sample(
                        mm_inputs,
                        batch_imglens,
                        batch_vidlens,
                        sample_idx,
                        images_per_subseq,
                        videos_per_subseq,
                        subseq_idx
                    )
                    self._compute_rope_position_ids(subseq_features, mm_inputs_for_subseq)
                    sample_position_ids.append(subseq_features["position_ids"])
                all_position_ids.append(torch.cat(sample_position_ids, dim=-1))

        batch_dim_for_position_ids = 1 if all_position_ids[0].dim() == 3 else 0

        features["position_ids"] = torch.cat(all_position_ids, dim=batch_dim_for_position_ids)
        if has_dummy_image:
            mm_inputs = dummy_mm_inputs

        expected_position_ids_shape = (bsz, seq_len) if all_position_ids[0].dim() == 2 else (
            all_position_ids[0].size(0),
            bsz,
            seq_len,
        )
        # Check if position_ids shape matches expected shape.
        # for further usage, we should padding to the right when some padding token on the right.
        if has_dummy_image:
            features["position_ids"] = torch.cat([features["position_ids"], dummy_image_right_padding_mrope], dim=-1)
            features["attention_mask"] = torch.cat([features["attention_mask"], dummy_image_right_padding_attention_mask], dim=-1)

        if features["position_ids"].shape != expected_position_ids_shape:
            raise ValueError(
                "Merged position_ids shape mismatch: "
                f"got {features['position_ids'].shape}, expected {expected_position_ids_shape}."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        batch_lidar_object_features: list[list[str] | None] = []
        batch_lidar_object_geometry_features: list[list[str] | None] = []
        batch_lidar_scene_features: list[list[str] | None] = []
        batch_lidar_template_family_ids: list[int] = []
        batch_numeric_count_targets: list[list[float]] = []
        batch_numeric_count_mask: list[list[float]] = []
        batch_numeric_motion_targets: list[list[float]] = []
        batch_numeric_motion_mask: list[list[float]] = []
        batch_numeric_coord_targets: list[list[float]] = []
        batch_numeric_coord_mask: list[list[float]] = []
        packing_params_list: list[dict[str, Any] | None] = []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            lidar_object_features = feature.pop("lidar_object_features", None) or []
            lidar_object_geometry_features = feature.pop("lidar_object_geometry_features", None) or []
            lidar_scene_features = feature.pop("lidar_scene_features", None) or []
            lidar_template_family_id = feature.pop("lidar_template_family_ids", 0)
            numeric_count_targets = feature.pop("numeric_count_targets", None)
            numeric_count_mask = feature.pop("numeric_count_mask", None)
            numeric_motion_targets = feature.pop("numeric_motion_targets", None)
            numeric_motion_mask = feature.pop("numeric_motion_mask", None)
            numeric_coord_targets = feature.pop("numeric_coord_targets", None)
            numeric_coord_mask = feature.pop("numeric_coord_mask", None)
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])
            batch_lidar_object_features.append(lidar_object_features)
            batch_lidar_object_geometry_features.append(lidar_object_geometry_features)
            batch_lidar_scene_features.append(lidar_scene_features)
            batch_lidar_template_family_ids.append(int(lidar_template_family_id or 0))
            batch_numeric_count_targets.append(numeric_count_targets or [])
            batch_numeric_count_mask.append(numeric_count_mask or [])
            batch_numeric_motion_targets.append(numeric_motion_targets or [])
            batch_numeric_motion_mask.append(numeric_motion_mask or [])
            batch_numeric_coord_targets.append(numeric_coord_targets or [])
            batch_numeric_coord_mask.append(numeric_coord_mask or [])
            packing_params_list.append(feature.pop("packing_params", None))

        fake_input_ids = []
        has_dummy_image = False
        if (
            self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1
            has_dummy_image = True

        if (
            self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        bsz, seq_len = features["input_ids"].shape[:2]
        model_type = getattr(self.model.config, "model_type", None) if self.model is not None else None
        is_omni = model_type in [
            "qwen2_5_omni_thinker",
            "qwen3_omni_moe_thinker",
        ]

        if self.get_rope_func is not None:
            # for mmrope situation, we should calculate position_ids and rope_deltas per sample.
            # When neat_packing is on, each sample has packing_params; None means no packing for that sample.
            boundaries_list = [
                p.get("sequence_boundaries") if p is not None else None for p in packing_params_list
            ]
            has_packing = any(b is not None and len(b) > 2 for b in boundaries_list)
            if has_dummy_image and has_packing:
                # FIXME: too tricky, need to be refactored
                features["has_dummy_image"] = True

            # When fake image/audio was injected, sequence_boundaries no longer match the tensor; use non-packing path.
            if not has_packing:
                self._compute_rope_position_ids(features, mm_inputs)
            else:
                if is_omni:
                    raise RuntimeError("Omni models are not supported for packed sequences for now.")

                self._compute_rope_position_ids_with_packing(
                    features,
                    mm_inputs,
                    packing_params_list,
                    batch_imglens,
                    batch_vidlens,
                    batch_audlens,
                    has_dummy_image,
                )

            # For transformers compatibility, after https://github.com/huggingface/transformers/issues/39400
            if features["position_ids"].dim() == 3:
                features["position_ids"] = torch.cat(
                    [features["position_ids"][0].unsqueeze(0), features["position_ids"]], dim=0
                )

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None) in MROPE_MODELS
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)
        use_lidar_numeric_supervision = (
            getattr(self.model.config, "use_lidar_numeric_supervision", False) if self.model is not None else False
        )
        if use_lidar_numeric_supervision and batch_numeric_count_targets:
            features["numeric_count_targets"] = torch.tensor(batch_numeric_count_targets, dtype=torch.float32)
            features["numeric_count_mask"] = torch.tensor(batch_numeric_count_mask, dtype=torch.float32)
            features["numeric_motion_targets"] = torch.tensor(batch_numeric_motion_targets, dtype=torch.float32)
            features["numeric_motion_mask"] = torch.tensor(batch_numeric_motion_mask, dtype=torch.float32)
            features["numeric_coord_targets"] = torch.tensor(batch_numeric_coord_targets, dtype=torch.float32)
            features["numeric_coord_mask"] = torch.tensor(batch_numeric_coord_mask, dtype=torch.float32)

        use_lidar_modality = (
            getattr(self.model.config, "use_lidar_modality", False) if self.model is not None else False
        )
        if use_lidar_modality:
            if batch_lidar_template_family_ids:
                features["lidar_template_family_ids"] = torch.tensor(batch_lidar_template_family_ids, dtype=torch.long)
            use_lidar_object_features = (
                getattr(self.model.config, "use_lidar_object_features", False) if self.model is not None else False
            )
            use_lidar_object_geometry_features = (
                getattr(self.model.config, "use_lidar_object_geometry_features", False)
                if self.model is not None
                else False
            )
            use_lidar_object_numeric_features = (
                getattr(self.model.config, "use_lidar_object_numeric_features", False)
                if self.model is not None
                else False
            )
            use_lidar_scene_features = (
                getattr(self.model.config, "use_lidar_scene_features", False)
                if self.model is not None
                else False
            )
            lidar_object_prefix_length = getattr(self.model.config, "lidar_object_prefix_length", 16)
            if use_lidar_object_features:
                lidar_object_features, lidar_object_feature_mask = _prepare_lidar_object_feature_batch(
                    batch_lidar_object_features, fixed_tokens=lidar_object_prefix_length
                )
                if lidar_object_features is not None:
                    features["lidar_object_features"] = lidar_object_features
                    features["lidar_object_feature_mask"] = lidar_object_feature_mask
                if use_lidar_object_geometry_features and lidar_object_features is not None:
                    lidar_object_geometry_features, lidar_object_geometry_feature_mask = (
                        _prepare_lidar_object_geometry_feature_batch(
                            batch_lidar_object_geometry_features,
                            fixed_tokens=lidar_object_prefix_length,
                        )
                    )
                    if lidar_object_geometry_features is None:
                        raise ValueError(
                            "LiDAR object geometry features are enabled, but the current batch does not contain "
                            "usable `lidar_object_geometry_features`."
                        )

                    features["lidar_object_geometry_features"] = lidar_object_geometry_features
                    features["lidar_object_geometry_feature_mask"] = lidar_object_geometry_feature_mask
                if use_lidar_object_numeric_features and lidar_object_features is not None:
                    lidar_object_numeric_token_dim = getattr(self.model.config, "lidar_object_numeric_token_dim", 10)
                    numeric_feature_batch = _prepare_lidar_object_numeric_feature_batch(
                        batch_lidar_object_features,
                        fixed_tokens=lidar_object_prefix_length,
                        feature_dim=lidar_object_numeric_token_dim,
                    )
                    if numeric_feature_batch is None:
                        raise ValueError(
                            "LiDAR object numeric features are enabled, but the current batch does not contain "
                            "usable `.object.pt` payloads."
                        )
                    features.update(numeric_feature_batch)
                if use_lidar_scene_features:
                    lidar_scene_token_length = getattr(self.model.config, "lidar_scene_token_length", 256)
                    lidar_scene_features, lidar_scene_feature_mask = _prepare_lidar_scene_feature_batch(
                        batch_lidar_scene_features,
                        fixed_tokens=lidar_scene_token_length,
                    )
                    if lidar_scene_features is None:
                        raise ValueError(
                            "LiDAR scene features are enabled, but the current batch does not contain "
                            "usable `lidar_scene_features`."
                        )
                    features["lidar_scene_features"] = lidar_scene_features
                    features["lidar_scene_feature_mask"] = lidar_scene_feature_mask

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32
    neat_packing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            if self.model is not None and getattr(self.model.config, "model_type", None) in ["qwen3_5", "qwen3_5_moe", "gpt_oss"]:
                raise ValueError("Neat packing is not supported for qwen3_5, qwen3_5_moe, gpt_oss models for now.")

    @staticmethod
    def _unpad_packed_features(features: dict[str, Any]) -> None:
        r"""Trim padded positions for packed FA2 batches."""
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
            return

        seq_len = attention_mask.size(1)
        non_padding_indices = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        if non_padding_indices.numel() == seq_len:
            return

        keys_on_seq_dim_1 = {"input_ids", "labels", "attention_mask", "token_type_ids"}
        for key, value in list(features.items()):
            if not torch.is_tensor(value):
                continue

            if key == "position_ids" and value.size(-1) == seq_len:
                features[key] = value.index_select(-1, non_padding_indices)
            elif key == "cross_attention_mask" and value.dim() >= 2 and value.size(0) == 1 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding_indices)
            elif key in keys_on_seq_dim_1 and value.dim() == 2 and value.size(0) == 1 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding_indices)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        has_dummy_image = features.pop("has_dummy_image", False)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        if self.neat_packing and self.attn_implementation == "flash_attention_2": # FIXME compatibility fa3/fa4
            assert features["input_ids"].shape[0] == 1, "bsz should be 1 for neat packing"
            if not has_dummy_image:
                self._unpad_packed_features(features)

            features["attention_mask"] = None  # let transformers handle causal packed mask.

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for pairwise data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        r"""Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                    "audios": feature["audios"],
                    "lidar_object_features": feature.get("lidar_object_features"),
                    "lidar_object_geometry_features": feature.get("lidar_object_geometry_features"),
                    "lidar_scene_features": feature.get("lidar_scene_features"),
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for KTO data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
                "lidar_object_features": feature.get("lidar_object_features"),
                "lidar_object_geometry_features": feature.get("lidar_object_geometry_features"),
                "lidar_scene_features": feature.get("lidar_scene_features"),
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
                "lidar_object_features": feature.get("lidar_object_features"),
                "lidar_object_geometry_features": feature.get("lidar_object_geometry_features"),
                "lidar_scene_features": feature.get("lidar_scene_features"),
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:  # for mllama inputs
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]

        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
