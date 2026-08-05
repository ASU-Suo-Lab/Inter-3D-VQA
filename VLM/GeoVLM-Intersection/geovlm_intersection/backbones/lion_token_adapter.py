from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

from geovlm_intersection.backbones.lion_adapter import DEFAULT_LION_CHECKPOINTS, DEFAULT_LION_CONFIG, patch_transformers_generation_for_openpcdet
from geovlm_intersection.config.common import DEFAULT_LION_QUALITY, OPENPCDET_ROOT
from geovlm_intersection.data.v5_io import PreparedSample, load_point_cloud_xyzit


@dataclass(frozen=True)
class LionModelRuntime:
    config_path: Path
    checkpoint_path: Path
    cfg: EasyDict
    dataset: object
    model: torch.nn.Module
    device: str


@dataclass(frozen=True)
class LionTokenOutputs:
    bev_tokens: torch.Tensor
    object_tokens: torch.Tensor
    raw_object_tokens: torch.Tensor
    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_labels: torch.Tensor
    bev_feature_shape: tuple[int, ...]
    bev_grid_size: tuple[int, int]
    query_feature_dim: int
    object_local_feature_dim: int


class _GeoVLMLionDataset:
    def __init__(self, dataset_cfg: EasyDict, class_names: list[str]) -> None:
        from pcdet.datasets.dataset import DatasetTemplate

        class _Impl(DatasetTemplate):
            def __len__(self) -> int:  # pragma: no cover - not used in smoke path
                return 1

            def __getitem__(self, index: int):  # pragma: no cover - not used in smoke path
                raise IndexError(index)

        self.impl = _Impl(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=Path(dataset_cfg.DATA_PATH),
            logger=None,
        )

    def prepare_points(self, points_xyzit: np.ndarray, frame_token: str) -> dict[str, np.ndarray | str | dict[str, str]]:
        data_dict = {
            "points": points_xyzit.astype(np.float32),
            "frame_id": frame_token,
            "metadata": {"token": frame_token},
        }
        return self.impl.prepare_data(data_dict=data_dict)

    @staticmethod
    def collate_batch(batch_list: list[dict[str, object]]) -> dict[str, object]:
        from pcdet.datasets.dataset import DatasetTemplate

        return DatasetTemplate.collate_batch(batch_list)


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def require_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "LION token extraction requires CUDA, but torch.cuda.is_available() is False in the current environment."
        )


def _resolve_config_with_base(config_path: Path) -> EasyDict:
    patch_transformers_generation_for_openpcdet()
    import sys

    root = str(OPENPCDET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from pcdet.config import cfg, cfg_from_yaml_file

    local_cfg = EasyDict()
    local_cfg.ROOT_DIR = cfg.ROOT_DIR
    local_cfg.LOCAL_RANK = cfg.LOCAL_RANK
    with _pushd(OPENPCDET_ROOT / "tools"):
        cfg_from_yaml_file(str(config_path), local_cfg)
    return local_cfg


def _capture_transfusion_query_features(model: torch.nn.Module) -> None:
    dense_head = getattr(model, "dense_head", None)
    if dense_head is None:
        raise AttributeError("Expected OpenPCDet model to expose dense_head for TransFusion query capture.")
    if getattr(dense_head, "_geovlm_query_capture_installed", False):
        return

    original_predict = dense_head.predict
    captured: dict[str, torch.Tensor] = {}

    def _prediction_head_pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if inputs:
            captured["query_feat"] = inputs[0].detach()

    dense_head.prediction_head.register_forward_pre_hook(_prediction_head_pre_hook)

    def _wrapped_predict(inputs: torch.Tensor):
        captured.clear()
        res = original_predict(inputs)
        dense_head._geovlm_last_query_feat = captured.get("query_feat")
        dense_head._geovlm_last_predict_res = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in res.items()
        }
        return res

    dense_head.predict = _wrapped_predict
    dense_head._geovlm_query_capture_installed = True


def _build_coarse_bev_tokens(
    spatial_features_2d: torch.Tensor,
    *,
    token_budget: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if spatial_features_2d.ndim != 4:
        raise ValueError(f"Expected spatial_features_2d [B, C, H, W], got shape={tuple(spatial_features_2d.shape)}")
    grid_side = int(token_budget**0.5)
    if grid_side * grid_side != token_budget:
        raise ValueError(f"BEV token budget must be a perfect square for coarse-grid pooling, got: {token_budget}")
    pooled = F.adaptive_avg_pool2d(spatial_features_2d, output_size=(grid_side, grid_side))
    batch, channels, height, width = pooled.shape
    xs = torch.linspace(0.0, 1.0, steps=height, device=pooled.device, dtype=pooled.dtype)
    ys = torch.linspace(0.0, 1.0, steps=width, device=pooled.device, dtype=pooled.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    bev_with_pos = torch.cat([pooled, pos], dim=1)
    bev_tokens = bev_with_pos.permute(0, 2, 3, 1).reshape(batch, height * width, channels + 2)
    return bev_tokens, (height, width)


def _world_xy_to_bev_indices(
    *,
    xy: torch.Tensor,
    feature_map_stride: int,
    voxel_size: tuple[float, float],
    point_cloud_range: tuple[float, float],
    spatial_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_x = torch.round((xy[:, 0] - point_cloud_range[0]) / (feature_map_stride * voxel_size[0]) - 0.5).long()
    grid_y = torch.round((xy[:, 1] - point_cloud_range[1]) / (feature_map_stride * voxel_size[1]) - 0.5).long()
    grid_x = grid_x.clamp(min=0, max=spatial_shape[0] - 1)
    grid_y = grid_y.clamp(min=0, max=spatial_shape[1] - 1)
    return grid_x, grid_y


def _extract_object_local_bev_features(
    *,
    spatial_features_2d: torch.Tensor,
    pred_boxes: torch.Tensor,
    feature_map_stride: int,
    voxel_size: tuple[float, float],
    point_cloud_range: tuple[float, float],
    roi_radius: int,
) -> torch.Tensor:
    if pred_boxes.ndim != 2:
        raise ValueError(f"Expected pred_boxes [N, box_dim], got shape={tuple(pred_boxes.shape)}")
    if pred_boxes.shape[0] == 0:
        return spatial_features_2d.new_zeros((0, spatial_features_2d.shape[1]))
    feature_map = spatial_features_2d[0]
    channels, height, width = feature_map.shape
    grid_x, grid_y = _world_xy_to_bev_indices(
        xy=pred_boxes[:, :2],
        feature_map_stride=feature_map_stride,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        spatial_shape=(height, width),
    )
    padded = F.pad(feature_map.unsqueeze(0), (roi_radius, roi_radius, roi_radius, roi_radius), mode="replicate").squeeze(0)
    local_features: list[torch.Tensor] = []
    for x_idx, y_idx in zip(grid_x.tolist(), grid_y.tolist()):
        x0 = x_idx
        y0 = y_idx
        patch = padded[:, x0 : x0 + 2 * roi_radius + 1, y0 : y0 + 2 * roi_radius + 1]
        local_features.append(patch.mean(dim=(1, 2)))
    return torch.stack(local_features, dim=0).view(-1, channels)


def _select_transfusion_query_indices(
    *,
    dense_head: torch.nn.Module,
    preds_dicts: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    from pcdet.models.model_utils import model_nms_utils

    batch_size = preds_dicts["heatmap"].shape[0]
    # `preds_dicts` was captured under torch.inference_mode() during the LION
    # forward pass. `decode_bbox()` performs in-place updates on `center`, so
    # we must clone here outside inference mode to materialize normal tensors.
    heatmap = preds_dicts["heatmap"].clone()
    center = preds_dicts["center"].clone()
    height = preds_dicts["height"].clone()
    dim = preds_dicts["dim"].clone()
    rot = preds_dicts["rot"].clone()
    vel = preds_dicts["vel"].clone() if "vel" in preds_dicts else None
    iou = preds_dicts["iou"].clone() if "iou" in preds_dicts else None

    batch_score = heatmap.sigmoid()
    one_hot = F.one_hot(dense_head.query_labels, num_classes=dense_head.num_classes).permute(0, 2, 1)
    batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
    batch_center = center
    batch_height = height
    batch_dim = dim
    batch_rot = rot
    batch_vel = vel
    batch_iou = (iou + 1) * 0.5 if iou is not None else None
    filtered = dense_head.decode_bbox(
        batch_score,
        batch_rot,
        batch_dim,
        batch_center,
        batch_height,
        batch_vel,
        filter=True,
    )

    if dense_head.dataset_name in {"nuScenes", "sunlakes"}:
        tasks = [
            dict(indices=[0, 1, 2, 3, 4, 5, 6, 7, 9], radius=-1),
            dict(indices=[8], radius=0.175),
        ]
    elif dense_head.dataset_name == "Waymo":
        tasks = [
            dict(indices=[0], radius=0.7),
            dict(indices=[1], radius=0.7),
            dict(indices=[2], radius=0.7),
        ]
    else:
        raise ValueError(f"Unsupported dataset_name for TransFusion query selection: {dense_head.dataset_name}")

    selected_indices_per_batch: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        scores = filtered[batch_index]["pred_scores"]
        labels = filtered[batch_index]["pred_labels"]
        cmask = filtered[batch_index]["cmask"]
        candidate_indices = torch.where(cmask)[0]
        if batch_iou is not None and dense_head.model_cfg.POST_PROCESSING.get("USE_IOU_TO_RECTIFY_SCORE", False):
            pred_iou = torch.clamp(batch_iou[batch_index][0][cmask], min=0, max=1.0)
            rectifier = scores.new_tensor(dense_head.model_cfg.POST_PROCESSING.IOU_RECTIFIER)
            if len(rectifier) == 1:
                rectifier = rectifier.repeat(dense_head.num_classes)
            scores = torch.pow(scores, 1 - rectifier[labels]) * torch.pow(pred_iou, rectifier[labels])
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for task in tasks:
            task_mask = torch.zeros_like(scores, dtype=torch.bool)
            for cls_idx in task["indices"]:
                task_mask |= labels == cls_idx
            if not task_mask.any():
                continue
            if task["radius"] > 0:
                top_scores = scores[task_mask]
                boxes_for_nms = filtered[batch_index]["pred_boxes"][task_mask][:, :7].clone().detach()
                task_nms_config = copy.deepcopy(dense_head.model_cfg.POST_PROCESSING.NMS_CONFIG)
                task_nms_config.NMS_THRESH = task["radius"]
                task_keep_indices, _ = model_nms_utils.class_agnostic_nms(
                    box_scores=top_scores,
                    box_preds=boxes_for_nms,
                    nms_config=task_nms_config,
                    score_thresh=task_nms_config.SCORE_THRES,
                )
                keep_indices = torch.where(task_mask)[0][task_keep_indices]
                keep_mask[keep_indices] = True
            else:
                task_keep_indices = torch.arange(int(task_mask.sum().item()), device=scores.device)
                keep_indices = torch.where(task_mask)[0][task_keep_indices]
                keep_mask[keep_indices] = True
        selected_indices_per_batch.append(candidate_indices[keep_mask])
    return selected_indices_per_batch


def _match_final_detections_to_query_indices(
    *,
    dense_head: torch.nn.Module,
    preds_dicts: dict[str, torch.Tensor],
    final_boxes: torch.Tensor,
    final_scores: torch.Tensor,
    final_labels: torch.Tensor,
) -> torch.Tensor:
    batch_size = preds_dicts["heatmap"].shape[0]
    if batch_size != 1:
        raise ValueError(f"GeoVLM query matching currently expects batch_size=1, got: {batch_size}")
    if final_boxes.ndim != 2:
        raise ValueError(f"Expected final_boxes [N, box_dim], got shape={tuple(final_boxes.shape)}")
    if final_boxes.shape[0] == 0:
        return final_boxes.new_zeros((0,), dtype=torch.long)

    heatmap = preds_dicts["heatmap"].clone()
    center = preds_dicts["center"].clone()
    height = preds_dicts["height"].clone()
    dim = preds_dicts["dim"].clone()
    rot = preds_dicts["rot"].clone()
    vel = preds_dicts["vel"].clone() if "vel" in preds_dicts else None

    batch_score = heatmap.sigmoid()
    one_hot = F.one_hot(dense_head.query_labels, num_classes=dense_head.num_classes).permute(0, 2, 1)
    batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
    decoded_all = dense_head.decode_bbox(
        batch_score,
        rot,
        dim,
        center,
        height,
        vel,
        filter=False,
    )[0]

    query_boxes = decoded_all["pred_boxes"]
    query_scores = decoded_all["pred_scores"]
    query_labels = decoded_all["pred_labels"].long()
    target_labels = final_labels.long()
    if target_labels.numel() > 0 and query_labels.numel() > 0 and int(target_labels.min().item()) >= 1 and int(query_labels.min().item()) == 0:
        target_labels = target_labels - 1

    used = torch.zeros(query_boxes.shape[0], dtype=torch.bool, device=query_boxes.device)
    matched_indices: list[torch.Tensor] = []
    box_dims = min(query_boxes.shape[-1], final_boxes.shape[-1])

    for final_index in range(final_boxes.shape[0]):
        available = ~used
        label_mask = query_labels == target_labels[final_index]
        candidate_mask = available & label_mask
        if not candidate_mask.any():
            candidate_mask = available
        if not candidate_mask.any():
            raise RuntimeError("Ran out of TransFusion queries while matching final detections.")

        box_cost = (query_boxes[:, :box_dims] - final_boxes[final_index, :box_dims]).abs().sum(dim=-1)
        score_cost = (query_scores - final_scores[final_index]).abs()
        total_cost = box_cost + 0.01 * score_cost
        total_cost = total_cost.masked_fill(~candidate_mask, float("inf"))
        best_query_index = torch.argmin(total_cost)
        if not torch.isfinite(total_cost[best_query_index]):
            raise RuntimeError("Failed to find a finite-cost TransFusion query match for final detection.")
        used[best_query_index] = True
        matched_indices.append(best_query_index)

    return torch.stack(matched_indices, dim=0)


def build_lion_model_runtime(quality: str = DEFAULT_LION_QUALITY, device: str = "cuda:0") -> LionModelRuntime:
    require_cuda_runtime()
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
    import sys

    root = str(OPENPCDET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from pcdet.models import build_network
    from pcdet.utils.common_utils import create_logger

    cfg = _resolve_config_with_base(config_path)
    dataset = _GeoVLMLionDataset(cfg.DATA_CONFIG, list(cfg.CLASS_NAMES))
    logger = create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset.impl)
    model.load_params_from_file(str(checkpoint_path), logger=logger, to_cpu=False)
    model = model.to(device)
    model.eval()
    _capture_transfusion_query_features(model)

    return LionModelRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        dataset=dataset,
        model=model,
        device=device,
    )


def _move_batch_to_model_device(batch_dict: dict[str, object], device: str) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch_dict.items():
        if isinstance(value, np.ndarray):
            if key in {"frame_id", "metadata", "calib", "image_paths", "ori_shape", "img_process_infos"}:
                moved[key] = value
            else:
                tensor = torch.from_numpy(value)
                if key == "image_shape":
                    moved[key] = tensor.int().to(device)
                else:
                    moved[key] = tensor.float().to(device)
        else:
            moved[key] = value
    return moved


def extract_lion_tokens(
    sample: PreparedSample,
    runtime: LionModelRuntime,
    *,
    max_objects: int = 128,
    bev_token_budget: int = 1024,
    object_roi_radius: int = 2,
) -> LionTokenOutputs:
    if max_objects <= 0:
        raise ValueError(f"max_objects must be positive, got: {max_objects}")

    points_xyzit = load_point_cloud_xyzit(sample.point_cloud_path)
    prepared = runtime.dataset.prepare_points(points_xyzit, sample.frame_token)
    batch_dict = runtime.dataset.collate_batch([prepared])
    batch_dict = _move_batch_to_model_device(batch_dict, runtime.device)

    with torch.inference_mode():
        for cur_module in runtime.model.module_list:
            batch_dict = cur_module(batch_dict)

    spatial_features_2d = batch_dict.get("spatial_features_2d")
    final_box_dicts = batch_dict.get("final_box_dicts")
    if spatial_features_2d is None:
        raise KeyError("LION forward did not produce 'spatial_features_2d'.")
    if final_box_dicts is None or not final_box_dicts:
        raise KeyError("LION forward did not produce 'final_box_dicts'.")
    dense_head = getattr(runtime.model, "dense_head", None)
    if dense_head is None:
        raise AttributeError("LION runtime model is missing dense_head.")
    query_feat = getattr(dense_head, "_geovlm_last_query_feat", None)
    predict_res = getattr(dense_head, "_geovlm_last_predict_res", None)
    if query_feat is None or predict_res is None:
        raise RuntimeError("TransFusion query capture is missing. Rebuild the LION runtime with GeoVLM capture hooks.")

    bev_tokens, bev_grid_size = _build_coarse_bev_tokens(spatial_features_2d, token_budget=bev_token_budget)

    pred_boxes = final_box_dicts[0]["pred_boxes"]
    pred_scores = final_box_dicts[0]["pred_scores"]
    pred_labels = final_box_dicts[0]["pred_labels"]
    if pred_boxes.ndim != 2:
        raise ValueError(f"Expected pred_boxes to be [N, box_dim], got shape={tuple(pred_boxes.shape)}")

    keep = min(max_objects, pred_boxes.shape[0])
    pred_boxes = pred_boxes[:keep]
    pred_scores = pred_scores[:keep]
    pred_labels = pred_labels[:keep]
    raw_object_tokens = torch.cat(
        [
            pred_boxes,
            pred_scores.unsqueeze(-1),
            pred_labels.float().unsqueeze(-1),
        ],
        dim=-1,
    ).unsqueeze(0)
    selected_query_indices = _select_transfusion_query_indices(
        dense_head=dense_head,
        preds_dicts=predict_res,
    )[0]
    if selected_query_indices.shape[0] < keep:
        selected_query_indices = _match_final_detections_to_query_indices(
            dense_head=dense_head,
            preds_dicts=predict_res,
            final_boxes=pred_boxes,
            final_scores=pred_scores,
            final_labels=pred_labels,
        )
    if selected_query_indices.shape[0] < keep:
        raise RuntimeError(
            f"TransFusion query selection count {selected_query_indices.shape[0]} is smaller than final detection count {keep}, even after geometry matching fallback."
        )
    selected_query_indices = selected_query_indices[:keep]
    query_feature_tokens = query_feat[0, :, selected_query_indices].transpose(0, 1).contiguous()
    object_local_features = _extract_object_local_bev_features(
        spatial_features_2d=spatial_features_2d,
        pred_boxes=pred_boxes,
        feature_map_stride=int(dense_head.feature_map_stride),
        voxel_size=(float(dense_head.voxel_size[0]), float(dense_head.voxel_size[1])),
        point_cloud_range=(float(dense_head.point_cloud_range[0]), float(dense_head.point_cloud_range[1])),
        roi_radius=object_roi_radius,
    )
    object_tokens = torch.cat(
        [
            query_feature_tokens,
            object_local_features,
            raw_object_tokens.squeeze(0),
        ],
        dim=-1,
    ).unsqueeze(0)

    return LionTokenOutputs(
        bev_tokens=bev_tokens,
        object_tokens=object_tokens,
        raw_object_tokens=raw_object_tokens,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        bev_feature_shape=tuple(spatial_features_2d.shape),
        bev_grid_size=bev_grid_size,
        query_feature_dim=int(query_feature_tokens.shape[-1]),
        object_local_feature_dim=int(object_local_features.shape[-1]),
    )
