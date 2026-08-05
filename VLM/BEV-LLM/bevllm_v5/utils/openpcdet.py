from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from bevllm_v5.utils.io import ensure, load_json, normalize_data_path


SUPPORTED_BEVFUSION_COMPONENTS = {
    "detector": "BevFusion",
    "vfe": {"MeanVFE"},
    "backbone_3d": {"VoxelResBackBone8x", "VoxelBackBone8x"},
    "map_to_bev": {"HeightCompression"},
    "image_backbone": {"SwinTransformer"},
    "neck": {"GeneralizedLSSFPN"},
    "vtransform": {"DepthLSSTransform"},
    "fuser": {"ConvFuser"},
    "backbone_2d": {"BaseBEVBackbone"},
}

_SHALLOW_PACKAGE_PATHS = {
    "pcdet.datasets": ("pcdet", "datasets"),
    "pcdet.datasets.augmentor": ("pcdet", "datasets", "augmentor"),
    "pcdet.datasets.processor": ("pcdet", "datasets", "processor"),
    "pcdet.datasets.carla": ("pcdet", "datasets", "carla"),
    "pcdet.models": ("pcdet", "models"),
    "pcdet.models.backbones_2d": ("pcdet", "models", "backbones_2d"),
    "pcdet.models.backbones_2d.map_to_bev": ("pcdet", "models", "backbones_2d", "map_to_bev"),
    "pcdet.models.backbones_2d.fuser": ("pcdet", "models", "backbones_2d", "fuser"),
    "pcdet.models.backbones_3d": ("pcdet", "models", "backbones_3d"),
    "pcdet.models.backbones_3d.vfe": ("pcdet", "models", "backbones_3d", "vfe"),
    "pcdet.models.backbones_image": ("pcdet", "models", "backbones_image"),
    "pcdet.models.backbones_image.img_neck": ("pcdet", "models", "backbones_image", "img_neck"),
    "pcdet.models.model_utils": ("pcdet", "models", "model_utils"),
    "pcdet.models.view_transforms": ("pcdet", "models", "view_transforms"),
}

_EXTRACT_MODULES = (
    "easydict",
    "SharedArray",
    "spconv",
    "open3d",
    "kornia",
    "pcdet.datasets.dataset",
    "pcdet.datasets.carla.carla_dataset",
    "pcdet.models.backbones_3d.vfe.mean_vfe",
    "pcdet.models.backbones_3d.spconv_backbone",
    "pcdet.models.backbones_2d.map_to_bev.height_compression",
    "pcdet.models.backbones_image.swin",
    "pcdet.models.backbones_image.img_neck.generalized_lss",
    "pcdet.models.view_transforms.depth_lss",
    "pcdet.models.backbones_2d.fuser.convfuser",
    "pcdet.models.backbones_2d.base_bev_backbone",
    "pcdet.utils.spconv_utils",
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _ensure_namespace_package(name: str, package_dir: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(package_dir)]
        spec = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_dir)]
        module.__spec__ = spec
        sys.modules[name] = module
        return
    if not hasattr(module, "__path__"):
        module.__path__ = [str(package_dir)]
    elif str(package_dir) not in module.__path__:
        module.__path__.append(str(package_dir))


def ensure_openpcdet_on_path(openpcdet_root: Path) -> Path:
    root = Path(openpcdet_root).resolve()
    ensure(root.is_dir(), f"OpenPCDet root not found: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def ensure_openpcdet_shallow_packages(openpcdet_root: Path) -> Path:
    root = ensure_openpcdet_on_path(openpcdet_root)
    importlib.import_module("pcdet")
    for package_name, relative_parts in _SHALLOW_PACKAGE_PATHS.items():
        _ensure_namespace_package(package_name, root.joinpath(*relative_parts))
    return root


def import_openpcdet_module(module_name: str, openpcdet_root: Path):
    ensure_openpcdet_shallow_packages(openpcdet_root)
    return importlib.import_module(module_name)


def load_openpcdet_cfg(config_path: Path, openpcdet_root: Path):
    root = ensure_openpcdet_shallow_packages(openpcdet_root)
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file

    cfg = EasyDict()
    cfg.ROOT_DIR = root
    cfg.LOCAL_RANK = 0
    tools_dir = root / "tools"
    ensure(tools_dir.is_dir(), f"OpenPCDet tools directory not found: {tools_dir}")
    with working_directory(tools_dir):
        cfg_from_yaml_file(str(Path(config_path).resolve()), cfg)
    cfg.ROOT_DIR = root
    cfg.LOCAL_RANK = 0
    return cfg


def validate_supported_bevfusion_cfg(cfg) -> None:
    ensure(cfg.MODEL.NAME == SUPPORTED_BEVFUSION_COMPONENTS["detector"], f"Unsupported detector: {cfg.MODEL.NAME}")
    ensure(cfg.MODEL.VFE.NAME in SUPPORTED_BEVFUSION_COMPONENTS["vfe"], f"Unsupported VFE: {cfg.MODEL.VFE.NAME}")
    ensure(
        cfg.MODEL.BACKBONE_3D.NAME in SUPPORTED_BEVFUSION_COMPONENTS["backbone_3d"],
        f"Unsupported BACKBONE_3D: {cfg.MODEL.BACKBONE_3D.NAME}",
    )
    ensure(
        cfg.MODEL.MAP_TO_BEV.NAME in SUPPORTED_BEVFUSION_COMPONENTS["map_to_bev"],
        f"Unsupported MAP_TO_BEV: {cfg.MODEL.MAP_TO_BEV.NAME}",
    )
    ensure(
        cfg.MODEL.IMAGE_BACKBONE.NAME in SUPPORTED_BEVFUSION_COMPONENTS["image_backbone"],
        f"Unsupported IMAGE_BACKBONE: {cfg.MODEL.IMAGE_BACKBONE.NAME}",
    )
    ensure(cfg.MODEL.NECK.NAME in SUPPORTED_BEVFUSION_COMPONENTS["neck"], f"Unsupported NECK: {cfg.MODEL.NECK.NAME}")
    ensure(
        cfg.MODEL.VTRANSFORM.NAME in SUPPORTED_BEVFUSION_COMPONENTS["vtransform"],
        f"Unsupported VTRANSFORM: {cfg.MODEL.VTRANSFORM.NAME}",
    )
    ensure(cfg.MODEL.FUSER.NAME in SUPPORTED_BEVFUSION_COMPONENTS["fuser"], f"Unsupported FUSER: {cfg.MODEL.FUSER.NAME}")
    ensure(
        cfg.MODEL.BACKBONE_2D.NAME in SUPPORTED_BEVFUSION_COMPONENTS["backbone_2d"],
        f"Unsupported BACKBONE_2D: {cfg.MODEL.BACKBONE_2D.NAME}",
    )


def probe_openpcdet_extract_stack(openpcdet_root: Path, cfg) -> None:
    validate_supported_bevfusion_cfg(cfg)
    for module_name in _EXTRACT_MODULES:
        import_openpcdet_module(module_name, openpcdet_root)


def resolve_openpcdet_data_path(path_like: str, *, tools_dir: Path) -> str:
    raw = str(path_like).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return str(path.resolve())
    return str((tools_dir / path).resolve())


def normalize_openpcdet_info(info: dict[str, Any], *, tools_dir: Path) -> dict[str, Any]:
    normalized = deepcopy(info)
    if "lidar_path" in normalized:
        normalized["lidar_path"] = resolve_openpcdet_data_path(normalized["lidar_path"], tools_dir=tools_dir)
    if "radar_path" in normalized:
        normalized["radar_path"] = resolve_openpcdet_data_path(normalized["radar_path"], tools_dir=tools_dir)
    cams = normalized.get("cams")
    if isinstance(cams, dict):
        for cam_payload in cams.values():
            image_path = cam_payload.get("image_paths")
            if image_path:
                cam_payload["image_paths"] = resolve_openpcdet_data_path(image_path, tools_dir=tools_dir)
    return normalized


def load_prepared_frame_lookup(prepared_dir: Path) -> dict[str, dict[str, Any]]:
    prepared_root = Path(prepared_dir).resolve()
    lookup: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        frame_path = prepared_root / f"frames_{split}.json"
        ensure(frame_path.is_file(), f"Prepared frame manifest not found: {frame_path}")
        rows = load_json(frame_path)
        ensure(isinstance(rows, list) and rows, f"{frame_path} does not contain a non-empty list.")
        for row in rows:
            frame_token = str(row["frame_token"])
            point_cloud_path = str(Path(normalize_data_path(str(row["point_cloud_path"]))).resolve())
            image_paths = [str(Path(normalize_data_path(str(image_path))).resolve()) for image_path in row["images"]]
            for path_like in [point_cloud_path, *image_paths]:
                ensure(Path(path_like).is_file(), f"Prepared frame asset not found: {path_like}")
            frame_record = {
                "scene_id": str(row["scene_id"]),
                "frame_token": frame_token,
                "point_cloud_path": point_cloud_path,
                "images": image_paths,
                "split": split,
            }
            if frame_token in lookup:
                ensure(lookup[frame_token] == frame_record, f"Conflicting prepared frame records for token {frame_token}.")
            lookup[frame_token] = frame_record
    ensure(lookup, f"No prepared frames found under {prepared_root}.")
    return lookup


def apply_prepared_frame_paths(info: dict[str, Any], frame_record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(info)
    frame_token = str(frame_record["frame_token"])

    prepared_lidar_path = str(Path(frame_record["point_cloud_path"]).resolve())
    info_lidar_path = str(Path(normalize_data_path(str(normalized["lidar_path"]))).resolve())
    ensure(
        info_lidar_path == prepared_lidar_path,
        f"Prepared lidar path mismatch for frame {frame_token}: info={info_lidar_path} prepared={prepared_lidar_path}",
    )
    normalized["lidar_path"] = prepared_lidar_path

    prepared_image_lookup = {
        str(Path(normalize_data_path(image_path)).resolve()): image_path for image_path in frame_record["images"]
    }
    cams = normalized.get("cams")
    ensure(isinstance(cams, dict) and cams, f"OpenPCDet info for frame {frame_token} is missing camera metadata.")
    for camera_name, camera_payload in cams.items():
        raw_image_path = camera_payload.get("image_paths")
        ensure(raw_image_path, f"Camera {camera_name} is missing image_paths for frame {frame_token}.")
        resolved_image_path = str(Path(normalize_data_path(str(raw_image_path))).resolve())
        ensure(
            resolved_image_path in prepared_image_lookup,
            f"Prepared images for frame {frame_token} do not contain {camera_name} path {resolved_image_path}.",
        )
        camera_payload["image_paths"] = prepared_image_lookup[resolved_image_path]
    return normalized


def create_openpcdet_logger(rank: int) -> logging.Logger:
    logger_name = f"bevllm_v5.openpcdet.rank{rank}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[openpcdet] %(message)s"))
    logger.addHandler(handler)
    return logger


def load_data_to_gpu(batch_dict: dict[str, Any]) -> None:
    import kornia

    for key, val in batch_dict.items():
        if key == "camera_imgs":
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ["frame_id", "metadata", "calib", "image_paths", "ori_shape", "img_process_infos"]:
            continue
        elif key in ["images"]:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ["image_shape"]:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


class ExtractOnlyBevFusionBackbone(nn.Module):
    def __init__(self, cfg, dataset, component_classes: dict[str, type[nn.Module]]):
        super().__init__()
        self.model_cfg = cfg.MODEL
        self.dataset = dataset
        self.class_names = dataset.class_names

        model_info_dict: dict[str, Any] = {
            "module_list": [],
            "num_rawpoint_features": self.dataset.point_feature_encoder.num_point_features,
            "num_point_features": self.dataset.point_feature_encoder.num_point_features,
            "grid_size": self.dataset.grid_size,
            "point_cloud_range": self.dataset.point_cloud_range,
            "voxel_size": self.dataset.voxel_size,
            "depth_downsample_factor": self.dataset.depth_downsample_factor,
        }

        self.vfe, model_info_dict = self._build_vfe(model_info_dict, component_classes["vfe"])
        self.backbone_3d, model_info_dict = self._build_backbone_3d(model_info_dict, component_classes["backbone_3d"])
        self.map_to_bev_module, model_info_dict = self._build_map_to_bev(model_info_dict, component_classes["map_to_bev"])
        self.image_backbone, model_info_dict = self._build_image_backbone(model_info_dict, component_classes["image_backbone"])
        self.neck, model_info_dict = self._build_neck(model_info_dict, component_classes["neck"])
        self.vtransform, model_info_dict = self._build_vtransform(model_info_dict, component_classes["vtransform"])
        self.fuser, model_info_dict = self._build_fuser(model_info_dict, component_classes["fuser"])
        self.backbone_2d, model_info_dict = self._build_backbone_2d(model_info_dict, component_classes["backbone_2d"])

        self.module_topology = [
            "vfe",
            "backbone_3d",
            "map_to_bev_module",
            "image_backbone",
            "neck",
            "vtransform",
            "fuser",
            "backbone_2d",
        ]
        self.module_list = [
            self.vfe,
            self.backbone_3d,
            self.map_to_bev_module,
            self.image_backbone,
            self.neck,
            self.vtransform,
            self.fuser,
            self.backbone_2d,
        ]

    def _build_vfe(self, model_info_dict: dict[str, Any], vfe_cls: type[nn.Module]):
        module = vfe_cls(
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict["num_rawpoint_features"],
            point_cloud_range=model_info_dict["point_cloud_range"],
            voxel_size=model_info_dict["voxel_size"],
            grid_size=model_info_dict["grid_size"],
            depth_downsample_factor=model_info_dict["depth_downsample_factor"],
        )
        model_info_dict["module_list"].append(module)
        model_info_dict["num_point_features"] = module.get_output_feature_dim()
        return module, model_info_dict

    def _build_backbone_3d(self, model_info_dict: dict[str, Any], backbone_3d_cls: type[nn.Module]):
        module = backbone_3d_cls(
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict["num_point_features"],
            grid_size=model_info_dict["grid_size"],
            voxel_size=model_info_dict["voxel_size"],
            point_cloud_range=model_info_dict["point_cloud_range"],
        )
        model_info_dict["module_list"].append(module)
        model_info_dict["num_point_features"] = module.num_point_features
        model_info_dict["backbone_channels"] = getattr(module, "backbone_channels", None)
        return module, model_info_dict

    def _build_map_to_bev(self, model_info_dict: dict[str, Any], map_to_bev_cls: type[nn.Module]):
        module = map_to_bev_cls(
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict["grid_size"],
        )
        model_info_dict["module_list"].append(module)
        model_info_dict["num_bev_features"] = module.num_bev_features
        return module, model_info_dict

    def _build_image_backbone(self, model_info_dict: dict[str, Any], image_backbone_cls: type[nn.Module]):
        module = image_backbone_cls(model_cfg=self.model_cfg.IMAGE_BACKBONE)
        module.init_weights()
        model_info_dict["module_list"].append(module)
        return module, model_info_dict

    def _build_neck(self, model_info_dict: dict[str, Any], neck_cls: type[nn.Module]):
        module = neck_cls(model_cfg=self.model_cfg.NECK)
        model_info_dict["module_list"].append(module)
        return module, model_info_dict

    def _build_vtransform(self, model_info_dict: dict[str, Any], vtransform_cls: type[nn.Module]):
        module = vtransform_cls(model_cfg=self.model_cfg.VTRANSFORM)
        model_info_dict["module_list"].append(module)
        return module, model_info_dict

    def _build_fuser(self, model_info_dict: dict[str, Any], fuser_cls: type[nn.Module]):
        module = fuser_cls(model_cfg=self.model_cfg.FUSER)
        model_info_dict["module_list"].append(module)
        model_info_dict["num_bev_features"] = self.model_cfg.FUSER.OUT_CHANNEL
        return module, model_info_dict

    def _build_backbone_2d(self, model_info_dict: dict[str, Any], backbone_2d_cls: type[nn.Module]):
        module = backbone_2d_cls(
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict["num_bev_features"],
        )
        model_info_dict["module_list"].append(module)
        model_info_dict["num_bev_features"] = module.num_bev_features
        return module, model_info_dict


def build_extract_only_bevfusion_model(cfg, dataset, openpcdet_root: Path) -> ExtractOnlyBevFusionBackbone:
    validate_supported_bevfusion_cfg(cfg)
    vfe_module = import_openpcdet_module("pcdet.models.backbones_3d.vfe.mean_vfe", openpcdet_root)
    backbone_3d_module = import_openpcdet_module("pcdet.models.backbones_3d.spconv_backbone", openpcdet_root)
    map_to_bev_module = import_openpcdet_module("pcdet.models.backbones_2d.map_to_bev.height_compression", openpcdet_root)
    image_backbone_module = import_openpcdet_module("pcdet.models.backbones_image.swin", openpcdet_root)
    neck_module = import_openpcdet_module("pcdet.models.backbones_image.img_neck.generalized_lss", openpcdet_root)
    vtransform_module = import_openpcdet_module("pcdet.models.view_transforms.depth_lss", openpcdet_root)
    fuser_module = import_openpcdet_module("pcdet.models.backbones_2d.fuser.convfuser", openpcdet_root)
    backbone_2d_module = import_openpcdet_module("pcdet.models.backbones_2d.base_bev_backbone", openpcdet_root)

    backbone_3d_classes = {
        "VoxelResBackBone8x": backbone_3d_module.VoxelResBackBone8x,
        "VoxelBackBone8x": backbone_3d_module.VoxelBackBone8x,
    }
    component_classes = {
        "vfe": vfe_module.MeanVFE,
        "backbone_3d": backbone_3d_classes[cfg.MODEL.BACKBONE_3D.NAME],
        "map_to_bev": map_to_bev_module.HeightCompression,
        "image_backbone": image_backbone_module.SwinTransformer,
        "neck": neck_module.GeneralizedLSSFPN,
        "vtransform": vtransform_module.DepthLSSTransform,
        "fuser": fuser_module.ConvFuser,
        "backbone_2d": backbone_2d_module.BaseBEVBackbone,
    }
    return ExtractOnlyBevFusionBackbone(cfg=cfg, dataset=dataset, component_classes=component_classes)


def load_openpcdet_checkpoint(model: nn.Module, checkpoint_path: Path, *, to_cpu: bool = False) -> dict[str, Any]:
    spconv_utils = importlib.import_module("pcdet.utils.spconv_utils")
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu") if to_cpu else None, weights_only=False)
    model_state_disk = checkpoint["model_state"]
    state_dict = model.state_dict()
    spconv_keys = spconv_utils.find_all_spconv_keys(model)

    update_model_state: dict[str, torch.Tensor] = {}
    for key, val in model_state_disk.items():
        if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
            val_native = val.transpose(-1, -2)
            if val_native.shape == state_dict[key].shape:
                val = val_native.contiguous()
            else:
                ensure(val.ndim == 5, f"Unsupported spconv weight rank for {key}: {val.shape}")
                val_implicit = val.permute(4, 0, 1, 2, 3)
                if val_implicit.shape == state_dict[key].shape:
                    val = val_implicit.contiguous()
        if key in state_dict and state_dict[key].shape == val.shape:
            update_model_state[key] = val

    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)
    return {
        "loaded": len(update_model_state),
        "total": len(state_dict),
        "version": checkpoint.get("version"),
    }
