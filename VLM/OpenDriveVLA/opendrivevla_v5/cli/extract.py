from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox import get_box_type
from tqdm import tqdm

from opendrivevla_v5.config.common import (
    CAN_BUS_DIM,
    DEFAULT_DATASET_VERSION,
    DEFAULT_FEATURE_TRAIN_DIR,
    DEFAULT_FEATURE_VAL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPARED_DIR,
    DEFAULT_WORK_DIR,
    STRICT_CAM_ORDER,
    resolve_dataset_version_paths,
)
from opendrivevla_v5.utils.dist import cleanup_distributed, init_distributed, resolve_local_rank_device, shard_sequence
from opendrivevla_v5.utils.io import dump_json, ensure, load_json, load_prepared_infos
from opendrivevla_v5.utils.modeling import load_feature_extractor


IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
PAD_DIVISOR = 32
LANE_MASK_SHAPE = (200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract strict 4-view UniAD features for Intersection V5.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--infos-pkl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_and_prepare_images(info: Dict) -> Tuple[List[str], List[Tuple[int, ...]], List[Tuple[int, ...]], List[np.ndarray], torch.Tensor]:
    image_paths: List[str] = []
    images = []
    ori_shapes: List[Tuple[int, ...]] = []
    pad_shapes: List[Tuple[int, ...]] = []
    lidar2img: List[np.ndarray] = []
    cams = info["cams"]

    for cam_key in STRICT_CAM_ORDER:
        ensure(cam_key in cams, f"Missing {cam_key} for frame {info['token']}")
        cam_info = cams[cam_key]
        image_path = str(cam_info["image_paths"])
        ensure(os.path.isfile(image_path), f"Image file not found for {info['token']} {cam_key}: {image_path}")
        image = mmcv.imread(image_path, "unchanged").astype(np.float32)
        matrix = np.asarray(cam_info["lidar2image"], dtype=np.float32)

        ori_shapes.append(image.shape)
        image = mmcv.imnormalize(image, IMG_MEAN, IMG_STD, to_rgb=True)
        image = mmcv.impad_to_multiple(image, PAD_DIVISOR, pad_val=0)
        pad_shapes.append(image.shape)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        images.append(image)
        image_paths.append(image_path)
        lidar2img.append(matrix)

    return image_paths, ori_shapes, pad_shapes, lidar2img, torch.stack(images, dim=0)


def build_img_meta(info: Dict, image_paths, ori_shapes, pad_shapes, lidar2img) -> Dict:
    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    return {
        "filename": image_paths,
        "ori_shape": ori_shapes,
        "img_shape": pad_shapes,
        "pad_shape": pad_shapes,
        "img_norm_cfg": {"mean": IMG_MEAN, "std": IMG_STD, "to_rgb": True},
        "sample_idx": info["token"],
        "pts_filename": str(info.get("lidar_path", info["token"])),
        "prev_idx": None,
        "next_idx": None,
        "scene_token": str(info.get("scene_id", "sunlakes_scene")),
        "can_bus": [0.0] * CAN_BUS_DIM,
        "lidar2img": lidar2img,
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    }


def build_dummy_lane_targets(device: torch.device):
    gt_lane_masks = [torch.zeros((1, 4, *LANE_MASK_SHAPE), dtype=torch.uint8, device=device)]
    gt_lane_labels = [torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)]
    return gt_lane_labels, gt_lane_masks


def build_dummy_planning_targets(device: torch.device):
    sdc_planning = torch.zeros((1, 1, 6, 3), dtype=torch.float32, device=device)
    sdc_planning_mask = torch.zeros((1, 1, 6, 3), dtype=torch.float32, device=device)
    command = torch.zeros((1,), dtype=torch.long, device=device)
    return sdc_planning, sdc_planning_mask, command


def extract_single_feature(detector, info: Dict, device: torch.device) -> Dict:
    image_paths, ori_shapes, pad_shapes, lidar2img, image_tensor = load_and_prepare_images(info)
    img_metas = [build_img_meta(info, image_paths, ori_shapes, pad_shapes, lidar2img)]
    img = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    l2g_t = torch.zeros((1, 3), dtype=torch.float32, device=device)
    l2g_r_mat = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    timestamp = torch.tensor([float(info["timestamp"])], dtype=torch.float64, device=device)
    gt_lane_labels, gt_lane_masks = build_dummy_lane_targets(device)
    sdc_planning, sdc_planning_mask, command = build_dummy_planning_targets(device)

    with torch.inference_mode():
        result_track = detector.simple_test_track(
            img=img,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            img_metas=img_metas,
            timestamp=timestamp,
        )
        result_track[0] = detector.upsample_bev_if_tiny(result_track[0])
        bev_embed = result_track[0]["bev_embed"]
        result_seg = detector.seg_head.forward_test(
            bev_embed,
            gt_lane_labels=gt_lane_labels,
            gt_lane_masks=gt_lane_masks,
            img_metas=img_metas,
            rescale=True,
        )
        return detector.get_results_for_vlm(
            img_metas[0],
            result_track[0],
            result_seg[0],
            sdc_planning[0],
            sdc_planning_mask[0],
            command[0],
            in_uniad_train=False,
        )


def extract_split(
    args: argparse.Namespace,
    split: str,
    rank: int,
    world_size: int,
    local_rank: int,
    device_name: str,
    device: torch.device,
    detector,
) -> None:
    prepared_dir = Path(args.prepared_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    infos_pkl = Path(args.infos_pkl).resolve() if args.infos_pkl else prepared_dir / f"infos_{split}.pkl"
    if args.output_dir and args.split != "all":
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = work_dir / ("features_train" if split == "train" else "features_val")
    output_dir.mkdir(parents=True, exist_ok=True)

    current_token: str | None = None
    current_stage = "setup"
    try:
        infos = load_prepared_infos(infos_pkl)
        infos.sort(key=lambda item: (str(item.get("scene_id", "")), float(item.get("timestamp", 0.0)), str(item["token"])))
        if args.limit is not None:
            infos = infos[: args.limit]
        shard = shard_sequence(infos, rank, world_size)

        print(
            f"[extract] split={split} rank={rank} local_rank={local_rank} device={device_name} "
            f"shard={len(shard)}/{len(infos)} output_dir={output_dir}",
            flush=True,
        )

        start_time = time.time()
        written = 0
        skipped = 0

        current_stage = "extract"
        for info in tqdm(shard, ncols=80, disable=rank != 0):
            current_token = str(info["token"])
            output_path = output_dir / f"{current_token}.pth"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            results_for_vlm = extract_single_feature(detector, info, device)
            current_stage = "save"
            torch.save(results_for_vlm, output_path)
            current_stage = "extract"
            written += 1

        elapsed = time.time() - start_time
        print(
            f"[extract] split={split} rank={rank} wrote={written} skipped={skipped} elapsed={elapsed:.1f}s",
            flush=True,
        )
    except Exception as exc:
        primary_error = exc
        context = f" token={current_token}" if current_token is not None else ""
        print(
            f"[extract] rank={rank} local_rank={local_rank} split={split} stage={current_stage}{context} "
            f"failed with {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device_name = resolve_local_rank_device(args.device)
    device = torch.device(device_name)
    primary_error: Exception | None = None
    current_stage = "load_model"
    current_split: str | None = None
    try:
        resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        work_dir = Path(resolved["work_dir"]).resolve()
        args.prepared_dir = str(prepared_dir)
        args.work_dir = str(work_dir)
        manifest_path = prepared_dir / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )
        _, detector = load_feature_extractor(str(Path(args.model_path).resolve()), args.device, args.attn_implementation)
        splits = ("train", "val") if args.split == "all" else (args.split,)
        current_stage = "extract"
        for split in splits:
            current_split = split
            extract_split(args, split, rank, world_size, local_rank, device_name, device, detector)
        if rank == 0:
            dump_json(
                work_dir / "feature_manifest.json",
                {
                    "dataset_version": str(resolved["dataset_version"]),
                    "prepared_dir": str(prepared_dir),
                },
            )
    except Exception as exc:
        primary_error = exc
        if current_stage == "load_model" or current_split is None:
            split_text = f" split={current_split}" if current_split is not None else ""
            print(
                f"[extract] rank={rank} local_rank={local_rank} stage={current_stage}{split_text} "
                f"failed with {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        cleanup_distributed(suppress_errors=primary_error is not None)


if __name__ == "__main__":
    main()
