from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geovlm_intersection.backbones import build_lion_model_runtime, extract_lion_tokens
from geovlm_intersection.config.common import DEFAULT_INFO_PKL, DEFAULT_LION_QUALITY
from geovlm_intersection.data import build_info_index, load_point_cloud_xyzit, load_prepared_records, resolve_prepared_sample
from geovlm_intersection.data.v5_io import resolve_default_prepared_dir
from geovlm_intersection.utils import dump_json


FONT = ImageFont.load_default()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize GeoVLM camera semantics with four images, GT refs, and an optional LION BEV overlay."
    )
    parser.add_argument("--dataset-version", default="v5", choices=["v5"])
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--info-pkl", type=Path, default=DEFAULT_INFO_PKL)
    parser.add_argument("--split", default="val_eval")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lion-quality", default=DEFAULT_LION_QUALITY, choices=["low", "mid", "high"])
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--skip-lion", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/geovlm_camera_semantics"))
    return parser


def _collect_image_refs(structured_targets: dict[str, Any] | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not isinstance(structured_targets, dict):
        return refs
    for key, value in structured_targets.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            image_name = item.get("image_name")
            x1 = item.get("x1")
            y1 = item.get("y1")
            if image_name in (None, "") or x1 is None or y1 is None:
                continue
            refs.append(
                {
                    "source_key": str(key),
                    "image_name": str(image_name),
                    "x": float(x1),
                    "y": float(y1),
                }
            )
    return refs


def _project_lidar_point(lidar2image: np.ndarray, xyz: tuple[float, float, float]) -> tuple[float, float] | None:
    matrix = np.asarray(lidar2image, dtype=np.float32)
    point = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float32)
    projected = matrix @ point
    if projected.shape[0] == 4:
        projected = projected[:3]
    if projected.shape[0] != 3:
        return None
    depth = float(projected[2])
    if abs(depth) < 1e-5:
        return None
    return float(projected[0] / depth), float(projected[1] / depth)


def _collect_tracking_targets(sample) -> list[dict[str, Any]]:
    targets = sample.structured_targets or {}
    tracking_ids = sample.info_record.get("tracking_id")
    gt_boxes = sample.info_record.get("gt_boxes")
    gt_names = sample.info_record.get("gt_names")
    if not isinstance(targets, dict) or not isinstance(tracking_ids, list) or gt_boxes is None or gt_names is None:
        return []
    collected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for key, value in targets.items():
        if key != "raw_tracking_id" and not str(key).endswith("_raw_tracking_id"):
            continue
        if value in (None, ""):
            continue
        raw_id = str(value)
        try:
            index = tracking_ids.index(raw_id)
        except ValueError:
            continue
        if index >= len(gt_boxes) or index >= len(gt_names):
            continue
        tag = str(key).removesuffix("_raw_tracking_id")
        dedupe_key = (tag, raw_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        box = gt_boxes[index]
        collected.append(
            {
                "tag": tag or "object",
                "raw_tracking_id": raw_id,
                "object_type": str(gt_names[index]),
                "xyz": (float(box[0]), float(box[1]), float(box[2])),
            }
        )
    return collected


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, size: int = 8) -> None:
    draw.line((x - size, y, x + size, y), fill=color, width=2)
    draw.line((x, y - size, x, y + size), fill=color, width=2)


def _build_camera_panel(sample, image_refs: list[dict[str, Any]], tracking_targets: list[dict[str, Any]]) -> Image.Image:
    panel_w, panel_h = 640, 420
    canvases: list[Image.Image] = []
    for view in sample.camera_views:
        image = Image.open(view.image_path).convert("RGB").resize((panel_w, panel_h - 60))
        canvas = Image.new("RGB", (panel_w, panel_h), "white")
        canvas.paste(image, (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, panel_w, 60), fill="#f3f3f3")
        draw.text((10, 8), f"sensor: {view.cam_key}", fill="black", font=FONT)
        draw.text((10, 28), f"view: {view.view_direction} ({view.image_name})", fill="black", font=FONT)
        for ref in image_refs:
            if ref["image_name"] != view.image_name:
                continue
            _draw_cross(draw, ref["x"] / 3.0, 60 + ref["y"] / 3.0, "#d62728")
            draw.text((ref["x"] / 3.0 + 10, 60 + ref["y"] / 3.0 - 10), ref["source_key"], fill="#d62728", font=FONT)
        for target in tracking_targets:
            projected = _project_lidar_point(view.lidar2image, target["xyz"])
            if projected is None:
                continue
            px, py = projected
            px /= 3.0
            py = 60 + (py / 3.0)
            if px < 0 or px >= panel_w or py < 60 or py >= panel_h:
                continue
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline="#1f77b4", width=2)
            draw.text((px + 8, py - 8), f"{target['tag']}:{target['object_type']}", fill="#1f77b4", font=FONT)
        canvases.append(canvas)

    montage = Image.new("RGB", (panel_w * 2, panel_h * 2), "#dddddd")
    montage.paste(canvases[0], (0, 0))
    montage.paste(canvases[1], (panel_w, 0))
    montage.paste(canvases[2], (0, panel_h))
    montage.paste(canvases[3], (panel_w, panel_h))
    return montage


def _xy_to_canvas(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int) -> tuple[float, float]:
    px = (x - x_min) / max(1e-6, (x_max - x_min)) * (width - 20) + 10
    py = height - (((y - y_min) / max(1e-6, (y_max - y_min)) * (height - 20)) + 10)
    return px, py


def _build_bev_panel(point_cloud: np.ndarray, tracking_targets: list[dict[str, Any]], lion_outputs) -> tuple[Image.Image, list[dict[str, Any]]]:
    width = 1280
    height = 720
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    xy = point_cloud[:, :2]
    qx = np.quantile(xy[:, 0], [0.01, 0.99])
    qy = np.quantile(xy[:, 1], [0.01, 0.99])
    x_min, x_max = float(qx[0] - 5.0), float(qx[1] + 5.0)
    y_min, y_max = float(qy[0] - 5.0), float(qy[1] + 5.0)

    if xy.shape[0] > 10000:
        indices = np.linspace(0, xy.shape[0] - 1, 10000, dtype=np.int64)
        sampled = xy[indices]
    else:
        sampled = xy
    for x, y in sampled:
        px, py = _xy_to_canvas(float(x), float(y), x_min, x_max, y_min, y_max, width, height)
        draw.point((px, py), fill="#c8c8c8")

    lion_matches: list[dict[str, Any]] = []
    pred_centers = None
    if lion_outputs is not None and lion_outputs.pred_boxes.numel() > 0:
        pred_centers = lion_outputs.pred_boxes[:, :2].detach().cpu().numpy()
        for pred_x, pred_y in pred_centers:
            px, py = _xy_to_canvas(float(pred_x), float(pred_y), x_min, x_max, y_min, y_max, width, height)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#4c78a8")

    for target in tracking_targets:
        tx, ty = float(target["xyz"][0]), float(target["xyz"][1])
        px, py = _xy_to_canvas(tx, ty, x_min, x_max, y_min, y_max, width, height)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline="#d62728", width=2)
        draw.text((px + 8, py - 8), f"{target['tag']}:{target['object_type']}", fill="#d62728", font=FONT)
        if pred_centers is not None and len(pred_centers) > 0:
            distances = np.linalg.norm(pred_centers - np.asarray([tx, ty], dtype=np.float32), axis=1)
            best_index = int(distances.argmin())
            best_x, best_y = pred_centers[best_index]
            best_px, best_py = _xy_to_canvas(float(best_x), float(best_y), x_min, x_max, y_min, y_max, width, height)
            draw.line((px, py, best_px, best_py), fill="#2ca02c", width=2)
            draw.ellipse((best_px - 4, best_py - 4, best_px + 4, best_py + 4), outline="#2ca02c", width=2)
            lion_matches.append(
                {
                    "tag": target["tag"],
                    "raw_tracking_id": target["raw_tracking_id"],
                    "object_type": target["object_type"],
                    "target_xy": [tx, ty],
                    "nearest_lion_xy": [float(best_x), float(best_y)],
                    "nearest_lion_distance_m": float(distances[best_index]),
                }
            )

    draw.text((10, 10), "BEV: gray=point cloud, red=GT target, blue=LION detections, green=nearest match", fill="black", font=FONT)
    return canvas, lion_matches


def main() -> None:
    args = build_parser().parse_args()
    prepared_dir = (args.prepared_dir or resolve_default_prepared_dir(args.dataset_version)).resolve()
    records = load_prepared_records(prepared_dir, split=args.split)
    if args.sample_index < 0 or args.sample_index >= len(records):
        raise IndexError(f"Sample index out of range: {args.sample_index} for {len(records)} records")
    info_index = build_info_index(args.info_pkl.resolve())
    sample = resolve_prepared_sample(
        records[args.sample_index],
        info_index,
        dataset_version=args.dataset_version,
        prepared_split=args.split,
        prepared_index=args.sample_index,
    )

    point_cloud = load_point_cloud_xyzit(sample.point_cloud_path)
    image_refs = _collect_image_refs(sample.structured_targets)
    tracking_targets = _collect_tracking_targets(sample)
    lion_outputs = None
    if not args.skip_lion:
        lion_runtime = build_lion_model_runtime(args.lion_quality)
        lion_outputs = extract_lion_tokens(sample, lion_runtime, max_objects=args.max_objects)

    image_panel = _build_camera_panel(sample, image_refs, tracking_targets)
    bev_panel, lion_matches = _build_bev_panel(point_cloud, tracking_targets, lion_outputs)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.frame_token}_{sample.question_id}"
    image_panel_path = output_dir / f"{stem}_cameras.png"
    bev_panel_path = output_dir / f"{stem}_bev.png"
    summary_path = output_dir / f"{stem}_summary.json"
    image_panel.save(image_panel_path)
    bev_panel.save(bev_panel_path)

    dump_json(
        summary_path,
        {
            "dataset_version": args.dataset_version,
            "split": args.split,
            "sample_index": args.sample_index,
            "question_id": sample.question_id,
            "frame_token": sample.frame_token,
            "subtemplate": sample.subtemplate,
            "camera_views": [
                {
                    "prepared_index": view.prepared_index,
                    "cam_key": view.cam_key,
                    "image_name": view.image_name,
                    "view_direction": view.view_direction,
                    "image_path": str(view.image_path),
                }
                for view in sample.camera_views
            ],
            "image_refs": image_refs,
            "tracking_targets": tracking_targets,
            "lion_matches": lion_matches,
            "camera_panel": str(image_panel_path),
            "bev_panel": str(bev_panel_path),
        },
    )
    print(
        json.dumps(
            {
                "camera_panel": str(image_panel_path),
                "bev_panel": str(bev_panel_path),
                "summary_json": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
