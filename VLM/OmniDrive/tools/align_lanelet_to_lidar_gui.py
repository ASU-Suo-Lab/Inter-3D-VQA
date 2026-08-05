import argparse
import json
import math
import os
import os.path as osp
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, RadioButtons, Slider

from align_lanelet_to_lidar import (
    alignment_cost,
    extract_lidar_reference_points,
    infer_lane_graph,
    load_infos,
    load_qa_tokens,
    parse_lanelet_map,
    select_best_axis_order,
    serialize_lanelets,
    subsample_points,
    transform_points_se2,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Desktop GUI for manually aligning a Lanelet2 .osm map to SunLakes LiDAR coordinates."
    )
    parser.add_argument("--osm-path", required=True, help="Path to Lanelet2 .osm file.")
    parser.add_argument("--ann-file", required=True, help="Path to sunlakes infos .pkl file.")
    parser.add_argument(
        "--qa-json",
        default=None,
        help="Optional QA json used to limit the reference frames.",
    )
    parser.add_argument(
        "--scene-ids",
        default=None,
        help="Optional comma-separated scene ids to keep.",
    )
    parser.add_argument(
        "--anchors-json",
        default=None,
        help="Optional anchors JSON passed through to the auto initializer.",
    )
    parser.add_argument(
        "--axis-order",
        default="auto",
        choices=["auto", "lonlat", "latlon"],
        help="Coordinate interpretation for the .osm nodes.",
    )
    parser.add_argument(
        "--vehicle-classes",
        default="car,truck,bus,trailer,van,golf_cart,motorcycle,bicycle,construction_vehicle",
        help="Comma-separated GT classes used to build LiDAR reference points.",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.5,
        help="Minimum planar speed for GT boxes used as LiDAR reference points.",
    )
    parser.add_argument(
        "--max-gt-points",
        type=int,
        default=12000,
        help="Maximum number of LiDAR reference points kept before subsampling.",
    )
    parser.add_argument(
        "--display-lidar-sample",
        type=int,
        default=5000,
        help="Maximum number of LiDAR points shown in the GUI.",
    )
    parser.add_argument(
        "--display-map-sample",
        type=int,
        default=2000,
        help="Maximum number of map centerline points used for cost preview.",
    )
    parser.add_argument(
        "--cost-lidar-sample",
        type=int,
        default=1200,
        help="Maximum number of LiDAR points used for the live cost preview.",
    )
    parser.add_argument(
        "--display-line-decimation",
        type=int,
        default=3,
        help="Keep every Nth centerline point for display. Larger values render faster.",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=12.0,
        help="Expand the LiDAR point cloud bounding box by this many meters when keeping lane lines for display.",
    )
    parser.add_argument(
        "--coarse-lidar-sample",
        type=int,
        default=400,
        help="Coarse LiDAR sample count for auto initialization.",
    )
    parser.add_argument(
        "--coarse-map-sample",
        type=int,
        default=800,
        help="Coarse map sample count for auto initialization.",
    )
    parser.add_argument(
        "--fine-lidar-sample",
        type=int,
        default=2000,
        help="Fine LiDAR sample count for auto initialization.",
    )
    parser.add_argument(
        "--fine-map-sample",
        type=int,
        default=5000,
        help="Fine map sample count for auto initialization.",
    )
    parser.add_argument(
        "--tx-range",
        type=float,
        default=250.0,
        help="Half range of the tx slider around the initial value, in meters.",
    )
    parser.add_argument(
        "--ty-range",
        type=float,
        default=250.0,
        help="Half range of the ty slider around the initial value, in meters.",
    )
    parser.add_argument(
        "--yaw-range-deg",
        type=float,
        default=180.0,
        help="Half range of the yaw slider around the initial value, in degrees.",
    )
    parser.add_argument(
        "--transform-json",
        default=None,
        help="Optional existing transform.json to preload.",
    )
    parser.add_argument(
        "--output-dir",
        default="./lanelet_align_gui_out",
        help="Directory where Save writes transform.json, aligned_centerlines.json, and lane_graph.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling.",
    )
    return parser.parse_args()


def wrap_angle_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def compute_bounds(points_xy):
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = np.maximum(max_xy - min_xy, 1.0)
    padding = np.maximum(span * 0.1, 5.0)
    return center, span + padding * 2.0


def format_status(axis_order, tx, ty, yaw_deg, cost, save_path=None):
    parts = [
        f"axis={axis_order}",
        f"tx={tx:.2f}",
        f"ty={ty:.2f}",
        f"yaw={yaw_deg:.2f} deg",
        f"cost={cost:.3f}",
    ]
    if save_path:
        parts.append(f"saved={save_path}")
    return " | ".join(parts)


def print_progress(message, step=None, total_steps=None, start_time=None):
    prefix = "[align-gui]"
    if step is not None and total_steps is not None:
        prefix += f" [{step}/{total_steps}]"
    if start_time is not None:
        prefix += f" [+{time.time() - start_time:0.1f}s]"
    print(f"{prefix} {message}", flush=True)


class ManualAlignGUI:
    def __init__(self, args, lidar_points, lidar_meta, lanelets_by_order, graph_by_order, auto_results):
        self.args = args
        self.lidar_meta = lidar_meta
        self.lidar_points = np.asarray(lidar_points, dtype=np.float64)
        self.display_lidar_points = subsample_points(self.lidar_points, args.display_lidar_sample)
        self.cost_lidar_points = subsample_points(self.lidar_points, args.cost_lidar_sample)
        self.lanelets_by_order = lanelets_by_order
        self.graph_by_order = graph_by_order
        self.auto_results = auto_results
        self.preview_map_points_by_order = {
            axis_order: subsample_points(
                np.concatenate([lanelet["centerline"] for lanelet in lanelets], axis=0),
                args.display_map_sample,
            )
            for axis_order, lanelets in lanelets_by_order.items()
        }
        self.display_centerlines_by_order = {
            axis_order: [self._decimate_polyline(lanelet["centerline"]) for lanelet in lanelets]
            for axis_order, lanelets in lanelets_by_order.items()
        }
        self.roi_min = self.lidar_points.min(axis=0) - args.roi_padding
        self.roi_max = self.lidar_points.max(axis=0) + args.roi_padding

        self.axis_orders = list(lanelets_by_order.keys())
        self.axis_order = self._initial_axis_order()
        self.initial_transform = self._initial_transform()
        self.current_transform = dict(self.initial_transform)

        self.figure = None
        self.ax_plot = None
        self.lidar_artist = None
        self.map_artist = None
        self.status_text = None
        self.tx_slider = None
        self.ty_slider = None
        self.yaw_slider = None
        self.axis_radio = None
        self.default_view = None

    def _decimate_polyline(self, polyline):
        polyline = np.asarray(polyline, dtype=np.float64)
        if len(polyline) <= 2:
            return polyline
        step = max(int(self.args.display_line_decimation), 1)
        decimated = polyline[::step]
        if not np.array_equal(decimated[-1], polyline[-1]):
            decimated = np.concatenate([decimated, polyline[-1:]], axis=0)
        return decimated

    def _filter_polyline_to_roi(self, polyline):
        polyline = np.asarray(polyline, dtype=np.float64)
        if len(polyline) < 2:
            return None
        bbox_min = polyline.min(axis=0)
        bbox_max = polyline.max(axis=0)
        if (
            bbox_max[0] < self.roi_min[0]
            or bbox_min[0] > self.roi_max[0]
            or bbox_max[1] < self.roi_min[1]
            or bbox_min[1] > self.roi_max[1]
        ):
            return None

        inside = (
            (polyline[:, 0] >= self.roi_min[0])
            & (polyline[:, 0] <= self.roi_max[0])
            & (polyline[:, 1] >= self.roi_min[1])
            & (polyline[:, 1] <= self.roi_max[1])
        )
        keep = inside.copy()
        keep[:-1] |= inside[1:]
        keep[1:] |= inside[:-1]
        clipped = polyline[keep]
        if len(clipped) < 2:
            return None
        return clipped

    def _initial_axis_order(self):
        if self.args.transform_json:
            with open(self.args.transform_json, "r", encoding="utf-8") as file:
                payload = json.load(file)
            axis_order = payload.get("axis_order")
            if axis_order in self.axis_orders:
                self._preloaded_transform = payload
                return axis_order
        self._preloaded_transform = None
        if self.args.axis_order == "auto":
            ranked = sorted(
                self.auto_results.items(),
                key=lambda item: item[1]["transform"]["cost"],
            )
            return ranked[0][0]
        return self.args.axis_order

    def _initial_transform(self):
        if getattr(self, "_preloaded_transform", None):
            map_to_lidar = self._preloaded_transform.get("map_to_lidar", {})
            return {
                "tx": float(map_to_lidar.get("tx", 0.0)),
                "ty": float(map_to_lidar.get("ty", 0.0)),
                "yaw_rad": float(map_to_lidar.get("yaw_rad", 0.0)),
            }
        return dict(self.auto_results[self.axis_order]["transform"])

    def _current_cost(self):
        map_points = self.preview_map_points_by_order[self.axis_order]
        transformed = transform_points_se2(
            map_points,
            self.current_transform["tx"],
            self.current_transform["ty"],
            self.current_transform["yaw_rad"],
        )
        return float(alignment_cost(self.cost_lidar_points, transformed))

    def _set_slider_ranges(self, tx_center, ty_center, yaw_center_deg):
        self.tx_slider.valmin = tx_center - self.args.tx_range
        self.tx_slider.valmax = tx_center + self.args.tx_range
        self.ty_slider.valmin = ty_center - self.args.ty_range
        self.ty_slider.valmax = ty_center + self.args.ty_range
        self.yaw_slider.valmin = yaw_center_deg - self.args.yaw_range_deg
        self.yaw_slider.valmax = yaw_center_deg + self.args.yaw_range_deg
        self.tx_slider.ax.set_xlim(self.tx_slider.valmin, self.tx_slider.valmax)
        self.ty_slider.ax.set_xlim(self.ty_slider.valmin, self.ty_slider.valmax)
        self.yaw_slider.ax.set_xlim(self.yaw_slider.valmin, self.yaw_slider.valmax)

    def _update_plot(self, save_path=None):
        yaw_deg = math.degrees(self.current_transform["yaw_rad"])
        transformed_segments = []
        for centerline in self.display_centerlines_by_order[self.axis_order]:
            transformed_centerline = transform_points_se2(
                centerline,
                self.current_transform["tx"],
                self.current_transform["ty"],
                self.current_transform["yaw_rad"],
            )
            clipped = self._filter_polyline_to_roi(transformed_centerline)
            if clipped is not None:
                transformed_segments.append(clipped)
        self.map_artist.set_segments(transformed_segments)
        cost = self._current_cost()
        self.status_text.set_text(
            format_status(
                self.axis_order,
                self.current_transform["tx"],
                self.current_transform["ty"],
                yaw_deg,
                cost,
                save_path=save_path,
            )
        )
        self.figure.canvas.draw_idle()

    def _zoom_axes(self, scale_factor, center_x=None, center_y=None):
        x_min, x_max = self.ax_plot.get_xlim()
        y_min, y_max = self.ax_plot.get_ylim()
        if center_x is None or not np.isfinite(center_x):
            center_x = (x_min + x_max) * 0.5
        if center_y is None or not np.isfinite(center_y):
            center_y = (y_min + y_max) * 0.5

        new_half_width = (x_max - x_min) * 0.5 * scale_factor
        new_half_height = (y_max - y_min) * 0.5 * scale_factor
        min_half_extent = 2.0
        new_half_width = max(new_half_width, min_half_extent)
        new_half_height = max(new_half_height, min_half_extent)

        self.ax_plot.set_xlim(center_x - new_half_width, center_x + new_half_width)
        self.ax_plot.set_ylim(center_y - new_half_height, center_y + new_half_height)
        self.figure.canvas.draw_idle()

    def _reset_view(self):
        if self.default_view is None:
            return
        x_limits, y_limits = self.default_view
        self.ax_plot.set_xlim(*x_limits)
        self.ax_plot.set_ylim(*y_limits)
        self.figure.canvas.draw_idle()

    def _sync_sliders_from_transform(self):
        yaw_deg = math.degrees(self.current_transform["yaw_rad"])
        self._set_slider_ranges(
            self.current_transform["tx"],
            self.current_transform["ty"],
            yaw_deg,
        )
        self.tx_slider.set_val(self.current_transform["tx"])
        self.ty_slider.set_val(self.current_transform["ty"])
        self.yaw_slider.set_val(yaw_deg)

    def _on_slider_change(self, _value):
        self.current_transform["tx"] = float(self.tx_slider.val)
        self.current_transform["ty"] = float(self.ty_slider.val)
        self.current_transform["yaw_rad"] = math.radians(float(self.yaw_slider.val))
        self._update_plot()

    def _load_auto_transform(self, axis_order):
        auto_transform = self.auto_results[axis_order]["transform"]
        self.current_transform = {
            "tx": float(auto_transform["tx"]),
            "ty": float(auto_transform["ty"]),
            "yaw_rad": float(auto_transform["yaw_rad"]),
        }
        self._sync_sliders_from_transform()
        self._update_plot()

    def _on_reset(self, _event):
        self.current_transform = dict(self.initial_transform)
        self._sync_sliders_from_transform()
        self._update_plot()

    def _on_auto(self, _event):
        self._load_auto_transform(self.axis_order)

    def _on_axis_change(self, label):
        label = str(label)
        if label == self.axis_order:
            return
        self.axis_order = label
        self.initial_transform = dict(self.auto_results[self.axis_order]["transform"])
        self._load_auto_transform(self.axis_order)

    def _save_outputs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        lanelets = self.lanelets_by_order[self.axis_order]
        graph = self.graph_by_order[self.axis_order]
        transform = {
            "tx": float(self.current_transform["tx"]),
            "ty": float(self.current_transform["ty"]),
            "yaw_rad": float(self.current_transform["yaw_rad"]),
            "yaw_deg": float(math.degrees(self.current_transform["yaw_rad"])),
            "cost": float(self._current_cost()),
        }
        aligned_lanelets = serialize_lanelets(lanelets, graph, transform)
        transform_payload = {
            "osm_path": osp.abspath(self.args.osm_path),
            "ann_file": osp.abspath(self.args.ann_file),
            "qa_json": osp.abspath(self.args.qa_json) if self.args.qa_json else None,
            "axis_order": self.axis_order,
            "transform_type": "SE(2)",
            "map_to_lidar": {
                "tx": transform["tx"],
                "ty": transform["ty"],
                "yaw_rad": transform["yaw_rad"],
                "yaw_deg": transform["yaw_deg"],
            },
            "cost": transform["cost"],
            "lidar_reference": self.lidar_meta,
            "lanelet_count": len(lanelets),
            "saved_from": "align_lanelet_to_lidar_gui.py",
        }
        lane_graph_payload = {
            "axis_order": self.axis_order,
            "transform_type": "SE(2)",
            "map_to_lidar": transform_payload["map_to_lidar"],
            "lane_graph": graph,
        }

        transform_path = osp.join(output_dir, "transform.json")
        centerline_path = osp.join(output_dir, "aligned_centerlines.json")
        graph_path = osp.join(output_dir, "lane_graph.json")

        with open(transform_path, "w", encoding="utf-8") as file:
            json.dump(transform_payload, file, ensure_ascii=False, indent=2)
        with open(centerline_path, "w", encoding="utf-8") as file:
            json.dump(aligned_lanelets, file, ensure_ascii=False, indent=2)
        with open(graph_path, "w", encoding="utf-8") as file:
            json.dump(lane_graph_payload, file, ensure_ascii=False, indent=2)
        self._update_plot(save_path=output_dir)

    def _on_save(self, _event):
        self._save_outputs(self.args.output_dir)

    def _on_key_press(self, event):
        translation_step = 1.0
        yaw_step_deg = 0.5
        if event.key and "shift" in event.key:
            translation_step = 5.0
            yaw_step_deg = 2.0
        key = event.key.replace("shift+", "") if event.key else ""
        if key in ("+", "=", "plus"):
            self._zoom_axes(scale_factor=0.8)
            return
        elif key in ("-", "minus"):
            self._zoom_axes(scale_factor=1.25)
            return
        elif key == "0":
            self._reset_view()
            return
        elif key == "left":
            self.current_transform["tx"] -= translation_step
        elif key == "right":
            self.current_transform["tx"] += translation_step
        elif key == "down":
            self.current_transform["ty"] -= translation_step
        elif key == "up":
            self.current_transform["ty"] += translation_step
        elif key == "q":
            self.current_transform["yaw_rad"] = math.radians(
                wrap_angle_deg(math.degrees(self.current_transform["yaw_rad"]) - yaw_step_deg)
            )
        elif key == "e":
            self.current_transform["yaw_rad"] = math.radians(
                wrap_angle_deg(math.degrees(self.current_transform["yaw_rad"]) + yaw_step_deg)
            )
        elif key == "a":
            self._load_auto_transform(self.axis_order)
            return
        elif key == "r":
            self._on_reset(None)
            return
        elif key == "s":
            self._save_outputs(self.args.output_dir)
            return
        else:
            return
        self._sync_sliders_from_transform()
        self._update_plot()

    def _on_scroll(self, event):
        if event.inaxes != self.ax_plot:
            return
        if event.button == "up":
            self._zoom_axes(scale_factor=0.85, center_x=event.xdata, center_y=event.ydata)
        elif event.button == "down":
            self._zoom_axes(scale_factor=1.18, center_x=event.xdata, center_y=event.ydata)

    def run(self):
        figure = plt.figure(figsize=(16, 10))
        figure.canvas.manager.set_window_title("Lanelet to LiDAR Manual Alignment")
        self.figure = figure

        self.ax_plot = figure.add_axes([0.05, 0.15, 0.7, 0.8])
        self.ax_plot.set_title("Manual SE(2) Alignment")
        self.ax_plot.set_xlabel("X (LiDAR meters)")
        self.ax_plot.set_ylabel("Y (LiDAR meters)")
        self.ax_plot.set_aspect("equal", adjustable="box")
        self.ax_plot.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        self.lidar_artist = self.ax_plot.scatter(
            self.display_lidar_points[:, 0],
            self.display_lidar_points[:, 1],
            s=7,
            c="#444444",
            alpha=0.55,
            label="LiDAR reference",
            zorder=3,
        )
        initial_segments = []
        for centerline in self.display_centerlines_by_order[self.axis_order]:
            transformed_centerline = transform_points_se2(
                centerline,
                self.current_transform["tx"],
                self.current_transform["ty"],
                self.current_transform["yaw_rad"],
            )
            clipped = self._filter_polyline_to_roi(transformed_centerline)
            if clipped is not None:
                initial_segments.append(clipped)
        self.map_artist = LineCollection(
            initial_segments,
            colors="#1f77b4",
            linewidths=1.3,
            alpha=0.75,
            zorder=1,
        )
        self.ax_plot.add_collection(self.map_artist)

        if initial_segments:
            initial_map_points = np.concatenate(initial_segments, axis=0)
            bounds_points = np.concatenate([self.display_lidar_points, initial_map_points], axis=0)
        else:
            bounds_points = self.display_lidar_points
        center, span = compute_bounds(bounds_points)
        x_limits = (center[0] - span[0] * 0.6, center[0] + span[0] * 0.6)
        y_limits = (center[1] - span[1] * 0.6, center[1] + span[1] * 0.6)
        self.ax_plot.set_xlim(*x_limits)
        self.ax_plot.set_ylim(*y_limits)
        self.default_view = (x_limits, y_limits)
        self.ax_plot.legend(loc="upper right")

        slider_face = "#f0f0f0"
        tx_ax = figure.add_axes([0.05, 0.08, 0.55, 0.03], facecolor=slider_face)
        ty_ax = figure.add_axes([0.05, 0.045, 0.55, 0.03], facecolor=slider_face)
        yaw_ax = figure.add_axes([0.05, 0.01, 0.55, 0.03], facecolor=slider_face)

        yaw_deg = math.degrees(self.current_transform["yaw_rad"])
        self.tx_slider = Slider(
            tx_ax,
            "tx",
            self.current_transform["tx"] - self.args.tx_range,
            self.current_transform["tx"] + self.args.tx_range,
            valinit=self.current_transform["tx"],
        )
        self.ty_slider = Slider(
            ty_ax,
            "ty",
            self.current_transform["ty"] - self.args.ty_range,
            self.current_transform["ty"] + self.args.ty_range,
            valinit=self.current_transform["ty"],
        )
        self.yaw_slider = Slider(
            yaw_ax,
            "yaw(deg)",
            yaw_deg - self.args.yaw_range_deg,
            yaw_deg + self.args.yaw_range_deg,
            valinit=yaw_deg,
        )
        self.tx_slider.on_changed(self._on_slider_change)
        self.ty_slider.on_changed(self._on_slider_change)
        self.yaw_slider.on_changed(self._on_slider_change)

        axis_ax = figure.add_axes([0.8, 0.72, 0.16, 0.16], facecolor=slider_face)
        self.axis_radio = RadioButtons(axis_ax, self.axis_orders, active=self.axis_orders.index(self.axis_order))
        self.axis_radio.on_clicked(self._on_axis_change)
        axis_ax.set_title("Axis order")

        auto_ax = figure.add_axes([0.8, 0.58, 0.16, 0.05])
        reset_ax = figure.add_axes([0.8, 0.51, 0.16, 0.05])
        save_ax = figure.add_axes([0.8, 0.44, 0.16, 0.05])
        auto_button = Button(auto_ax, "Auto Init")
        reset_button = Button(reset_ax, "Reset")
        save_button = Button(save_ax, f"Save -> {self.args.output_dir}")
        auto_button.on_clicked(self._on_auto)
        reset_button.on_clicked(self._on_reset)
        save_button.on_clicked(self._on_save)

        help_lines = [
            "Keys:",
            "arrows = move",
            "shift+arrows = move faster",
            "q / e = rotate",
            "+ / - = zoom",
            "0 = reset view",
            "wheel = zoom to cursor",
            "a = auto init",
            "r = reset",
            "s = save",
        ]
        figure.text(0.79, 0.26, "\n".join(help_lines), fontsize=10, family="monospace")

        meta_lines = [
            f"frames={self.lidar_meta['frame_count']}",
            f"boxes={self.lidar_meta['box_count']}",
            f"moving_boxes={self.lidar_meta['moving_box_count']}",
            f"display_lidar={len(self.display_lidar_points)}",
        ]
        figure.text(0.79, 0.17, "\n".join(meta_lines), fontsize=10, family="monospace")

        self.status_text = figure.text(0.05, 0.95, "", fontsize=11, family="monospace")
        figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._update_plot()
        plt.show()


def load_axis_data(args):
    start_time = time.time()
    total_steps = 4
    print_progress("Loading SunLakes infos...", step=1, total_steps=total_steps, start_time=start_time)
    infos = load_infos(args.ann_file)
    print_progress(f"Loaded {len(infos)} samples from ann file.", step=1, total_steps=total_steps, start_time=start_time)

    print_progress("Loading QA token filter...", step=2, total_steps=total_steps, start_time=start_time)
    qa_tokens = load_qa_tokens(args.qa_json) if args.qa_json else None
    if qa_tokens is not None:
        print_progress(
            f"Loaded {len(qa_tokens)} QA frame tokens.",
            step=2,
            total_steps=total_steps,
            start_time=start_time,
        )
    else:
        print_progress("No QA subset filter provided.", step=2, total_steps=total_steps, start_time=start_time)

    scene_ids = set(args.scene_ids.split(",")) if args.scene_ids else None
    allowed_classes = [item.strip() for item in args.vehicle_classes.split(",") if item.strip()]
    print_progress("Extracting LiDAR reference points...", step=3, total_steps=total_steps, start_time=start_time)
    lidar_points, lidar_meta = extract_lidar_reference_points(
        infos,
        allowed_classes=allowed_classes,
        speed_threshold=args.speed_threshold,
        qa_tokens=qa_tokens,
        scene_ids=scene_ids,
        max_points=args.max_gt_points,
    )
    print_progress(
        (
            f"Prepared {len(lidar_points)} LiDAR reference points from "
            f"{lidar_meta['frame_count']} frames / {lidar_meta['moving_box_count']} moving boxes."
        ),
        step=3,
        total_steps=total_steps,
        start_time=start_time,
    )

    axis_orders = ["lonlat", "latlon"] if args.axis_order == "auto" else [args.axis_order]
    lanelets_by_order = {}
    graph_by_order = {}
    print_progress(
        f"Parsing OSM lanelets for axis orders: {', '.join(axis_orders)}",
        step=4,
        total_steps=total_steps,
        start_time=start_time,
    )
    for axis_order in axis_orders:
        lanelets = parse_lanelet_map(args.osm_path, axis_order=axis_order)
        if not lanelets:
            raise ValueError(f"No lanelets found in {args.osm_path} for axis order {axis_order}.")
        lanelets_by_order[axis_order] = lanelets
        graph_by_order[axis_order] = infer_lane_graph(lanelets)
        print_progress(
            f"Axis {axis_order}: parsed {len(lanelets)} lanelets.",
            step=4,
            total_steps=total_steps,
            start_time=start_time,
        )
    print_progress("Input preparation finished.", step=4, total_steps=total_steps, start_time=start_time)
    return lidar_points, lidar_meta, lanelets_by_order, graph_by_order


def build_auto_results(args, lidar_points, lanelets_by_order):
    start_time = time.time()
    class AutoArgs:
        pass

    auto_args = AutoArgs()
    auto_args.coarse_lidar_sample = args.coarse_lidar_sample
    auto_args.coarse_map_sample = args.coarse_map_sample
    auto_args.fine_lidar_sample = args.fine_lidar_sample
    auto_args.fine_map_sample = args.fine_map_sample

    if args.axis_order == "auto":
        total_steps = len(lanelets_by_order) + 1
        print_progress("Running automatic initialization across axis orders...", step=1, total_steps=total_steps, start_time=start_time)
        best_result = select_best_axis_order(
            args.axis_order,
            lanelets_by_order,
            lidar_points,
            anchors_json=args.anchors_json,
            args=auto_args,
        )
        auto_results = {}
        current_step = 2
        for axis_order in lanelets_by_order:
            print_progress(
                f"Refining auto initialization for axis order {axis_order}...",
                step=current_step,
                total_steps=total_steps,
                start_time=start_time,
            )
            axis_best = select_best_axis_order(
                axis_order,
                lanelets_by_order,
                lidar_points,
                anchors_json=args.anchors_json,
                args=auto_args,
            )
            auto_results[axis_order] = axis_best
            print_progress(
                (
                    f"Axis {axis_order}: tx={axis_best['transform']['tx']:.2f}, "
                    f"ty={axis_best['transform']['ty']:.2f}, "
                    f"yaw={math.degrees(axis_best['transform']['yaw_rad']):.2f} deg, "
                    f"cost={axis_best['transform']['cost']:.3f}"
                ),
                step=current_step,
                total_steps=total_steps,
                start_time=start_time,
            )
            current_step += 1
        auto_results[best_result["axis_order"]] = best_result
        print_progress(
            f"Auto initialization complete. Best axis order: {best_result['axis_order']}.",
            step=total_steps,
            total_steps=total_steps,
            start_time=start_time,
        )
        return auto_results

    print_progress("Running automatic initialization...", step=1, total_steps=1, start_time=start_time)
    result = select_best_axis_order(
        args.axis_order,
        lanelets_by_order,
        lidar_points,
        anchors_json=args.anchors_json,
        args=auto_args,
    )
    print_progress(
        (
            f"Auto initialization complete: tx={result['transform']['tx']:.2f}, "
            f"ty={result['transform']['ty']:.2f}, "
            f"yaw={math.degrees(result['transform']['yaw_rad']):.2f} deg, "
            f"cost={result['transform']['cost']:.3f}"
        ),
        step=1,
        total_steps=1,
        start_time=start_time,
    )
    return {
        args.axis_order: result
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print_progress("Starting manual alignment GUI...")
    lidar_points, lidar_meta, lanelets_by_order, graph_by_order = load_axis_data(args)
    auto_results = build_auto_results(args, lidar_points, lanelets_by_order)
    print_progress("Launching interactive window...")

    gui = ManualAlignGUI(
        args=args,
        lidar_points=lidar_points,
        lidar_meta=lidar_meta,
        lanelets_by_order=lanelets_by_order,
        graph_by_order=graph_by_order,
        auto_results=auto_results,
    )
    gui.run()


if __name__ == "__main__":
    main()
