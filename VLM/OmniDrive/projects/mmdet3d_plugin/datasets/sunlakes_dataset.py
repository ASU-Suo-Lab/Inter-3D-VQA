import json
import os
import os.path as osp
import pathlib
import pickle
from collections import defaultdict

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type
from mmdet3d.datasets.pipelines import Compose
from torch.utils.data import Dataset


class _WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)


@DATASETS.register_module()
class SunLakesSequentialDataset(Dataset):
    CLASSES = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    )

    def __init__(
        self,
        ann_file,
        pipeline,
        qa_json,
        data_root=".",
        seq_by="scene_id",
        lane_file=None,
        image_root=None,
        lidar_root=None,
        lane_n_control=11,
        classes=None,
        modality=None,
        test_mode=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
        **kwargs,
    ):
        self.ann_file = osp.abspath(ann_file)
        self.qa_json = osp.abspath(qa_json)
        self.data_root = osp.abspath(data_root)
        self.seq_by = seq_by
        self.lane_file = self._resolve_optional_path(lane_file)
        self.image_root = self._resolve_optional_path(image_root)
        self.lidar_root = self._resolve_optional_path(lidar_root)
        self.lane_n_control = lane_n_control
        self.modality = modality or {}
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.load_interval = load_interval
        if classes is not None:
            self.CLASSES = tuple(classes)
        self.pipeline = Compose(pipeline)

        self.qa_by_frame = self._load_qa_groups(self.qa_json)
        self.lane_lookup = self._load_lane_lookup(self.lane_file)
        self.data_infos = self._load_infos(self.ann_file)
        self.flag = self._build_sequence_flags()

    def _load_qa_groups(self, qa_json):
        with open(qa_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        grouped = defaultdict(list)
        for qa_pair in payload["qa_pairs"]:
            grouped[qa_pair["frame_token"]].append(qa_pair)
        for frame_token in grouped:
            grouped[frame_token] = sorted(
                grouped[frame_token], key=lambda item: item["question_id"]
            )
        return dict(grouped)

    def _load_infos(self, ann_file):
        with open(ann_file, "rb") as file:
            infos = _WindowsPathUnpickler(file).load()
        if isinstance(infos, dict) and "infos" in infos:
            infos = infos["infos"]

        qa_tokens = set(self.qa_by_frame)
        filtered = []
        for info in infos:
            if info["token"] in qa_tokens:
                if self._has_usable_cams(info):
                    filtered.append(info)
        filtered.sort(key=lambda item: (item[self.seq_by], item["timestamp"], item["token"]))
        if self.load_interval > 1:
            filtered = filtered[:: self.load_interval]
        return filtered

    def _has_usable_cams(self, info):
        cams = info.get("cams") or {}
        cam_keys = info.get("cam_keys") or list(cams.keys())
        return bool(cams) and bool(cam_keys)

    def _resolve_optional_path(self, raw_path):
        if not raw_path:
            return None
        if osp.isabs(raw_path):
            return raw_path
        return osp.abspath(raw_path)

    def _build_sequence_flags(self):
        flags = []
        scene_to_flag = {}
        next_flag = 0
        for info in self.data_infos:
            scene_id = info[self.seq_by]
            if scene_id not in scene_to_flag:
                scene_to_flag[scene_id] = next_flag
                next_flag += 1
            flags.append(scene_to_flag[scene_id])
        return np.asarray(flags, dtype=np.int64)

    def _resolve_path(self, raw_path):
        raw_path = str(raw_path)
        if osp.isabs(raw_path):
            return raw_path
        candidates = [
            osp.abspath(osp.join(self.data_root, raw_path)),
            osp.abspath(osp.join(osp.dirname(self.ann_file), raw_path)),
            osp.abspath(raw_path),
        ]
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        remapped = self._remap_sunlakes_path(raw_path)
        if remapped is not None and osp.exists(remapped):
            return remapped
        return candidates[0]

    def _remap_sunlakes_path(self, raw_path):
        normalized = raw_path.replace("\\", "/")
        if "resource/" in normalized and self.image_root is not None:
            suffix = normalized.split("resource/", 1)[1].lstrip("/")
            candidate = osp.abspath(osp.join(self.image_root, *suffix.split("/")))
            if osp.exists(candidate):
                return candidate
        if "lidar_pc_with_ts/" in normalized and self.lidar_root is not None:
            suffix = normalized.split("lidar_pc_with_ts/", 1)[1].lstrip("/")
            suffix_parts = suffix.split("/")
            candidate = osp.abspath(osp.join(self.lidar_root, *suffix_parts))
            if osp.exists(candidate):
                return candidate
            # SunLakes lidar label directory changed after re-export.
            remapped_parts = [
                "128_128b_label" if part == "16_16b_label" else part for part in suffix_parts
            ]
            candidate = osp.abspath(osp.join(self.lidar_root, *remapped_parts))
            if osp.exists(candidate):
                return candidate
        return None

    def _normalize_lane_pts(self, payload):
        if payload is None:
            return []
        if isinstance(payload, dict):
            if "lane_pts" in payload:
                return self._normalize_lane_pts(payload["lane_pts"])
            if "centerline_xy_lidar" in payload:
                return self._normalize_lane_pts(payload["centerline_xy_lidar"])
            if "points" in payload:
                return self._normalize_lane_pts(payload["points"])
            if "centerlines" in payload:
                return self._normalize_lane_pts(payload["centerlines"])
            lane_pts = []
            for value in payload.values():
                lane_pts.extend(self._normalize_lane_pts(value))
            return lane_pts
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()
        if not isinstance(payload, list):
            return []
        if not payload:
            return []
        first = payload[0]
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 2 and isinstance(first[0], (int, float, np.floating)):
            lane = np.asarray(payload, dtype=np.float32)
            return [self._resample_lane_points(lane).tolist()]
        lane_pts = []
        for item in payload:
            lane_pts.extend(self._normalize_lane_pts(item))
        return lane_pts

    def _resample_lane_points(self, lane):
        lane = np.asarray(lane, dtype=np.float32)
        if lane.ndim != 2 or lane.shape[0] == 0:
            return np.zeros((self.lane_n_control, 3), dtype=np.float32)
        if lane.shape[1] == 2:
            lane = np.concatenate(
                [lane, np.zeros((lane.shape[0], 1), dtype=np.float32)], axis=1
            )
        else:
            lane = lane[:, :3]

        if lane.shape[0] == 1:
            return np.repeat(lane, self.lane_n_control, axis=0)

        deltas = np.linalg.norm(np.diff(lane[:, :2], axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(deltas, dtype=np.float32)])
        total = float(cumulative[-1])
        if total <= 1e-6:
            return np.repeat(lane[:1], self.lane_n_control, axis=0)

        sample_distances = np.linspace(0.0, total, self.lane_n_control, dtype=np.float32)
        sampled = np.zeros((self.lane_n_control, 3), dtype=np.float32)
        for dim in range(3):
            sampled[:, dim] = np.interp(sample_distances, cumulative, lane[:, dim])
        return sampled

    def _load_lane_lookup(self, lane_file):
        if lane_file is None:
            return {}
        if not osp.exists(lane_file):
            raise FileNotFoundError(f"Lane file not found: {lane_file}")
        if lane_file.endswith(".json"):
            with open(lane_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
        else:
            with open(lane_file, "rb") as file:
                payload = _WindowsPathUnpickler(file).load()

        if isinstance(payload, dict):
            if "aligned_centerlines" in payload:
                payload = payload["aligned_centerlines"]
            elif "lanelets" in payload:
                payload = payload["lanelets"]
            elif "lane_centerline" in payload:
                payload = payload["lane_centerline"]

        if isinstance(payload, list):
            return {"__all__": self._normalize_lane_pts(payload)}

        lane_lookup = {}
        for key, value in payload.items():
            lane_lookup[str(key)] = self._normalize_lane_pts(value)
        return lane_lookup

    def _get_lane_pts(self, info):
        if "lane_pts" in info:
            lane_pts = self._normalize_lane_pts(info["lane_pts"])
            if lane_pts:
                return lane_pts
        for lookup_key in (info["token"], info.get(self.seq_by), "__all__"):
            if lookup_key in self.lane_lookup:
                return self.lane_lookup[lookup_key]
        return []

    def _build_gt_boxes_3d(self, info):
        gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
        if gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 9)
        if gt_boxes.ndim != 2:
            gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])
        if gt_boxes.shape[1] == 7:
            velocities = np.asarray(info.get("gt_boxes_velocity", []), dtype=np.float32)
            if velocities.shape[0] != gt_boxes.shape[0]:
                velocities = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
            gt_boxes = np.concatenate([gt_boxes, velocities], axis=1)
        elif gt_boxes.shape[1] == 8:
            gt_boxes = np.concatenate([gt_boxes, np.zeros((gt_boxes.shape[0], 1), dtype=np.float32)], axis=1)
        elif gt_boxes.shape[1] > 9:
            gt_boxes = gt_boxes[:, :9]
        return LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))

    def _build_ann_info(self, info):
        gt_names = np.asarray(info.get("gt_names", []), dtype=object)
        lane_pts = self._get_lane_pts(info)
        return dict(
            gt_bboxes_3d=self._build_gt_boxes_3d(info),
            gt_names_3d=gt_names,
            lane_pts=lane_pts,
        )

    def __len__(self):
        return len(self.data_infos)

    def pre_pipeline(self, results):
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def get_data_info(self, index):
        info = self.data_infos[index]
        cam_keys = info.get("cam_keys") or list(info["cams"].keys())
        image_paths = []
        lidar2img = []
        intrinsics = []
        extrinsics = []
        for cam_key in cam_keys:
            cam_info = info["cams"][cam_key]
            image_path = self._resolve_path(cam_info["image_paths"])
            if not osp.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image_paths.append(image_path)
            lidar2img.append(cam_info["lidar2image"])
            intrinsics.append(cam_info["camera_intrinsics"])
            extrinsics.append(cam_info["lidar2camera"])

        pts_filename = self._resolve_path(info["lidar_path"])
        if not osp.exists(pts_filename):
            raise FileNotFoundError(f"LiDAR file not found: {pts_filename}")

        prev_exists = False
        if index > 0:
            prev_exists = self.data_infos[index - 1][self.seq_by] == info[self.seq_by]

        input_dict = dict(
            sample_idx=info["token"],
            token=info["token"],
            scene_id=info[self.seq_by],
            scene_token=info[self.seq_by],
            timestamp=float(info["timestamp"]),
            img_timestamp=[float(info["timestamp"])] * len(image_paths),
            pts_filename=pts_filename,
            img_filename=image_paths,
            lidar2img=lidar2img,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            prev_exists=prev_exists,
            cam_infos=info["cams"],
        )
        if not self.test_mode:
            ann_info = self._build_ann_info(info)
            input_dict.update(
                gt_bboxes_3d=ann_info["gt_bboxes_3d"],
                gt_names_3d=ann_info["gt_names_3d"],
                lane_pts=ann_info["lane_pts"],
                ann_info=ann_info,
            )
        return input_dict

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)
        return self.prepare_train_data(index)

    def evaluate(self, results, **kwargs):
        return {}


@DATASETS.register_module()
class SunLakesSequentialDatasetV5(SunLakesSequentialDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        qa_json,
        data_root=".",
        seq_by="scene_id",
        lane_file=None,
        image_root=None,
        lidar_root=None,
        lane_n_control=11,
        classes=None,
        modality=None,
        test_mode=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
        **kwargs,
    ):
        if image_root is not None or lidar_root is not None:
            raise ValueError(
                "SunLakesSequentialDatasetV5 expects Linux-normalized infos and "
                "does not accept image_root/lidar_root remapping."
            )
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            qa_json=qa_json,
            data_root=data_root,
            seq_by=seq_by,
            lane_file=lane_file,
            image_root=None,
            lidar_root=None,
            lane_n_control=lane_n_control,
            classes=classes,
            modality=modality,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            box_type_3d=box_type_3d,
            load_interval=load_interval,
            **kwargs,
        )

    def _load_infos(self, ann_file):
        with open(ann_file, "rb") as file:
            infos = pickle.load(file)
        if isinstance(infos, dict) and "infos" in infos:
            infos = infos["infos"]
        if not isinstance(infos, list):
            raise TypeError(f"Expected list or dict['infos'] in {ann_file}, got {type(infos).__name__}")

        qa_tokens = set(self.qa_by_frame)
        filtered = []
        for info in infos:
            token = str(info["token"])
            if token in qa_tokens and self._has_usable_cams(info):
                filtered.append(info)
        filtered.sort(key=lambda item: (item[self.seq_by], item["timestamp"], item["token"]))
        if self.load_interval > 1:
            filtered = filtered[:: self.load_interval]
        return filtered

    def _resolve_path(self, raw_path):
        path = str(raw_path)
        if not osp.isabs(path):
            raise ValueError(
                "SunLakesSequentialDatasetV5 requires absolute normalized paths. "
                f"Got: {path}"
            )
        normalized = osp.abspath(path)
        if not osp.exists(normalized):
            raise FileNotFoundError(f"Required asset not found: {normalized}")
        return normalized


@DATASETS.register_module()
class SunLakesSequentialQATrainDatasetV5(SunLakesSequentialDatasetV5):
    def __init__(
        self,
        ann_file,
        pipeline,
        qa_json,
        data_root=".",
        seq_by="scene_id",
        lane_file=None,
        image_root=None,
        lidar_root=None,
        lane_n_control=11,
        classes=None,
        modality=None,
        test_mode=False,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
        **kwargs,
    ):
        if test_mode:
            raise ValueError("SunLakesSequentialQATrainDatasetV5 is train-only and does not support test_mode=True.")
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            qa_json=qa_json,
            data_root=data_root,
            seq_by=seq_by,
            lane_file=lane_file,
            image_root=image_root,
            lidar_root=lidar_root,
            lane_n_control=lane_n_control,
            classes=classes,
            modality=modality,
            test_mode=False,
            filter_empty_gt=filter_empty_gt,
            box_type_3d=box_type_3d,
            load_interval=load_interval,
            **kwargs,
        )
        self.qa_samples = self._build_qa_samples()
        self.flag = self._build_qa_flags()

    def _build_qa_samples(self):
        qa_samples = []
        for info_index, info in enumerate(self.data_infos):
            frame_token = str(info["token"])
            qa_pairs = self.qa_by_frame.get(frame_token)
            if not qa_pairs:
                continue
            for qa_pair in qa_pairs:
                qa_samples.append(
                    dict(
                        info_index=info_index,
                        qa_pair=qa_pair,
                        scene_id=str(info[self.seq_by]),
                    )
                )
        return qa_samples

    def _build_qa_flags(self):
        flags = []
        scene_to_flag = {}
        next_flag = 0
        for sample in self.qa_samples:
            scene_id = sample["scene_id"]
            if scene_id not in scene_to_flag:
                scene_to_flag[scene_id] = next_flag
                next_flag += 1
            flags.append(scene_to_flag[scene_id])
        return np.asarray(flags, dtype=np.int64)

    def __len__(self):
        return len(self.qa_samples)

    def prepare_train_data(self, index):
        qa_sample = self.qa_samples[index]
        input_dict = self.get_data_info(qa_sample["info_index"])
        input_dict["qa_pair"] = qa_sample["qa_pair"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
