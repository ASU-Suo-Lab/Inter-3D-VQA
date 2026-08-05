from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import re


@dataclass(frozen=True)
class V5TemplateSpec:
    template_id: str
    chapter: str
    section: str
    title: str
    question_schema: str
    answer_schema: str
    required_signals: Tuple[str, ...]
    enabled_by_default: bool
    disabled_reason: Optional[str] = None


_WORKER_RUNTIME: Optional["IntersectionQAGeneratorV5Runtime"] = None


def _init_v5_worker(pkl_path: str, subtemplate_patch_style: str) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = IntersectionQAGeneratorV5Runtime(pkl_path, subtemplate_patch_style=subtemplate_patch_style)


def _generate_v5_frame_worker(frame_index: int, max_per_type: int, keyframe_fps: float) -> List[Dict]:
    if _WORKER_RUNTIME is None:
        raise RuntimeError("V5 worker runtime is not initialized.")
    _WORKER_RUNTIME.active_keyframe_fps = keyframe_fps
    frame = _WORKER_RUNTIME.infos[frame_index]
    return _WORKER_RUNTIME.generate_frame(frame, max_per_type)


class IntersectionQAGeneratorV5Runtime:
    DEFAULT_OUTPUT_NAME = "intersection_qa_pairs_v5.json"
    DEFAULT_MAX_PER_TYPE = 100
    PROMPT_METADATA_VERSION = "v5_simple_prompt_v3"
    REF_SCHEMA_TOKEN = "<ID,f,X,Y,image_name_1,X1_1,Y1_1,...,image_name_n,X1_n,Y1_n>"
    BUCKET_FIELD_BY_TEMPLATE = {
        "1_1_1_fine_type": "object_type",
        "1_1_2_side_exists": "exists",
        "1_1_3_side_count": "object_type",
        "1_1_4_relative_neighbor_type": "rel_dir",
        "1_2_1_size_bucket": "size_bucket",
        "1_3_1_weather": "weather",
        "1_3_2_vehicle_signal_state": "signal_state",
        "2_1_1_stopline_distance": "object_type",
        "2_1_2_ped_to_far_edge": "crosswalk",
        "2_1_3_participant_distance": "pair_type",
        "2_1_4_nearest_vehicle_to_ped": "direction",
        "2_2_1_lane_function": "lane_function",
        "2_2_2_ped_zone": "ped_zone",
        "2_2_3_left_turn_queue_count": "lane_function",
        "2_2_4_stopline_back_5m_count": "side",
        "2_2_5_longest_queue_lane": "lane_function",
        "2_2_6_crosswalk_blocking": "crosswalk_blocked",
        "3_1_1_current_motion_state": "motion_state",
        "3_1_2_vehicle_maneuver": "maneuver",
        "3_3_1_safe_following": "is_safe",
        "3_3_2_likely_long_queue_lane": "lane_function",
        "3_2_2_future_region": "future_region",
        "3_2_3_waypoints": "trajectory",
        "3_4_1_vehicle_ped_conflict": "has_conflict",
        "3_4_2_nearest_conflict_participant": "conflict_partner_type",
        "3_4_3_primary_risk_subject": "risk_reason",
        "3_4_4_risk_pattern": "interaction_pattern",
        "4_1_1_overall_state": "overall_state",
        "4_1_2_side_motion_status": "motion_label",
        "4_1_3_scene_summary": "summary_type",
        "4_1_4_flow_imbalance": "dominant_side",
        "4_2_1_speeding_risk": "has_speeding_risk",
        "4_2_2_notable_abnormal": "notable_abnormal",
        "4_3_1_intersection_action": "action_state",
        "4_3_2_side_action": "action_state",
        "4_3_3_lane_action": "action_state",
        "4_3_4_object_action": "action_state",
    }
    TEMPLATE_GLOBAL_BUCKET_CAPS = {}
    TEMPLATE_GLOBAL_GROUP_RATIOS = {}
    TEMPLATE_GLOBAL_YES_NO_RATIOS = {
        "1_1_2_side_exists": (7, 3),
        "2_2_6_crosswalk_blocking": (7, 3),
        "3_3_1_safe_following": (1, 1),
        "3_4_1_vehicle_ped_conflict": (7, 3),
        "4_2_1_speeding_risk": (7, 3),
    }
    TEMPLATE_GLOBAL_LABEL_RATIOS = {
        "1_1_1_fine_type": {
            "bicycle": 1,
            "bus": 1,
            "car": 1,
            "golf cart": 1,
            "motorcycle": 1,
            "pedestrian": 1,
            "trailer": 1,
            "truck": 1,
            "van": 1,
        },
        "1_1_3_side_count": {
            "bicycle": 1,
            "bus": 1,
            "car": 1,
            "construction vehicle": 1,
            "golf cart": 1,
            "motorcycle": 1,
            "pedestrian": 1,
            "truck": 1,
            "trailer": 1,
            "van": 1,
        },
        "1_2_1_size_bucket": {
            "small": 1,
            "medium": 1,
            "large": 1,
            "extra-large": 1,
        },
        "2_1_1_stopline_distance": {
            "car": 1,
            "truck": 1,
            "van": 1,
        },
        "2_1_2_ped_to_far_edge": {
            "north": 1,
            "south": 1,
            "east": 1,
            "west": 1,
        },
        "2_1_4_nearest_vehicle_to_ped": {
            "north": 1,
            "south": 1,
            "east": 1,
            "west": 1,
        },
        "2_2_1_lane_function": {
            "through lane": 1,
            "left-turn lane": 1,
            "right-turn lane": 1,
        },
        "2_2_2_ped_zone": {
            "within the crosswalk": 3,
            "waiting zone": 2,
            "entry zone": 1,
        },
        "2_2_3_left_turn_queue_count": {
            "left-turn lane": 1,
            "through lane": 1,
            "right-turn lane": 1,
        },
        "2_2_4_stopline_back_5m_count": {
            "north": 1,
            "south": 1,
            "east": 1,
            "west": 1,
        },
        "2_2_5_longest_queue_lane": {
            "left-turn lane": 1,
            "through lane": 1,
            "right-turn lane": 1,
        },
        "3_1_1_current_motion_state": {
            "moving": 10,
            "stopped": 10,
            "braking": 10,
            "creeping": 10,
            "starting": 1,
        },
        "3_1_2_vehicle_maneuver": {
            "left turn": 1,
            "straight": 1,
            "right turn": 1,
            "lane change": 1,
            "stop-and-wait": 1,
        },
        "3_3_2_likely_long_queue_lane": {
            "left-turn lane": 1,
            "through lane": 1,
            "right-turn lane": 1,
        },
        "3_2_2_future_region": {
            "before stop line": 1,
            "intersection center": 1,
            "left-turn exit": 1,
            "through exit": 1,
        },
        "3_4_3_primary_risk_subject": {
            "proximity": 1,
            "path_crossing": 1,
            "overspeed": 1,
            "vru_conflict": 1,
            "lane_change_conflict": 1,
        },
        "4_1_1_overall_state": {
            "free-flowing": 1,
            "light traffic": 1,
            "slightly congested": 1,
            "moderately congested": 1,
            "heavily congested": 1,
        },
        "4_1_2_side_motion_status": {
            "mostly moving": 1,
            "mostly stopped": 1,
            "mixed movement": 1,
        },
        "4_2_2_notable_abnormal": {
            "stopline_overrun": 1,
            "abnormal_proximity": 1,
            "speeding": 1,
        },
        "4_3_1_intersection_action": {
            "QUEUE_MANAGEMENT": 1,
            "CONFLICT_SUPPRESSION": 1,
            "FLOW_STABLE": 1,
            "FLOW_CALMING": 1,
        },
        "4_3_2_side_action": {
            "SIDE_CLEARANCE_PROTECTION": 1,
            "SIDE_SPEED_MODERATION": 1,
            "SIDE_QUEUE_STABILIZATION": 1,
            "SIDE_GENERAL_CAUTION": 1,
        },
        "4_3_3_lane_action": {
            "LANE_CLEARANCE_MAINTENANCE": 1,
            "LANE_PREPARE_TO_STOP": 1,
            "LANE_QUEUE_PRESERVATION": 1,
            "LANE_SPEED_REDUCTION": 1,
            "LANE_GENERAL_ORDER": 1,
        },
        "4_3_4_object_action": {
            "OBJECT_YIELD_NOW": 1,
            "OBJECT_PREPARE_TO_STOP": 1,
            "OBJECT_SLOW_DOWN": 1,
            "OBJECT_PROCEED_CAUTIOUSLY": 1,
        },
    }
    TEMPLATE_FINAL_RATIOS = {
        "1_1_1_fine_type": 0.5,
        "1_1_2_side_exists": 0.5,
        "1_1_3_side_count": 1.0,
        "1_1_4_relative_neighbor_type": 0.2,
        "1_2_1_size_bucket": 0.05,
        "1_2_2_visibility": 0.05,
        "1_3_1_weather": 0.10,
        "1_3_2_vehicle_signal_state": 0.02,
        "2_1_1_stopline_distance": 1.0,
        "2_1_2_ped_to_far_edge": 1.0,
        "2_1_3_participant_distance": 1.0,
        "2_1_4_nearest_vehicle_to_ped": 1.0,
        "2_2_1_lane_function": 0.1,
        "2_2_2_ped_zone": 1.0,
        "2_2_3_left_turn_queue_count": 0.8,
        "2_2_4_stopline_back_5m_count": 1.0,
        "2_2_5_longest_queue_lane": 1.0,
        "2_2_6_crosswalk_blocking": 1.0,
        "3_1_1_current_motion_state": 0.1,
        "3_1_2_vehicle_maneuver": 0.6,
        "3_3_1_safe_following": 0.5,
        "3_3_2_likely_long_queue_lane": 1.0,
        "3_2_2_future_region": 1.0,
        "3_2_3_waypoints": 0.05,
        "3_4_1_vehicle_ped_conflict": 1.0,
        "3_4_2_nearest_conflict_participant": 0.5,
        "3_4_3_primary_risk_subject": 1.0,
        "3_4_4_risk_pattern": 1.0,
        "4_1_1_overall_state": 0.5,
        "4_1_2_side_motion_status": 0.5,
        "4_1_3_scene_summary": 0.4,
        "4_1_4_flow_imbalance": 0.5,
        "4_2_1_speeding_risk": 0.5,
        "4_2_2_notable_abnormal": 1.0,
        "4_3_1_intersection_action": 1.0,
        "4_3_2_side_action": 0.5,
        "4_3_3_lane_action": 1.0,
        "4_3_4_object_action": 0.1,
    }
    TEMPLATE_FINAL_RATIO_EXEMPT_LABELS = {
        "1_1_1_fine_type": {"construction_vehicle"},
        "1_1_4_relative_neighbor_type": {
            "front",
            "behind",
            "front-left",
            "rear-left",
            "front-right",
            "rear-right",
        },
        "1_3_1_weather": {"cloudy", "rainy", ""},
        "3_1_1_current_motion_state": {"running", "standing", "starting"},
        "3_2_2_future_region": {"right-turn exit"},
        "3_4_2_nearest_conflict_participant": {
            "bicycle",
            "bus",
            "construction vehicle",
            "golf cart",
            "motorcycle",
            "pedestrian",
            "trailer",
            "van",
        },
        "4_1_4_flow_imbalance": {"north"},
        "4_2_2_notable_abnormal": {
            "crosswalk_blocking",
            "lingering_pedestrian",
            "wrong_way_two_wheeler",
        },
        "4_3_2_side_action": {"SIDE_CROSSING_AWARENESS"},
    }
    TEMPLATE_FINAL_RATIO_INCLUDED_LABELS = {
        "2_1_3_participant_distance": {
            "bicycle-car",
            "bus-car",
            "car-car",
            "car-golf cart",
            "car-pedestrian",
            "car-truck",
            "golf cart-truck",
        },
        "1_3_2_vehicle_signal_state": {"red light", "green light", "yellow light"},
    }
    TEMPLATE_FINAL_RATIO_BY_LABEL = {
        "1_3_1_weather": {"sunny": 0.10},
        "1_3_2_vehicle_signal_state": {
            "red light": 0.04,
            "green light": 0.04,
            "yellow light": 0.30,
        },
        "2_1_1_stopline_distance": {"golf cart": 0.25},
        "2_1_3_participant_distance": {
            "bicycle-car": 0.5,
            "bus-car": 0.5,
            "car-car": 0.5,
            "car-golf cart": 0.5,
            "car-pedestrian": 0.5,
            "car-truck": 0.5,
            "golf cart-truck": 0.5,
        },
    }
    TEMPLATE_TEMPORAL_SUPPRESSION_THRESHOLD_SEC = 3.0
    OBJECT_TYPE_MAPPING = {"traffic_cone": "golf cart", "barrier": "van"}
    VEHICLE_TYPES = {"car", "truck", "bus", "motorcycle", "bicycle", "golf cart", "van", "construction vehicle", "trailer"}
    VRU_TYPES = {"pedestrian", "bicycle", "motorcycle", "golf cart"}
    COARSE_VEHICLE_TYPES = VEHICLE_TYPES - VRU_TYPES
    SOURCE_SCENE_FPS = 10.0
    DEFAULT_KEYFRAME_FPS = 2.0
    MOVING_THRESHOLD = 0.5
    PEDESTRIAN_WALKING_THRESHOLD = 0.8
    PEDESTRIAN_RUNNING_THRESHOLD = 1.8
    VEHICLE_OVERSPEED_THRESHOLD = 20.0
    SIDE_SPEEDING_MARGIN = 2.0
    PRED_HORIZON = 3.0
    PRED_STEP = 0.1
    NEAR_MISS_DIST = 1.5
    ABNORMAL_PROXIMITY_THRESHOLD = 2.0
    IRREGULAR_CROSSING_THRESHOLD = 2.0
    CENTER_X = 17.7
    CENTER_Y = 13.2
    LANE_RANGES = {
        "north": [("left-turn lane", 3.2, 7.2), ("through lane", 7.2, 11.2), ("right-turn lane", 11.2, 15.2), ("fourth", 15.2, 19.2), ("fifth", 19.2, 23.4)],
        "south": [("left-turn lane", 11.0, 15.0), ("through lane", 15.0, 19.0), ("right-turn lane", 19.0, 23.2), ("fourth", 7.0, 11.0), ("fifth", 3.0, 7.0)],
        "east": [("right-turn lane", 7.4, 11.8), ("through lane", 11.8, 15.8), ("left-turn lane", 15.8, 19.8), ("fourth", 19.8, 23.8), ("fifth", 23.8, 27.8)],
        "west": [("right-turn lane", 24.0, 28.0), ("through lane", 20.0, 24.0), ("left-turn lane", 16.0, 20.0), ("fourth", 12.0, 16.0), ("fifth", 8.0, 12.0)],
    }
    SIDE_ORDER = ["north", "south", "east", "west", "center"]
    LANE_ORDER = ["left-turn lane", "through lane", "right-turn lane", "fourth", "fifth"]
    RELATIVE_DIRECTION_QUERY_PHRASES = {
        "front": "in front of",
        "behind": "behind",
        "left": "to the left of",
        "right": "to the right of",
        "front-left": "to the front-left of",
        "rear-left": "to the rear-left of",
        "front-right": "to the front-right of",
        "rear-right": "to the rear-right of",
    }

    FINE_TYPES = (
        "car",
        "truck",
        "bus",
        "trailer",
        "motorcycle",
        "pedestrian",
        "van",
        "golf cart",
        "bicycle",
        "construction vehicle",
    )
    COARSE_GROUPS = ("Vehicle", "VRU")
    CHAPTER_ORDER = (
        "1_base_perception",
        "2_spatial_reasoning",
        "3_temporal_reasoning",
        "4_scene_understanding",
    )
    COLOR_LABELS = ("red", "white", "black", "silver", "blue", "yellow", "green", "other", "multicolor", "hard to judge")
    SIZE_LABELS = ("small", "medium", "large", "extra-large")
    LIGHT_LABELS = ("red", "yellow", "green", "red_arrow", "yellow_arrow", "green_arrow")
    PED_LIGHT_LABELS = ("allow", "do_not_allow", "flashing_transition", "not_visible")
    TIME_OF_DAY_LABELS = ("daytime", "nighttime")
    SUN_GLARE_SIDE_LABELS = ("east", "north", "west", "south")
    VEHICLE_SIGNAL_LIGHT_STATES = ("red light", "yellow light", "green light")
    VEHICLE_SIGNAL_ARROW_STATES = ("red arrow", "yellow arrow", "green arrow")
    OVERALL_STATE_LABELS = (
        "free-flowing",
        "light traffic",
        "slightly congested",
        "moderately congested",
        "heavily congested",
    )
    RISK_REASON_LABELS = (
        "proximity",
        "path_crossing",
        "overspeed",
        "lane_change_conflict",
        "vru_conflict",
    )
    NOTABLE_ABNORMAL_LABELS = (
        "speeding",
        "abnormal_proximity",
        "crosswalk_blocking",
        "lingering_pedestrian",
        "stopline_overrun",
        "wrong_way_two_wheeler",
        "queue_spillback",
    )
    DISABLED_TEMPLATE_REASONS = {}
    PLACEHOLDER_TEMPLATE_IDS = {
        "1_2_2_visibility",
        "1_3_1_weather",
        "1_3_2_vehicle_signal_state",
    }
    CAM_NAME_TO_VIEW = {
        "CAM_SOUTH": "north_image",
        "CAM_NORTH": "south_image",
        "CAM_WEST": "east_image",
        "CAM_EAST": "west_image",
    }
    SIDE_LABELS = {"north": "north", "south": "south", "east": "east", "west": "west"}
    SIDE_IMAGE_NAMES = {
        "north": "north",
        "south": "south",
        "east": "east",
        "west": "west",
    }
    WAITING_ZONE_LABELS = ("north-east corner", "south-east corner", "north-west corner", "south-west corner")
    RIGHT_TURN_LANE = "right-turn lane"
    LEFT_TURN_LANE = "left-turn lane"
    STRAIGHT_LANE = "through lane"
    SIZE_QUANTILES = (0.25, 0.5, 0.75)

    NORTH_CROSSWALK = (2.2, 5.2, 2.6, 25.0)
    EAST_CROSSWALK = (7.4, 28.4, 26.8, 30.1)
    SOUTH_CROSSWALK = (30.4, 33.7, 1.7, 25.1)
    WEST_CROSSWALK = (6.5, 29.6, -2.9, 0.3)

    CROSSWALK_CONFIGS = {
        "north": {
            "axis": "y",
            "crosswalk": NORTH_CROSSWALK,
            "entries": (
                ("north_west_entry", "low", (2.2, 5.2, 0.3, 2.6)),
                ("north_east_entry", "high", (2.2, 5.2, 25.0, 26.8)),
            ),
        },
        "east": {
            "axis": "x",
            "crosswalk": EAST_CROSSWALK,
            "entries": (
                ("east_north_entry", "low", (5.2, 7.4, 26.8, 30.1)),
                ("east_south_entry", "high", (28.4, 30.4, 26.8, 30.1)),
            ),
        },
        "south": {
            "axis": "y",
            "crosswalk": SOUTH_CROSSWALK,
            "entries": (
                ("south_west_entry", "low", (30.4, 33.7, 0.3, 1.7)),
                ("south_east_entry", "high", (30.4, 33.7, 25.1, 26.8)),
            ),
        },
        "west": {
            "axis": "x",
            "crosswalk": WEST_CROSSWALK,
            "entries": (
                ("west_north_entry", "low", (5.2, 6.5, -2.9, 0.3)),
                ("west_south_entry", "high", (29.6, 30.4, -2.9, 0.3)),
            ),
        },
    }
    ENTRY_REGION_LABELS = {
        "north_west_entry": "entry zone",
        "north_east_entry": "entry zone",
        "east_north_entry": "entry zone",
        "east_south_entry": "entry zone",
        "south_west_entry": "entry zone",
        "south_east_entry": "entry zone",
        "west_north_entry": "entry zone",
        "west_south_entry": "entry zone",
    }
    EXIT_REGION_NAMES = {
        "north": (
            ("north_west_exit", (2.2, 5.2, 25.0, 26.8)),
            ("north_east_exit", (2.2, 5.2, 0.3, 2.6)),
        ),
        "east": (
            ("east_north_exit", (28.4, 30.4, 26.8, 30.1)),
            ("east_south_exit", (5.2, 7.4, 26.8, 30.1)),
        ),
        "south": (
            ("south_west_exit", (30.4, 33.7, 25.1, 26.8)),
            ("south_east_exit", (30.4, 33.7, 0.3, 1.7)),
        ),
        "west": (
            ("west_north_exit", (29.6, 30.4, -2.9, 0.3)),
            ("west_south_exit", (5.2, 6.5, -2.9, 0.3)),
        ),
    }
    WAITING_ZONES = {
        "north-west corner": (None, 7.5, None, 2.8),
        "north-east corner": (None, 7.0, 24.2, None),
        "south-west corner": (28.9, None, None, 2.7),
        "south-east corner": (29.1, None, 24.0, None),
    }

    def __init__(self, pkl_path: str, subtemplate_patch_style: str = "simple"):
        self.pkl_path = pkl_path
        self.subtemplate_patch_style = subtemplate_patch_style
        pathlib.WindowsPath = pathlib.PosixPath
        with open(pkl_path, "rb") as f:
            self.infos = pickle.load(f)
        self.scene_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.status_cache: Dict[Tuple[str, str], str] = {}
        self.motion_cache: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
        self.future_track_cache: Dict[Tuple[str, str], List[Optional[Dict]]] = {}
        self.active_keyframe_fps = self.DEFAULT_KEYFRAME_FPS
        for idx, frame in enumerate(self.infos):
            self.scene_to_indices[self._scene_id(frame)].append(idx)
        self.frame_number_by_token = self._build_frame_number_map()
        self.frame_index_by_token = {self._token(frame): idx for idx, frame in enumerate(self.infos)}
        self.snapshots_by_token = {
            self._token(frame): self._extract_snapshots(frame)
            for frame in self.infos
        }
        self.scene_ref_id_map = self._build_scene_ref_id_map()
        self._build_track_cache()
        self.token_to_frame = {self._token(frame): frame for frame in self.infos}
        self.token_to_timestamp_seconds = self._build_token_timestamp_map()
        self.scene_start_time_seconds = self._build_scene_start_time_map()
        self.token_to_scene_index = self._build_scene_index_map()
        self.history_cache = self._build_history_cache()
        self.size_bins = self._compute_size_bins()
        self.template_registry = self._build_template_registry()
        self.sampling_policy_registry = self._build_sampling_policy_registry()
        self._active_frame_cache: Optional[Dict] = None

    def _build_frame_number_map(self) -> Dict[str, int]:
        frame_numbers: Dict[str, int] = {}
        for indices in self.scene_to_indices.values():
            for local_idx, frame_idx in enumerate(indices, 1):
                token = self._token(self.infos[frame_idx])
                frame_numbers[token] = local_idx
        return frame_numbers

    def _build_scene_index_map(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for indices in self.scene_to_indices.values():
            for order_idx, frame_idx in enumerate(indices):
                mapping[self._token(self.infos[frame_idx])] = order_idx
        return mapping

    def _build_scene_ref_id_map(self) -> Dict[Tuple[str, str], str]:
        scene_ref_id_map: Dict[Tuple[str, str], str] = {}
        for scene_id, indices in self.scene_to_indices.items():
            first_seen: Dict[str, Tuple[int, float, str]] = {}
            for local_idx, frame_idx in enumerate(indices):
                frame = self.infos[frame_idx]
                token = self._token(frame)
                snapshots = self.snapshots_by_token.get(token, {})
                for obj_id, obj in snapshots.items():
                    if obj_id in first_seen:
                        continue
                    first_seen[obj_id] = (local_idx, self._center_dist(obj), str(obj_id))
            ordered_ids = [
                obj_id
                for obj_id, _ in sorted(first_seen.items(), key=lambda item: item[1])
            ]
            for idx, obj_id in enumerate(ordered_ids, 1):
                scene_ref_id_map[(scene_id, obj_id)] = f"o{idx}"
        return scene_ref_id_map

    def _parse_token_timestamp_seconds(self, frame_token: str) -> Optional[float]:
        if not frame_token:
            return None
        match = re.fullmatch(r"(\d+)-(\d+)", frame_token)
        if match:
            sec = int(match.group(1))
            frac_digits = match.group(2)
            return float(sec) + float(f"0.{frac_digits}")
        try:
            return float(frame_token)
        except ValueError:
            return None

    def _build_token_timestamp_map(self) -> Dict[str, Optional[float]]:
        return {
            token: self._parse_token_timestamp_seconds(token)
            for token in self.token_to_frame
        }

    def _build_scene_start_time_map(self) -> Dict[str, Optional[float]]:
        scene_start_times: Dict[str, Optional[float]] = {}
        for scene_id, indices in self.scene_to_indices.items():
            start_time: Optional[float] = None
            for frame_idx in indices:
                token = self._token(self.infos[frame_idx])
                parsed = self.token_to_timestamp_seconds.get(token)
                if parsed is not None:
                    start_time = parsed
                    break
            scene_start_times[scene_id] = start_time
        return scene_start_times

    def _relative_scene_time_seconds(self, frame_token: str) -> float:
        frame = self.token_to_frame.get(frame_token)
        if frame is not None:
            scene_id = self._scene_id(frame)
            frame_time = self.token_to_timestamp_seconds.get(frame_token)
            scene_start = self.scene_start_time_seconds.get(scene_id)
            if frame_time is not None and scene_start is not None:
                return max(0.0, frame_time - scene_start)
        scene_index = self.token_to_scene_index.get(frame_token, 0)
        return float(scene_index) / float(self.SOURCE_SCENE_FPS)

    def _format_relative_scene_time(self, frame_token: str) -> str:
        seconds = self._relative_scene_time_seconds(frame_token)
        rounded = round(seconds, 1)
        if abs(rounded) < 1e-9:
            return "0"
        return f"{rounded:.1f}"

    def _build_history_cache(self) -> Dict[Tuple[str, str], List[Dict]]:
        cache: Dict[Tuple[str, str], List[Dict]] = {}
        for indices in self.scene_to_indices.values():
            snapshots_per_frame = [self.snapshots_by_token[self._token(self.infos[i])] for i in indices]
            for local_idx, frame_idx in enumerate(indices):
                token = self._token(self.infos[frame_idx])
                current = snapshots_per_frame[local_idx]
                for obj_id in current:
                    history = [snapshots_per_frame[j][obj_id] for j in range(local_idx) if obj_id in snapshots_per_frame[j]]
                    cache[(token, obj_id)] = history
        return cache

    def _compute_size_bins(self) -> Tuple[float, float, float]:
        volumes: List[float] = []
        for snapshots in self.snapshots_by_token.values():
            for obj in snapshots.values():
                volumes.append(max(obj.get("dx", 0.1), 0.1) * max(obj.get("dy", 0.1), 0.1) * max(obj.get("dz", 0.1), 0.1))
        if not volumes:
            return (1.0, 2.0, 4.0)
        return tuple(float(v) for v in np.quantile(np.asarray(volumes, dtype=float), self.SIZE_QUANTILES))

    def _build_template_registry(self) -> Dict[str, V5TemplateSpec]:
        ref = self.REF_SCHEMA_TOKEN
        specs = [
            V5TemplateSpec("1_1_1_fine_type", "1_base_perception", "1_1_object_identify", "Fine-Grained Type", f"What is the type of `{ref}`?", "{fine type}", ("3d_boxes", "tracking"), True),
            V5TemplateSpec("1_1_2_side_exists", "1_base_perception", "1_1_object_identify", "Approach-Level Existence", "Does the {side} approach contain any vulnerable road user?", "{yes / no}", ("3d_boxes",), True),
            V5TemplateSpec("1_1_3_side_count", "1_base_perception", "1_1_object_identify", "Approach-Level Count", "How many {type}s are currently on the {side} approach?", "{integer}", ("3d_boxes",), True),
            V5TemplateSpec("1_1_4_relative_neighbor_type", "1_base_perception", "1_1_object_identify", "Relative Neighbor Type", f"What is located {{relative_phrase}} `{ref}`?", "It is a {car} at <X,Y,image_name_1,X1_1,Y1_1,...,image_name_n,X1_n,Y1_n>.", ("3d_boxes", "tracking"), True),
            V5TemplateSpec("1_2_1_size_bucket", "1_base_perception", "1_2_appearance_attributes", "Size Bucket", f"Which size best matches `{ref}`?", "{small / medium / large / extra-large}", ("3d_boxes",), True),
            V5TemplateSpec("1_2_2_visibility", "1_base_perception", "1_2_appearance_attributes", "Visibility", f"How visible is `{ref}`?", "{fully visible / partially occluded / heavily occluded}", ("images", "visibility"), True, None),
            V5TemplateSpec("1_3_1_weather", "1_base_perception", "1_3_scene_conditions_and_signals", "Environment", "Please describe the current environmental conditions in the scene.", "{weather / time of day / sun glare description}", ("scene_metadata",), True),
            V5TemplateSpec("1_3_2_vehicle_signal_state", "1_base_perception", "1_3_scene_conditions_and_signals", "Vehicle Signal State", "Please describe the signal state for {movement} traffic on the {side} approach.", "{red light / yellow light / green light / red arrow / yellow arrow / green arrow}", ("signals", "images"), True),
            V5TemplateSpec("2_1_1_stopline_distance", "2_spatial_reasoning", "2_1_geometric_localization", "Distance to Stop Line", f"How far is `{ref}` from its relevant stop line?", "{value} m", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_1_2_ped_to_far_edge", "2_spatial_reasoning", "2_1_geometric_localization", "Pedestrian to Crossing Exit Area", f"How far is `{ref}` from the crossing exit area?", "{value} m", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("2_1_3_participant_distance", "2_spatial_reasoning", "2_1_geometric_localization", "Participant Pair Distance", f"What is the distance between `{ref}` and `{ref}`?", "{value} m", ("3d_boxes",), True),
            V5TemplateSpec("2_1_4_nearest_vehicle_to_ped", "2_spatial_reasoning", "2_1_geometric_localization", "Nearest Vehicle to Pedestrian", f"Please identify the vehicle nearest to `{ref}`.", "It is a {car} on the {east} approach, {3.1 m} away.", ("3d_boxes",), True),
            V5TemplateSpec("2_2_1_lane_function", "2_spatial_reasoning", "2_2_topological_relations", "Lane Function", f"Which lane does `{ref}` currently occupy?", "{left-turn lane / through lane / right-turn lane}", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_2_2_ped_zone", "2_spatial_reasoning", "2_2_topological_relations", "Pedestrian Zone", f"Which area of the crosswalk is `{ref}` currently in?", "{within the crosswalk / entry zone / exit zone / waiting zone}", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("2_2_3_left_turn_queue_count", "2_spatial_reasoning", "2_2_topological_relations", "Lane Queue Count", "How many queued vehicles are currently in the {lane} lane on the {side} approach?", "{integer}", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_2_4_stopline_back_5m_count", "2_spatial_reasoning", "2_2_topological_relations", "Vehicles Within 5m Behind Stop Line", "How many vehicles are within 5 meters behind the stop line on the {side} approach?", "{integer}", ("3d_boxes",), True),
            V5TemplateSpec("2_2_5_longest_queue_lane", "2_spatial_reasoning", "2_2_topological_relations", "Longest Queue Lane", "Which lane currently has the longest queue right now?", "{left-turn lane / through lane / right-turn lane}", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_2_6_crosswalk_blocking", "2_spatial_reasoning", "2_2_topological_relations", "Crosswalk Blocking", "Is any vehicle blocking the crosswalk?", "{yes / no}", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("3_1_1_current_motion_state", "3_temporal_reasoning", "3_1_motion_state_recognition", "Current Motion State", f"Please describe the current motion state of `{ref}`.", "{standing / walking / running / stopped / starting / moving / creeping / braking}", ("history_tracks",), True),
            V5TemplateSpec("3_1_2_vehicle_maneuver", "3_temporal_reasoning", "3_1_motion_state_recognition", "Vehicle Maneuver", f"What maneuver is `{ref}` most likely executing?", "{left turn / straight / right turn / lane change / stop-and-wait}", ("history_tracks", "future_tracks"), True),
            V5TemplateSpec("3_3_1_safe_following", "3_temporal_reasoning", "3_3_following_and_queue_dynamics", "Safe Following", f"Are straight-moving `{ref}` and `{ref}` maintaining a safe following gap?", "{yes / no}", ("history_tracks", "lane_side"), True),
            V5TemplateSpec("3_3_2_likely_long_queue_lane", "3_temporal_reasoning", "3_3_following_and_queue_dynamics", "Likely Long-Queue Lane", "Which lane is most likely to form a long queue soon?", "{structured natural language}", ("history_tracks", "lane_side"), True),
            V5TemplateSpec("3_2_2_future_region", "3_temporal_reasoning", "3_2_trajectory_trend_judgment", "Future Region", f"Which region is `{ref}` most likely to enter within the next 3 seconds?", "{intersection center / left-turn exit / through exit / right-turn exit / before stop line}", ("future_tracks", "lane_side"), True),
            V5TemplateSpec("3_2_3_waypoints", "3_temporal_reasoning", "3_2_trajectory_trend_judgment", "Trajectory Prediction", f"Please predict the short-term future trajectory for `{ref}`.", "Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)", ("future_tracks",), True),
            V5TemplateSpec("3_4_1_vehicle_ped_conflict", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Vehicle-VRU Conflict", f"Is there a potential conflict between `{ref}` and `{ref}`?", "{yes / no}", ("future_tracks",), True),
            V5TemplateSpec("3_4_2_nearest_conflict_participant", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Nearest Conflict Participant", f"Which participant is most likely to conflict with `{ref}`?", "The most probable participant is the {car} at <X,Y,image_name_1,X1_1,Y1_1,...,image_name_n,X1_n,Y1_n>.", ("future_tracks",), True),
            V5TemplateSpec("3_4_3_primary_risk_subject", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Primary Risk Subject", "Please identify the key participant associated with the major potential safety hazards.", "The primary risk subject is the {car} at <X,Y,image_name_1,X1_1,Y1_1,...,image_name_n,X1_n,Y1_n> because of {path crossing}.", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("3_4_4_risk_pattern", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Risk Interaction Pattern", "What is the dominant conflict interaction pattern in the current scene?", "The dominant conflict pattern is {rear-end} on the {east} approach.", ("future_tracks",), True),
            V5TemplateSpec("4_1_1_overall_state", "4_scene_understanding", "4_1_overall_intersection_state", "Overall Operating State", "What is the overall traffic condition of the intersection?", "{free-flowing / light traffic / slightly congested / moderately congested / heavily congested}", ("history_tracks", "3d_boxes"), True),
            V5TemplateSpec("4_1_2_side_motion_status", "4_scene_understanding", "4_1_overall_intersection_state", "Approach Motion Status", "Describe the motion status of traffic participants on the {side} approach of the intersection.", "{structured natural language}", ("3d_boxes", "history_tracks"), True),
            V5TemplateSpec("4_1_3_scene_summary", "4_scene_understanding", "4_1_overall_intersection_state", "Scene Summary", "Provide a brief summary of the current intersection scene.", "{natural language summary}", ("3d_boxes", "history_tracks"), True),
            V5TemplateSpec("4_1_4_flow_imbalance", "4_scene_understanding", "4_1_overall_intersection_state", "Heaviest Traffic Approach", "Which approach to the intersection has the heaviest traffic?", "The {east} approach is the busiest, with {6} traffic participants.", ("3d_boxes",), True),
            V5TemplateSpec("4_2_1_speeding_risk", "4_scene_understanding", "4_2_abnormal_events", "Speeding Risk", "Is there still a risk of speeding at the intersection?", "{Yes/No + structured natural language}", ("future_tracks",), True),
            V5TemplateSpec("4_2_2_notable_abnormal", "4_scene_understanding", "4_2_abnormal_events", "Most Notable Abnormal Event", "What is the most notable current abnormal event in the scene?", "{fixed label + short explanation}", ("future_tracks", "history_tracks", "crosswalk_geometry"), True),
            V5TemplateSpec("4_3_1_intersection_action", "4_scene_understanding", "4_3_planning_guidance", "Intersection-Level Guidance", "How should traffic proceed through the intersection right now?", "{concise narrative guidance}", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_2_side_action", "4_scene_understanding", "4_3_planning_guidance", "Approach-Level Guidance", "What precaution should traffic take on the {side} approach of the intersection right now?", "{concise narrative guidance}", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_3_lane_action", "4_scene_understanding", "4_3_planning_guidance", "Lane-Level Guidance", "How should traffic behave in the {lane} on the {side} approach right now?", "{concise narrative guidance}", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_4_object_action", "4_scene_understanding", "4_3_planning_guidance", "Object-Level Guidance", f"What is the safest action guidance for `{ref}`?", "{concise narrative guidance}", ("future_tracks", "history_tracks"), True),
        ]
        return {spec.template_id: spec for spec in specs}

    def _build_sampling_policy_registry(self) -> Dict[str, Dict]:
        return {
            "1_1_1_fine_type": {"coverage_priority": ("object_type", "direction")},
            "1_1_2_side_exists": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "1_1_3_side_count": {"coverage_priority": ("object_type", "direction")},
            "1_1_4_relative_neighbor_type": {"coverage_priority": ("rel_dir",)},
            "1_2_1_size_bucket": {"coverage_priority": ("size_bucket",)},
            "1_2_2_visibility": {"coverage_priority": ("visibility",)},
            "1_3_1_weather": {"coverage_priority": ("weather", "time_of_day", "sun_glare"), "placeholder": True},
            "1_3_2_vehicle_signal_state": {"coverage_priority": ("signal_state", "movement", "direction"), "placeholder": True},
            "2_1_1_stopline_distance": {"coverage_priority": ("object_type",)},
            "2_1_2_ped_to_far_edge": {"coverage_priority": ("crosswalk",)},
            "2_1_3_participant_distance": {"coverage_priority": ("pair_type",)},
            "2_1_4_nearest_vehicle_to_ped": {"coverage_priority": ("direction",)},
            "2_2_1_lane_function": {"coverage_priority": ("lane_function", "object_type")},
            "2_2_2_ped_zone": {"coverage_priority": ("zone",)},
            "2_2_3_left_turn_queue_count": {"coverage_priority": ("direction", "lane_function")},
            "2_2_4_stopline_back_5m_count": {"coverage_priority": ("direction",)},
            "2_2_5_longest_queue_lane": {"coverage_priority": ("lane_function", "direction")},
            "2_2_6_crosswalk_blocking": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "3_1_1_current_motion_state": {"coverage_priority": ("motion_state", "object_id", "direction")},
            "3_1_2_vehicle_maneuver": {"coverage_priority": ("maneuver", "object_id", "direction")},
            "3_2_2_future_region": {"coverage_priority": ("future_region", "object_id", "direction")},
            "3_2_3_waypoints": {"coverage_priority": ("object_id", "direction")},
            "3_3_1_safe_following": {"coverage_priority": ("pair_type",), "yes_no_ratio": (5, 5)},
            "3_3_2_likely_long_queue_lane": {"coverage_priority": ("lane_function", "direction")},
            "3_4_1_vehicle_ped_conflict": {"coverage_priority": ("has_conflict", "pair_type"), "yes_no_ratio": (7, 3)},
            "3_4_2_nearest_conflict_participant": {"coverage_priority": ("object_id", "direction")},
            "3_4_3_primary_risk_subject": {"coverage_priority": ("risk_reason", "object_id")},
            "3_4_4_risk_pattern": {"coverage_priority": ("interaction_pattern",)},
            "4_1_1_overall_state": {"coverage_priority": ("overall_state",), "single_instance_global": True},
            "4_1_2_side_motion_status": {"coverage_priority": ("direction",)},
            "4_1_3_scene_summary": {"single_instance_global": True, "fixed_cap": 1},
            "4_1_4_flow_imbalance": {"coverage_priority": ("dominant_side",)},
            "4_2_1_speeding_risk": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "4_2_2_notable_abnormal": {"coverage_priority": ("notable_abnormal",)},
            "4_3_1_intersection_action": {"coverage_priority": ("action_state",), "single_instance_global": True},
            "4_3_2_side_action": {"coverage_priority": ("action_state", "direction")},
            "4_3_3_lane_action": {"coverage_priority": ("action_state", "lane_function")},
            "4_3_4_object_action": {"coverage_priority": ("action_state", "object_id")},
        }

    def _scene_id(self, frame: Dict) -> str:
        return str(frame.get("scene_id") or frame.get("scene_token") or frame.get("log_id") or "default")

    def _token(self, frame: Dict) -> str:
        return str(frame.get("token") or frame.get("sample_token") or frame.get("frame_token") or "unknown")

    def _stringify_path(self, value) -> Optional[str]:
        if value is None:
            return None
        path_str = str(value).replace("\\", "/")
        parts = [part for part in path_str.split("/") if part]
        for idx, part in enumerate(parts):
            if part.startswith("rosbag"):
                return "data/" + "/".join(parts[idx:])
        return path_str

    def _frame_media_paths(self, frame: Dict) -> Dict[str, Optional[str]]:
        cams = frame.get("cams") or {}
        return {
            "north_image_path": self._stringify_path((cams.get("CAM_SOUTH") or {}).get("image_paths")),
            "south_image_path": self._stringify_path((cams.get("CAM_NORTH") or {}).get("image_paths")),
            "east_image_path": self._stringify_path((cams.get("CAM_WEST") or {}).get("image_paths")),
            "west_image_path": self._stringify_path((cams.get("CAM_EAST") or {}).get("image_paths")),
            "point_cloud_path": self._stringify_path(frame.get("lidar_path")),
        }

    def _image_only_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.\n\n"
            "You are given four synchronized camera images from the same timestamp:\n"
            "- north_image\n"
            "- south_image\n"
            "- east_image\n"
            "- west_image\n\n"
            "Available inputs and grounding rules:\n"
            "- north_image shows the north approach of the intersection.\n"
            "- south_image shows the south approach of the intersection.\n"
            "- east_image shows the east approach of the intersection.\n"
            "- west_image shows the west approach of the intersection.\n"
            "- Use only the four images as evidence.\n"
            "- Use cross-view visual evidence conservatively and do not hallucinate unsupported objects, states, or interactions.\n"
            "- Do not infer hidden signal states or invisible objects from behavior alone.\n\n"
            "Object reference rules:\n"
            f"- Object references use the format {self.REF_SCHEMA_TOKEN}.\n"
            "- ID is scene-stable within the same scene.\n"
            "- f is the relative time offset in seconds from the first frame of the same scene.\n"
            "- X,Y are global point-cloud planar coordinates in meters.\n"
            "- In image-only mode, do not rely on X,Y as evidence; use only the four images for reasoning.\n"
            "- Each repeated triplet image_name_k,X1_k,Y1_k describes one image in which the same object is visible.\n"
            "- image_name_k is written directly as north_image, south_image, east_image, or west_image.\n"
            "- X1_k,Y1_k are the 2D box center coordinates in that image.\n"
            "- The reference may contain one or multiple image triplets, depending on how many views contain the object.\n\n"
            "Geometry and rule grounding:\n"
            "- You may use the referenced approach names, lane names, and region names given in the question metadata.\n"
            "- In image-only mode, do not ground the answer in global X,Y coordinates or point-cloud geometry.\n"
            "- Distances between participants are defined on the ground plane as the minimum distance between their 3D box footprints, ignoring z.\n"
            "- Potential conflict judgments use the task-defined short-horizon rule and must stay consistent with the provided task instructions.\n"
            "- Do not invent geometry that is not supported by the images and the provided object references.\n\n"
            "Spatial wording rules:\n"
            "- Use approach names north, south, east, west, and center area consistently.\n"
            "- Use lane names left-turn lane, through lane, and right-turn lane.\n"
            "- Use crosswalk, entry zone, exit zone, and waiting zone consistently.\n"
            "- Relative directions such as front, rear, left, and right are defined by the referenced object's own heading.\n\n"
            "Answering rules:\n"
            "- Answer only the queried task.\n"
            "- Keep the answer concise and grounded in the visible scene.\n"
            "- Do not add extra explanation unless the task itself asks for evidence or summary.\n"
            "- Keep numbers and units when the task is about counts, distances, speeds, or trajectories.\n"
            "- If the task is yes/no, keep negative answers brief.\n"
        )

    def _pointcloud_plus_image_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.\n\n"
            "You are given synchronized multimodal evidence from the same timestamp:\n"
            "- four camera images: north_image, south_image, east_image, west_image\n"
            "- global intersection point cloud: point_cloud\n\n"
            "Available inputs and grounding rules:\n"
            "- north_image shows the north approach of the intersection.\n"
            "- south_image shows the south approach of the intersection.\n"
            "- east_image shows the east approach of the intersection.\n"
            "- west_image shows the west approach of the intersection.\n"
            "- Use both the images and the point cloud conservatively as evidence.\n"
            "- For 3D geometry, spatial layout, global position, and distance inference, primarily rely on the point cloud, with the images as auxiliary evidence.\n"
            "- For visible appearance, visible signal state, local occlusion, and cross-view scene context, primarily rely on the images, with the point cloud as auxiliary evidence.\n"
            "- Do not hallucinate unsupported objects, states, or interactions.\n"
            "- Do not infer hidden signal states or invisible objects from behavior alone.\n\n"
            "Object reference rules:\n"
            f"- Object references use the format {self.REF_SCHEMA_TOKEN}.\n"
            "- ID is scene-stable within the same scene.\n"
            "- f is the relative time offset in seconds from the first frame of the same scene.\n"
            "- X,Y are global point-cloud planar coordinates in meters and come from 3D scene geometry.\n"
            "- Each repeated triplet image_name_k,X1_k,Y1_k describes one image in which the same object is visible.\n"
            "- image_name_k is written directly as north_image, south_image, east_image, or west_image.\n"
            "- X1_k,Y1_k are the 2D box center coordinates in that image.\n"
            "- The reference may contain one or multiple image triplets, depending on how many views contain the object.\n\n"
            "Point-cloud reference rules:\n"
            "- The point-cloud coordinate system is global and shared across frames within the same scene.\n"
            "- The coordinate unit is meters.\n"
            "- The positive X direction points toward the south approach.\n"
            "- The positive Y direction points toward the east approach.\n"
            "- Smaller X means farther north; larger X means farther south.\n"
            "- Smaller Y means farther west; larger Y means farther east.\n"
            "- The center reference point is (17.7, 13.2).\n\n"
            "Geometry and rule grounding:\n"
            "- Use the point cloud and the provided object references for geometry-aware reasoning.\n"
            "- Distances between participants are defined on the ground plane as the minimum distance between their 3D box footprints, ignoring z.\n"
            "- Potential conflict judgments use the task-defined short-horizon rule and must stay consistent with the provided task instructions.\n"
            "- Lane, approach, crosswalk, entry-zone, exit-zone, waiting-zone, and relative-direction reasoning must follow the task metadata and geometric rules.\n\n"
            "Spatial wording rules:\n"
            "- Use approach names north, south, east, west, and center area consistently.\n"
            "- Approach-region ranges are: north if x < 7.5; south if x > 28.6; west if y < 3.1; east if y > 23.8; center area if 7.5 <= x < 28.6 and 3.1 <= y < 23.8.\n"
            "- Use lane names left-turn lane, through lane, and right-turn lane.\n"
            "- Lane-region ranges are: north approach left-turn if 4.8 <= y < 8.4, through if 8.4 <= y < 12.0, right-turn if 12.0 <= y < 15.6; south approach left-turn if 11.0 <= y < 15.0, through if 15.0 <= y < 19.0, right-turn if 19.0 <= y < 23.2; east approach left-turn if 19.2 <= x < 23.1, through if 15.5 <= x < 19.2, right-turn if 11.8 <= x < 15.5; west approach left-turn if 11.6 <= x < 15.6, through if 15.6 <= x < 19.2, right-turn if 19.2 <= x < 23.0.\n"
            "- Use crosswalk, entry zone, exit zone, and waiting zone consistently.\n"
            "- Crosswalk-related ranges are: north crosswalk if 2.2 <= x < 5.2 and 2.6 <= y < 25.0, with entry/exit bands at 0.3 <= y < 2.6 and 25.0 <= y < 26.8; east crosswalk if 7.4 <= x < 28.4 and 26.8 <= y < 30.1, with entry/exit bands at 5.2 <= x < 7.4 and 28.4 <= x < 30.4; south crosswalk if 30.4 <= x < 33.7 and 1.7 <= y < 25.1, with entry/exit bands at 0.3 <= y < 1.7 and 25.1 <= y < 26.8; west crosswalk if 6.5 <= x < 29.6 and -2.9 <= y < 0.3, with entry/exit bands at 5.2 <= x < 6.5 and 29.6 <= x < 30.4.\n"
            "- Waiting-zone ranges are: north-west if x < 7.5 and y < 2.8, north-east if x < 7.0 and y >= 24.2, south-west if x >= 28.9 and y < 2.7, and south-east if x >= 29.1 and y >= 24.0, each excluding any overlap with crosswalk, entry, or exit regions.\n"
            "- Relative directions such as front, rear, left, and right are defined by the referenced object's own heading.\n\n"
            "- Relative-direction angular ranges in the referenced object's local heading frame are: front if -22.5° <= angle < 22.5°; front-left if 22.5° <= angle < 67.5°; left if 67.5° <= angle < 112.5°; rear-left if 112.5° <= angle < 157.5°; behind if angle >= 157.5° or angle < -157.5°; rear-right if -157.5° <= angle < -112.5°; right if -112.5° <= angle < -67.5°; front-right if -67.5° <= angle < -22.5°.\n\n"
            "Answering rules:\n"
            "- Answer only the queried task.\n"
            "- Keep the answer concise and grounded in the provided multimodal evidence.\n"
            "- Do not add extra explanation unless the task itself asks for evidence or summary.\n"
            "- Keep numbers and units when the task is about counts, distances, speeds, or trajectories.\n"
            "- If the task is yes/no, keep negative answers brief.\n"
        )

    def _user_prompt_wrapper(self, modality: str = "image_only") -> str:
        if modality == "pointcloud_plus_image":
            return (
                "Image-to-direction mapping:\n"
                "- image 1 = north view\n"
                "- image 2 = south view\n"
                "- image 3 = east view\n"
                "- image 4 = west view\n"
                "- global intersection point cloud: {point_cloud}\n\n"
                "Task metadata:\n"
                "- chapter: {chapter}\n"
                "- section: {section}\n"
                "- subtemplate: {subtemplate}\n\n"
                "Question:\n"
                "{question}\n\n"
                "Task instructions:\n"
                "{task_rule_block}"
            )
        return (
            "Image-to-direction mapping:\n"
            "- image 1 = north view\n"
            "- image 2 = south view\n"
            "- image 3 = east view\n"
            "- image 4 = west view\n\n"
            "Task metadata:\n"
            "- chapter: {chapter}\n"
            "- section: {section}\n"
            "- subtemplate: {subtemplate}\n\n"
            "Question:\n"
            "{question}\n\n"
            "Task instructions:\n"
            "{task_rule_block}"
        )

    def _strict_answer_schemas(self) -> Dict[str, str]:
        return {
            "3_2_3_waypoints": "Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)",
        }

    def _subtemplate_patches(self) -> Dict[str, str]:
        return {
            "1_1_1_fine_type": (
                "Answer briefly with the referenced object's fine-grained type plus its current approach/location phrase.\n"
                "Do not expand to motion or interaction details."
            ),
            "1_1_2_side_exists": (
                "For this task, treat vulnerable road users as pedestrians, bicycles, motorcycles, and golf carts only.\n"
                "If yes, mention the count concisely; if not, answer briefly without extra explanation."
            ),
            "1_1_3_side_count": (
                "Report only that count and avoid expanding into scene-level summary."
            ),
            "1_1_4_relative_neighbor_type": (
                "Interpret the relative direction in the referenced object's own heading frame, not in the global north-up frame.\n"
                "Answer with the target object's type plus its location-and-image reference only, without extra scene description."
            ),
            "1_2_1_size_bucket": (
                "Use the object's overall 3D size rather than motion, interaction, or scene context cues.\n"
                "Answer only with the size category."
            ),
            "1_2_2_visibility": (
                "Judge the visibility level by how fully the object is visible versus occluded.\n"
                "Answer only with the object type, its approach, and the visibility label."
            ),
            "1_3_1_weather": (
                "Use whole-scene visual evidence rather than local lighting effects.\n"
                "Answer only with the available environmental attributes for this task: weather, time of day, and any visible strong sun glare."
            ),
            "1_3_2_vehicle_signal_state": (
                "Use only directly visible signal evidence, and do not infer hidden lights from traffic behavior or traffic flow.\n"
                "Answer only with the signal-state label allowed for this task."
            ),
            "2_1_1_stopline_distance": (
                "Focus only on that stop-line distance, retain the numeric value with units, and do not expand into motion or queue interpretation."
            ),
            "2_1_2_ped_to_far_edge": (
                "Use the pedestrian's current crossing direction and crosswalk context, and answer only with the relevant exit-approach context plus the distance value."
            ),
            "2_1_3_participant_distance": (
                "Answer only with the distance value and keep the unit."
            ),
            "2_1_4_nearest_vehicle_to_ped": (
                "Answer only with the vehicle type, its approach, and the distance."
            ),
            "2_2_1_lane_function": (
                "Judge the current lane assignment rather than the object's future maneuver, and answer only with the lane-function label."
            ),
            "2_2_2_ped_zone": (
                "Answer only with the normalized zone label for this task, and do not add extra explanation."
            ),
            "2_2_3_left_turn_queue_count": (
                "Focus only on that approach-lane pair, return only the count, and do not compare with other lanes."
            ),
            "2_2_4_stopline_back_5m_count": (
                "Return only the count and do not turn the answer into a congestion summary."
            ),
            "2_2_5_longest_queue_lane": (
                "State the winning lane with its approach and concise local queue evidence, without turning the answer into control advice."
            ),
            "2_2_6_crosswalk_blocking": (
                "If yes, mention the blocking vehicle type and the relevant approach briefly; otherwise keep the answer brief."
            ),
            "3_1_1_current_motion_state": (
                "Report the current state together with the current speed in m/s.\n"
                "Use standing/walking/running for pedestrians and stopped/starting/moving/braking/creeping for other objects.\n"
                "For non-pedestrians in starting or braking states, also report accelerating/decelerating together with the current acceleration in m/s^2.\n"
                "Focus only on the current state of the referenced object, without expanding into future intent or maneuver explanation."
            ),
            "3_1_2_vehicle_maneuver": (
                "Focus on the dominant current maneuver only, and do not expand into a full future path or trajectory explanation."
            ),
            "3_2_2_future_region": (
                "Return only the most likely region and avoid expanding into a multi-step path description."
            ),
            "3_2_3_waypoints": (
                "Return the short-horizon future trajectory only as trajectory coordinates in temporal order.\n"
                "Each point must be an (dx,dy) XY offset in meters relative to the referenced object's current position, not an absolute global coordinate.\n"
                "Follow the required `Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)` format exactly and avoid any explanatory text."
            ),
            "3_3_1_safe_following": (
                "Answer with the safety judgment, the current distance, and the time headway only, without expanding into broader collision analysis."
            ),
            "3_3_2_likely_long_queue_lane": (
                "State the winning lane with its approach and concise local queue evidence, without turning the answer into control advice."
            ),
            "3_4_1_vehicle_ped_conflict": (
                "Focus only on this pair, and keep negative answers brief."
            ),
            "3_4_2_nearest_conflict_participant": (
                "Answer only with the participant type plus its location-and-image reference, using X,Y and the available image_name/x/y entries without object ID or speed."
            ),
            "3_4_3_primary_risk_subject": (
                "Answer only with the participant type plus its location-and-image reference, followed by the dominant risk reason, without object ID or speed."
            ),
            "3_4_4_risk_pattern": (
                "Return only the dominant pattern together with the relevant approach, without expanding into a long event description."
            ),
            "4_1_1_overall_state": (
                "Answer concisely and include the key supporting quantity or ratio that best explains the state."
            ),
            "4_1_2_side_motion_status": (
                "Use concise natural wording and include the most important motion-count evidence, without listing individual participants."
            ),
            "4_1_3_scene_summary": (
                "Provide a brief scene-level summary of the current intersection.\n"
                "Keep it short, highlight the main traffic condition, and mention the most notable current phenomenon."
            ),
            "4_1_4_flow_imbalance": (
                "Answer only with the busiest approach and its traffic-participant count.\n"
                "Do not turn the answer into a yes/no imbalance judgment or broader scene explanation."
            ),
            "4_2_1_speeding_risk": (
                "If yes, mention the key risky object or speed evidence briefly; otherwise keep the answer brief."
            ),
            "4_2_2_notable_abnormal": (
                "Identify the most notable current abnormal event in the scene.\n"
                "Mention the abnormal type and the most important location or event evidence without listing multiple abnormalities."
            ),
            "4_3_1_intersection_action": (
                "Focus on the recommended action itself and do not explain the full reasoning chain."
            ),
            "4_3_2_side_action": (
                "Keep the guidance local to that approach and avoid broad scene explanation."
            ),
            "4_3_3_lane_action": (
                "Keep the response focused on that lane and do not list alternatives."
            ),
            "4_3_4_object_action": (
                "Focus on what that object should do now, and avoid discussing other participants or alternative actions."
            ),
        }

    def _single_prompt_metadata(self, modality: str) -> Dict[str, Dict[str, str]]:
        system_prompt = self._image_only_system_prompt()
        if modality == "pointcloud_plus_image":
            system_prompt = self._pointcloud_plus_image_system_prompt()
        return {
            "version": self.PROMPT_METADATA_VERSION,
            "subtemplate_patch_style": self.subtemplate_patch_style,
            "system_prompt": system_prompt,
            "user_prompt_template": self._user_prompt_wrapper(modality),
            "subtemplate_patches": self._subtemplate_patches(),
            "strict_answer_schemas": self._strict_answer_schemas(),
        }

    def _prompt_metadata(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return {
            "default_mode": "image_only",
            "image_only": self._single_prompt_metadata("image_only"),
            "pointcloud_plus_image": self._single_prompt_metadata("pointcloud_plus_image"),
        }

    def _side(self, x: float, y: float) -> str:
        if x < 7.4:
            return "north"
        if x > 28.0:
            return "south"
        if y < 3.0:
            return "west"
        if y > 23.4:
            return "east"
        return "center"

    def _lane(self, x: float, y: float, side: str) -> Optional[str]:
        if side == "center":
            return None
        axis = y if side in {"north", "south"} else x
        for name, lo, hi in self.LANE_RANGES[side]:
            if lo < axis < hi:
                return name
        return None

    def _speed(self, obj: Dict) -> float:
        return float(np.hypot(obj.get("vx", 0.0), obj.get("vy", 0.0)))

    def _make_frame_cache(self, frame_token: str) -> Dict:
        return {
            "frame_token": frame_token,
            "pair_distance": {},
            "future_conflict": {},
            "events": {},
        }

    def _pair_key(self, a: Dict, b: Dict) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        key_a = (str(a.get("frame_token", "")), str(a["id"]))
        key_b = (str(b.get("frame_token", "")), str(b["id"]))
        return (key_a, key_b) if key_a <= key_b else (key_b, key_a)

    def _object_ids_key(self, objects: List[Dict]) -> Tuple[str, ...]:
        return tuple(sorted(str(obj["id"]) for obj in objects))

    def _dist(self, a: Dict, b: Dict) -> float:
        frame_cache = self._active_frame_cache
        if frame_cache is None:
            return self._box_min_distance(a, b)
        key = self._pair_key(a, b)
        cache = frame_cache["pair_distance"]
        if key not in cache:
            cache[key] = self._box_min_distance(a, b)
        return cache[key]

    def _center_dist(self, obj: Dict) -> float:
        return float(np.hypot(obj["x"] - self.CENTER_X, obj["y"] - self.CENTER_Y))

    def _center_distance_between(self, a: Dict, b: Dict) -> float:
        return float(np.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))

    def _turn_map(self) -> Dict[str, Dict[str, str]]:
        return {
            "north": {"south": "straight-going", "east": "turning-left", "west": "turning-right"},
            "south": {"north": "straight-going", "west": "turning-left", "east": "turning-right"},
            "east": {"west": "straight-going", "north": "turning-left", "south": "turning-right"},
            "west": {"east": "straight-going", "south": "turning-left", "north": "turning-right"},
        }

    def _first_non_center(self, track: List[Dict]) -> Optional[str]:
        for obj in track:
            if obj["side"] != "center":
                return obj["side"]
        return None

    def _last_non_center(self, track: List[Dict]) -> Optional[str]:
        for obj in reversed(track):
            if obj["side"] != "center":
                return obj["side"]
        return None

    def _infer_vehicle_status(self, history: List[Dict], current: Dict, future: List[Dict], prev_status: Optional[str]) -> Tuple[str, Dict[str, Optional[str]]]:
        speed = self._speed(current)
        if speed > self.VEHICLE_OVERSPEED_THRESHOLD:
            return "speeding", {"target_lane": None}
        if speed < self.MOVING_THRESHOLD:
            return "stopped", {"target_lane": None}
        full = history + [current] + future
        origin = self._first_non_center(full)
        destination = self._last_non_center([current] + future) or current["side"]
        target_lane = None
        for item in future:
            if item["side"] == current["side"] and item["lane"] not in {None, current["lane"]}:
                target_lane = item["lane"]
                break
        if target_lane is not None and current["side"] != "center":
            return "lane-changing", {"target_lane": target_lane}
        mapping = self._turn_map()
        if origin and destination and origin != destination and destination in mapping.get(origin, {}):
            if current["side"] in {origin, "center"}:
                return mapping[origin][destination], {"target_lane": target_lane}
            return "straight-going", {"target_lane": target_lane}
        if future:
            return "straight-going", {"target_lane": target_lane}
        return prev_status or "straight-going", {"target_lane": target_lane}

    def _infer_non_vehicle_status(self, obj: Dict) -> str:
        speed = self._speed(obj)
        if obj["type"] == "pedestrian":
            if speed > self.PEDESTRIAN_RUNNING_THRESHOLD:
                return "running"
            if speed >= self.PEDESTRIAN_WALKING_THRESHOLD:
                return "walking"
            return "standing"
        return "moving" if speed >= self.MOVING_THRESHOLD else "stopped"

    def _build_track_cache(self) -> None:
        step_options = [int(self.PRED_HORIZON * self.SOURCE_SCENE_FPS), int(2.0 * self.SOURCE_SCENE_FPS), int(1.0 * self.SOURCE_SCENE_FPS)]
        for indices in self.scene_to_indices.values():
            snapshots_per_frame = [self.snapshots_by_token[self._token(self.infos[i])] for i in indices]
            prev_status = {}
            for local_idx, frame_idx in enumerate(indices):
                token = self._token(self.infos[frame_idx])
                current = snapshots_per_frame[local_idx]
                for obj_id, obj in current.items():
                    full_future = [snapshots_per_frame[j].get(obj_id) for j in range(local_idx + 1, len(indices))]
                    self.future_track_cache[(token, obj_id)] = full_future
                    if obj["type"] not in self.VEHICLE_TYPES:
                        self.status_cache[(token, obj_id)] = self._infer_non_vehicle_status(obj)
                        continue
                    future = []
                    for limit in step_options:
                        end_idx = min(local_idx + limit, len(indices) - 1)
                        future = [snapshots_per_frame[j][obj_id] for j in range(local_idx + 1, end_idx + 1) if obj_id in snapshots_per_frame[j]]
                        if future:
                            break
                    history = [snapshots_per_frame[j][obj_id] for j in range(local_idx) if obj_id in snapshots_per_frame[j]]
                    status, motion = self._infer_vehicle_status(history, obj, future, prev_status.get(obj_id))
                    self.status_cache[(token, obj_id)] = status
                    self.motion_cache[(token, obj_id)] = motion
                    prev_status[obj_id] = status

    def _extract_snapshots(self, frame: Dict) -> Dict[str, Dict]:
        snapshots: Dict[str, Dict] = {}
        gt_boxes = frame.get("gt_boxes", np.array([]))
        gt_names = frame.get("gt_names", np.array([]))
        tracking_ids = frame.get("tracking_id", [])
        token = self._token(frame)
        frame_number = self.frame_number_by_token.get(token, 0) if hasattr(self, "frame_number_by_token") else 0
        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            obj_type = self.OBJECT_TYPE_MAPPING.get(str(gt_names[i]), str(gt_names[i]))
            obj_id = str(tracking_ids[i]) if i < len(tracking_ids) else f"{token}_obj_{i}"
            x, y, z = float(box[0]), float(box[1]), float(box[2])
            dx = float(box[3]) if len(box) > 3 else 0.0
            dy = float(box[4]) if len(box) > 4 else 0.0
            dz = float(box[5]) if len(box) > 5 else 0.0
            yaw = float(box[6]) if len(box) > 6 else 0.0
            vx = float(box[7]) if len(box) > 7 else 0.0
            vy = float(box[8]) if len(box) > 8 else 0.0
            side = self._side(x, y)
            lane = self._lane(x, y, side)
            cam_refs = self._camera_refs_for_object(frame, i, np.asarray(box, dtype=float), side)
            primary_ref = cam_refs[0] if cam_refs else self._format_cam_ref(self._default_cam_for_side(side), None)
            snapshots[obj_id] = {
                "id": obj_id,
                "type": obj_type,
                "x": x,
                "y": y,
                "z": z,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "yaw": yaw,
                "vx": vx,
                "vy": vy,
                "side": side,
                "lane": lane,
                "cam_name": primary_ref["cam_name"],
                "cam_view": primary_ref["cam_view"],
                "x1": primary_ref["x1"],
                "y1": primary_ref["y1"],
                "cam_refs": cam_refs,
                "frame_number": frame_number,
            }
        return snapshots

    def extract_objects(self, frame: Dict) -> List[Dict]:
        token = self._token(frame)
        scene_id = self._scene_id(frame)
        out = []
        for obj_id, obj in self.snapshots_by_token[token].items():
            status = self.status_cache.get((token, obj_id), self._infer_non_vehicle_status(obj))
            out.append({**obj, "status": status, "frame_token": token})
        for obj in out:
            obj["ref_id"] = self.scene_ref_id_map.get((scene_id, obj["id"]), obj["id"])
            obj["frame_number"] = self.frame_number_by_token.get(obj["frame_token"], 0)
        return out

    def _default_cam_for_side(self, side: str) -> str:
        mapping = {
            "north": "CAM_SOUTH",
            "south": "CAM_NORTH",
            "east": "CAM_WEST",
            "west": "CAM_EAST",
            "center": "CAM_SOUTH",
        }
        return mapping.get(side, "CAM_SOUTH")

    def _box_corners_3d(self, box: np.ndarray) -> np.ndarray:
        cx, cy, cz = float(box[0]), float(box[1]), float(box[2])
        dx = float(box[3]) if len(box) > 3 else 0.0
        dy = float(box[4]) if len(box) > 4 else 0.0
        dz = float(box[5]) if len(box) > 5 else 0.0
        yaw = float(box[6]) if len(box) > 6 else 0.0
        hx = max(dx, 0.1) / 2.0
        hy = max(dy, 0.1) / 2.0
        hz = max(dz, 0.1) / 2.0
        corners = np.asarray(
            [
                [hx, hy, hz],
                [hx, -hy, hz],
                [-hx, -hy, hz],
                [-hx, hy, hz],
                [hx, hy, -hz],
                [hx, -hy, -hz],
                [-hx, -hy, -hz],
                [-hx, hy, -hz],
            ],
            dtype=float,
        )
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return corners @ rot.T + np.asarray([cx, cy, cz], dtype=float)

    def _project_box_to_camera_bbox(self, frame: Dict, box: np.ndarray, cam_name: str) -> Optional[np.ndarray]:
        cams = frame.get("cams") or {}
        cam = cams.get(cam_name) or {}
        lidar2camera = np.asarray(cam.get("lidar2camera")) if cam.get("lidar2camera") is not None else None
        lidar2image = np.asarray(cam.get("lidar2image")) if cam.get("lidar2image") is not None else None
        if lidar2camera is None or lidar2image is None:
            return None
        points = []
        for point in self._box_corners_3d(box):
            hom = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
            cam_pt = lidar2camera @ hom
            if float(cam_pt[2]) <= 0.01:
                continue
            img_pt = lidar2image @ hom
            if abs(float(img_pt[2])) <= 1e-6:
                continue
            points.append((float(img_pt[0] / img_pt[2]), float(img_pt[1] / img_pt[2])))
        if not points:
            return None
        pts = np.asarray(points, dtype=float)
        return np.asarray([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=float)

    def _bbox_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        return float(inter / (aa + bb - inter + 1e-6))

    def _format_cam_ref(self, cam_name: str, bbox: Optional[np.ndarray] = None) -> Dict:
        if bbox is None or len(bbox) < 4:
            x1 = y1 = None
            area = 0.0
        else:
            x1 = float((float(bbox[0]) + float(bbox[2])) / 2.0)
            y1 = float((float(bbox[1]) + float(bbox[3])) / 2.0)
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        return {
            "cam_name": cam_name,
            "cam_view": self.CAM_NAME_TO_VIEW.get(cam_name, "north_image"),
            "x1": x1,
            "y1": y1,
            "area": area,
        }

    def _dedupe_and_order_cam_refs(self, cam_refs: List[Dict], cam_keys: List[str]) -> List[Dict]:
        best_by_cam: Dict[str, Dict] = {}
        for item in cam_refs:
            cam_name = str(item.get("cam_name"))
            previous = best_by_cam.get(cam_name)
            if previous is None or float(item.get("area", 0.0)) > float(previous.get("area", 0.0)):
                best_by_cam[cam_name] = item
        ordered: List[Dict] = []
        for cam_name in cam_keys:
            if cam_name in best_by_cam:
                ordered.append(best_by_cam[cam_name])
        for cam_name in sorted(k for k in best_by_cam if k not in cam_keys):
            ordered.append(best_by_cam[cam_name])
        for item in ordered:
            item.pop("area", None)
        return ordered

    def _camera_refs_for_object(self, frame: Dict, obj_index: int, box: np.ndarray, side: str) -> List[Dict]:
        cam_keys = list(frame.get("cam_keys") or [])
        refs: List[Dict] = []

        centers_multi = frame.get("gt_centers_2d_multi")
        mask_multi = frame.get("gt_boxes_2d_multi_mask")
        boxes_multi = frame.get("gt_boxes_2d_multi")
        if (
            isinstance(centers_multi, np.ndarray)
            and isinstance(mask_multi, np.ndarray)
            and obj_index < len(centers_multi)
            and mask_multi.ndim >= 2
        ):
            for cam_index, cam_name in enumerate(cam_keys):
                if cam_index >= mask_multi.shape[1] or not bool(mask_multi[obj_index, cam_index]):
                    continue
                bbox = None
                if isinstance(boxes_multi, np.ndarray) and boxes_multi.ndim >= 3 and obj_index < len(boxes_multi) and cam_index < boxes_multi.shape[1]:
                    candidate_bbox = np.asarray(boxes_multi[obj_index, cam_index], dtype=float)
                    if candidate_bbox.shape[0] >= 4 and float(candidate_bbox[0]) >= 0.0 and float(candidate_bbox[1]) >= 0.0:
                        bbox = candidate_bbox
                if bbox is None:
                    center = np.asarray(centers_multi[obj_index, cam_index], dtype=float)
                    if center.shape[0] >= 2 and float(center[0]) >= 0.0 and float(center[1]) >= 0.0:
                        bbox = np.asarray([center[0], center[1], center[0], center[1]], dtype=float)
                refs.append(self._format_cam_ref(cam_name, bbox))
            if refs:
                return self._dedupe_and_order_cam_refs(refs, cam_keys)

        all_boxes = frame.get("all_2d_boxes")
        all_obj_indices = frame.get("all_2d_box_obj_indices")
        if isinstance(all_boxes, np.ndarray) and isinstance(all_obj_indices, np.ndarray) and len(all_boxes) == len(all_obj_indices):
            for row, matched_obj_idx in zip(all_boxes, all_obj_indices):
                if int(matched_obj_idx) != obj_index or len(row) < 5:
                    continue
                cam_index = int(row[4])
                if not (0 <= cam_index < len(cam_keys)):
                    continue
                refs.append(self._format_cam_ref(cam_keys[cam_index], np.asarray(row[:4], dtype=float)))
            if refs:
                return self._dedupe_and_order_cam_refs(refs, cam_keys)

        if isinstance(all_boxes, np.ndarray) and len(all_boxes) and cam_keys:
            for cam_index, cam_name in enumerate(cam_keys):
                projected = self._project_box_to_camera_bbox(frame, box, cam_name)
                if projected is None:
                    continue
                candidates = all_boxes[all_boxes[:, 4].astype(np.int32) == cam_index]
                best_bbox = None
                best_iou = 0.0
                for candidate in candidates:
                    if len(candidate) < 4:
                        continue
                    overlap = self._bbox_iou(projected, np.asarray(candidate[:4], dtype=float))
                    if overlap > best_iou:
                        best_iou = overlap
                        best_bbox = np.asarray(candidate[:4], dtype=float)
                if best_bbox is not None and best_iou > 0.05:
                    refs.append(self._format_cam_ref(cam_name, best_bbox))
            if refs:
                return self._dedupe_and_order_cam_refs(refs, cam_keys)

        gt_boxes_2d = frame.get("gt_boxes_2d", np.array([]))
        if obj_index < len(gt_boxes_2d):
            bbox = np.asarray(gt_boxes_2d[obj_index], dtype=float)
            cam_name = None
            if len(bbox) >= 5:
                cam_idx = int(bbox[4])
                if 0 <= cam_idx < len(cam_keys):
                    cam_name = cam_keys[cam_idx]
            if cam_name is None:
                cam_name = self._default_cam_for_side(side)
            if len(bbox) >= 4:
                refs.append(self._format_cam_ref(cam_name, bbox[:4]))
            else:
                refs.append(self._format_cam_ref(cam_name, None))
            return self._dedupe_and_order_cam_refs(refs, cam_keys)

        return [self._format_cam_ref(self._default_cam_for_side(side), None)]

    def _coarse_group(self, obj_type: str) -> str:
        return "VRU" if obj_type in self.VRU_TYPES else "Vehicle"

    def _camera_label(self, cam_view: str) -> str:
        return str(cam_view).replace("_image", "")

    def _side_location_phrase(self, side: str) -> str:
        if side == "center":
            return "in the center area of the intersection"
        return f"on the {side} approach of the intersection"

    def _side_subject(self, side: str) -> str:
        if side == "center":
            return "the center area"
        return f"the {side} approach"

    def _object_phrase(self, obj: Dict, include_side: bool = True, capitalized: bool = False) -> str:
        phrase = f"the {obj['type']}"
        if include_side:
            if obj["side"] == "center":
                phrase = f"{phrase} in the center area"
            else:
                phrase = f"{phrase} on the {obj['side']} approach"
        if capitalized:
            return phrase[:1].upper() + phrase[1:]
        return phrase

    def _crosswalk_phrase(self, crosswalk: str) -> str:
        return f"{crosswalk} crosswalk"

    def _ped_zone_phrase(self, crosswalk: str, region: str) -> str:
        if region == "crosswalk":
            return f"within the {crosswalk} crosswalk"
        if region == "entry zone":
            return f"in the {crosswalk} crosswalk entry zone"
        if region == "exit zone":
            return f"in the {crosswalk} crosswalk exit zone"
        if region == "waiting zone":
            return f"in the waiting zone near the {crosswalk} crosswalk"
        return "none of the marked crosswalk-related zones"

    def _turn_reason_phrase(self, reason: str) -> str:
        return {
            "proximity": "proximity",
            "path_crossing": "path crossing",
            "overspeed": "overspeed",
            "lane_change_conflict": "lane change conflict",
            "vru_conflict": "VRU conflict",
        }.get(reason, reason.replace("_", " "))

    def _future_region_phrase(self, region: str) -> str:
        return {
            "before stop line": "remain before the stop line",
            "intersection center": "move into the intersection center",
            "through exit": "move toward the through exit",
            "left-turn exit": "move toward the left-turn exit",
            "right-turn exit": "move toward the right-turn exit",
        }.get(region, f"move toward the {region}")

    def _relative_direction_info(self, ref_obj: Dict, other_obj: Dict) -> Tuple[str, float]:
        rel = np.array([float(other_obj["x"]) - float(ref_obj["x"]), float(other_obj["y"]) - float(ref_obj["y"])], dtype=float)
        yaw = float(ref_obj.get("yaw", 0.0))
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)
        forward_coord = float(np.dot(rel, forward))
        left_coord = float(np.dot(rel, left))
        angle = float(np.degrees(np.arctan2(left_coord, forward_coord)))
        if -22.5 <= angle < 22.5:
            return "front", abs(angle - 0.0)
        if 22.5 <= angle < 67.5:
            return "front-left", abs(angle - 45.0)
        if 67.5 <= angle < 112.5:
            return "left", abs(angle - 90.0)
        if 112.5 <= angle < 157.5:
            return "rear-left", abs(angle - 135.0)
        if angle >= 157.5 or angle < -157.5:
            wrapped_delta = min(abs(angle - 180.0), abs(angle + 180.0))
            return "behind", wrapped_delta
        if -157.5 <= angle < -112.5:
            return "rear-right", abs(angle + 135.0)
        if -112.5 <= angle < -67.5:
            return "right", abs(angle + 90.0)
        return "front-right", abs(angle + 45.0)

    def _relative_direction_label(self, ref_obj: Dict, other_obj: Dict) -> str:
        return self._relative_direction_info(ref_obj, other_obj)[0]

    def _ref(self, obj: Dict) -> str:
        parts = [
            str(obj["ref_id"]),
            self._format_relative_scene_time(obj["frame_token"]),
            f"{float(obj['x']):.1f}",
            f"{float(obj['y']):.1f}",
        ]
        cam_refs = list(obj.get("cam_refs") or [])
        if not cam_refs:
            cam_refs = [
                {
                    "cam_view": obj.get("cam_view", "north_image"),
                    "x1": obj.get("x1"),
                    "y1": obj.get("y1"),
                }
            ]
        for cam_ref in cam_refs:
            x1 = -1.0 if cam_ref.get("x1") is None else float(cam_ref["x1"])
            y1 = -1.0 if cam_ref.get("y1") is None else float(cam_ref["y1"])
            parts.extend([str(cam_ref.get("cam_view", "north_image")), f"{x1:.1f}", f"{y1:.1f}"])
        return f"<{','.join(parts)}>"

    def _location_ref(self, obj: Dict) -> str:
        parts = [
            f"{float(obj['x']):.1f}",
            f"{float(obj['y']):.1f}",
        ]
        cam_refs = list(obj.get("cam_refs") or [])
        if not cam_refs:
            cam_refs = [
                {
                    "cam_view": obj.get("cam_view", "north_image"),
                    "x1": obj.get("x1"),
                    "y1": obj.get("y1"),
                }
            ]
        for cam_ref in cam_refs:
            x1 = -1.0 if cam_ref.get("x1") is None else float(cam_ref["x1"])
            y1 = -1.0 if cam_ref.get("y1") is None else float(cam_ref["y1"])
            parts.extend([str(cam_ref.get("cam_view", "north_image")), f"{x1:.1f}", f"{y1:.1f}"])
        return f"<{','.join(parts)}>"

    def _v5_qa(
        self,
        template_id: str,
        question: str,
        answer,
        priority: float = 0.0,
        structured_targets: Optional[Dict] = None,
        answer_bucket: Optional[str] = None,
        sample_meta: Optional[Dict] = None,
        placeholder: bool = False,
    ) -> Dict:
        spec = self.template_registry[template_id]
        qa = self._qa(spec.chapter, spec.section, template_id, question, answer, priority, answer_bucket, structured_targets, sample_meta, placeholder)
        qa["chapter"] = spec.chapter
        qa["section"] = spec.section
        return qa

    def _template_cap(self, template_id: str, max_per_type: int) -> int:
        policy = self.sampling_policy_registry.get(template_id, {})
        fixed_cap = policy.get("fixed_cap")
        cap = max_per_type if fixed_cap is None else min(max_per_type, int(fixed_cap))
        return max(1, int(cap))

    def _yes_no_label(self, qa: Dict) -> Optional[str]:
        answer = qa.get("answer")
        if isinstance(answer, str):
            normalized = answer.strip().lower()
            if normalized.startswith("yes"):
                return "yes"
            if normalized.startswith("no"):
                return "no"
        meta = qa.get("_sample_meta", {})
        for key in ("exists", "crosswalk_blocked", "is_safe", "will_enter_crosswalk", "has_conflict", "has_abnormal"):
            if key in meta:
                return "yes" if bool(meta[key]) else "no"
        return None

    def _coverage_value(self, qa: Dict, key: str):
        meta = qa.get("_sample_meta", {})
        if key in meta:
            return meta[key]
        structured = qa.get("structured_targets") or {}
        if key in structured:
            return structured[key]
        nested = structured.get("numeric_targets")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        return None

    def _rank_tuple(self, qa: Dict) -> Tuple:
        return (-float(qa.get("_sampling_priority", 0.0)), bool(qa.get("placeholder", False)), qa["question"])

    def _normalize_bucket_value(self, value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "None"
        if isinstance(value, (dict, list)):
            return "structured"
        return str(value)

    def _infer_bucket_for_statistics(self, qa: Dict) -> Optional[str]:
        template_id = str(qa.get("subtemplate") or "")
        structured = qa.get("structured_targets") or {}
        field = self.BUCKET_FIELD_BY_TEMPLATE.get(template_id)
        if field is not None and field in structured:
            value = structured[field]
            return self._normalize_bucket_value(value)
        if field == "trajectory" and "trajectory" in structured:
            return "trajectory"
        if field == "signal_state" and qa.get("placeholder"):
            return "None"
        answer = qa.get("answer")
        if qa.get("placeholder") and answer is None:
            return "None"
        if isinstance(answer, str):
            lowered = answer.strip().lower()
            if lowered.startswith("yes"):
                return "yes"
            if lowered.startswith("no"):
                return "no"
        return None

    def _compute_statistics(self, qas: List[Dict]):
        chapter_stats = defaultdict(int)
        section_stats = defaultdict(int)
        template_stats = defaultdict(int)
        bucket_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for qa in qas:
            chapter_stats[qa["chapter"]] += 1
            section_stats[qa["section"]] += 1
            template_stats[qa["subtemplate"]] += 1
            bucket = self._infer_bucket_for_statistics(qa)
            if bucket is not None:
                bucket_stats[qa["subtemplate"]][bucket] += 1
        return (
            dict(sorted(chapter_stats.items())),
            dict(sorted(section_stats.items())),
            dict(sorted(template_stats.items())),
            {k: dict(sorted(v.items())) for k, v in sorted(bucket_stats.items())},
        )

    def _apply_yes_no_ratio(self, items: List[Dict], ratio: Tuple[int, int], cap: int) -> List[Dict]:
        if not items or cap <= 0:
            return []
        positives = [item for item in items if self._yes_no_label(item) == "yes"]
        negatives = [item for item in items if self._yes_no_label(item) == "no"]
        others = [item for item in items if self._yes_no_label(item) not in {"yes", "no"}]
        positives.sort(key=self._rank_tuple)
        negatives.sort(key=self._rank_tuple)
        others.sort(key=self._rank_tuple)
        pos_weight, neg_weight = ratio
        total_weight = max(pos_weight + neg_weight, 1)
        pos_quota = int(round(cap * pos_weight / total_weight))
        pos_quota = min(len(positives), max(0, pos_quota))
        neg_quota = min(len(negatives), max(0, cap - pos_quota))
        selected = positives[:pos_quota] + negatives[:neg_quota]
        used_ids = {id(item) for item in selected}
        leftovers = [item for item in positives[pos_quota:] + negatives[neg_quota:] + others if id(item) not in used_ids]
        leftovers.sort(key=self._rank_tuple)
        selected.extend(leftovers[: max(0, cap - len(selected))])
        return selected[:cap]

    def _select_template_items(self, template_id: str, items: List[Dict], max_per_type: int) -> List[Dict]:
        if not items:
            return []
        policy = self.sampling_policy_registry.get(template_id, {})
        cap = self._template_cap(template_id, max_per_type)
        if len(items) <= cap:
            chosen = list(items)
            if "yes_no_ratio" in policy:
                return self._apply_yes_no_ratio(chosen, policy["yes_no_ratio"], cap)
            chosen.sort(key=self._rank_tuple)
            return chosen
        filtered = list(items)
        for key, excluded in (policy.get("exclude_values") or {}).items():
            keep = [item for item in filtered if self._coverage_value(item, key) not in excluded]
            if keep:
                filtered = keep
        filtered.sort(key=self._rank_tuple)
        selected: List[Dict] = []
        selected_ids = set()
        for coverage_key in policy.get("coverage_priority", ()):
            seen_values = set()
            for item in filtered:
                item_id = id(item)
                if item_id in selected_ids:
                    continue
                value = self._coverage_value(item, coverage_key)
                if value is None or value in seen_values:
                    continue
                selected.append(item)
                selected_ids.add(item_id)
                seen_values.add(value)
                if len(selected) >= cap:
                    break
            if len(selected) >= cap:
                break
        if len(selected) < cap:
            for item in filtered:
                item_id = id(item)
                if item_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item_id)
                if len(selected) >= cap:
                    break
        if "yes_no_ratio" in policy:
            return self._apply_yes_no_ratio(selected, policy["yes_no_ratio"], cap)
        return selected[:cap]

    def _format_v5_waypoints(self, waypoints: List[Dict[str, float]]) -> str:
        coords = [f"({item['dx']:.2f},{item['dy']:.2f})" for item in waypoints]
        return "Future trajectory:" + ",".join(coords)

    def _diverse_keep_key(
        self,
        qa: Dict,
        selected_direction_counts: Dict[str, int],
        selected_scene_times: Dict[str, List[float]],
    ) -> Tuple:
        sample_meta = qa.get("_sample_meta", {})
        direction = str(sample_meta.get("direction") or "unknown")
        direction_count = int(selected_direction_counts.get(direction, 0))
        time_value = self._relative_scene_time_seconds(qa["frame_token"])
        scene_times = selected_scene_times.get(qa["scene_id"], [])
        min_gap = min((abs(time_value - existing) for existing in scene_times), default=float("inf"))
        finite_gap = 1e9 if min_gap == float("inf") else float(min_gap)
        return (
            direction_count,
            -finite_gap,
            -float(qa.get("_sampling_priority", 0.0)),
            qa.get("question_id", ""),
        )

    def _select_scene_diverse_subset(self, items: List[Dict], cap: int) -> List[Dict]:
        if len(items) <= cap:
            return list(items)

        by_scene: Dict[str, List[Dict]] = defaultdict(list)
        for qa in items:
            by_scene[qa["scene_id"]].append(qa)

        selected: List[Dict] = []
        selected_direction_counts: Dict[str, int] = defaultdict(int)
        selected_scene_times: Dict[str, List[float]] = defaultdict(list)

        while len(selected) < cap:
            progress = False
            for scene_id in sorted(by_scene):
                candidates = by_scene[scene_id]
                if not candidates:
                    continue
                best = min(
                    candidates,
                    key=lambda qa: self._diverse_keep_key(
                        qa,
                        selected_direction_counts,
                        selected_scene_times,
                    ),
                )
                candidates.remove(best)
                selected.append(best)
                direction = str((best.get("_sample_meta") or {}).get("direction") or "unknown")
                selected_direction_counts[direction] += 1
                selected_scene_times[scene_id].append(self._relative_scene_time_seconds(best["frame_token"]))
                progress = True
                if len(selected) >= cap:
                    break
            if not progress:
                break

        return selected[:cap]

    def _cap_fine_type_template(self, qas: List[Dict], cap: int) -> List[Dict]:
        fine_type_items = [qa for qa in qas if qa["subtemplate"] == "1_1_1_fine_type"]
        if not fine_type_items:
            return qas

        kept_question_ids = set()
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for qa in fine_type_items:
            object_type = str((qa.get("structured_targets") or {}).get("object_type") or "")
            by_type[object_type].append(qa)

        for object_type, items in by_type.items():
            if len(items) <= cap:
                kept_question_ids.update(qa["question_id"] for qa in items)
                continue
            selected = self._select_scene_diverse_subset(items, cap)
            kept_question_ids.update(qa["question_id"] for qa in selected)

        return [
            qa
            for qa in qas
            if qa["subtemplate"] != "1_1_1_fine_type" or qa["question_id"] in kept_question_ids
        ]

    def _apply_coarse_group_ratio(self, qas: List[Dict], template_id: str, major_label: str, minor_label: str, major_weight: int, minor_weight: int) -> List[Dict]:
        target_items = [qa for qa in qas if qa["subtemplate"] == template_id]
        if not target_items:
            return qas

        by_group: Dict[str, List[Dict]] = defaultdict(list)
        for qa in target_items:
            group = str((qa.get("structured_targets") or {}).get("coarse_group") or "")
            by_group[group].append(qa)

        major_items = by_group.get(major_label, [])
        minor_items = by_group.get(minor_label, [])
        if not major_items or not minor_items:
            return qas

        max_major = (len(minor_items) * major_weight) // max(minor_weight, 1)

        kept_question_ids = set()
        if len(major_items) > max_major:
            kept_question_ids.update(qa["question_id"] for qa in self._select_scene_diverse_subset(major_items, max_major))
        else:
            kept_question_ids.update(qa["question_id"] for qa in major_items)

        kept_question_ids.update(qa["question_id"] for qa in minor_items)

        for group, items in by_group.items():
            if group not in {major_label, minor_label}:
                kept_question_ids.update(qa["question_id"] for qa in items)

        return [
            qa
            for qa in qas
            if qa["subtemplate"] != template_id or qa["question_id"] in kept_question_ids
        ]

    def _apply_template_yes_no_ratio(self, qas: List[Dict], template_id: str, yes_weight: int, no_weight: int) -> List[Dict]:
        target_items = [qa for qa in qas if qa["subtemplate"] == template_id]
        if not target_items:
            return qas

        yes_items = [qa for qa in target_items if self._yes_no_label(qa) == "yes"]
        no_items = [qa for qa in target_items if self._yes_no_label(qa) == "no"]
        other_items = [qa for qa in target_items if self._yes_no_label(qa) not in {"yes", "no"}]

        if not yes_items or not no_items:
            return qas

        scale = min(len(yes_items) / max(yes_weight, 1), len(no_items) / max(no_weight, 1))
        yes_keep = min(len(yes_items), int(yes_weight * scale))
        no_keep = min(len(no_items), int(no_weight * scale))

        kept_question_ids = set()
        kept_question_ids.update(qa["question_id"] for qa in self._select_scene_diverse_subset(yes_items, yes_keep))
        kept_question_ids.update(qa["question_id"] for qa in self._select_scene_diverse_subset(no_items, no_keep))
        kept_question_ids.update(qa["question_id"] for qa in other_items)

        return [
            qa
            for qa in qas
            if qa["subtemplate"] != template_id or qa["question_id"] in kept_question_ids
        ]

    def _apply_template_label_ratio(
        self,
        qas: List[Dict],
        template_id: str,
        label_weights: Dict[str, int],
        field_name: str,
        fallback_label: Optional[str] = None,
    ) -> List[Dict]:
        target_items = [qa for qa in qas if qa["subtemplate"] == template_id]
        if not target_items:
            return qas

        by_label: Dict[str, List[Dict]] = defaultdict(list)
        for qa in target_items:
            value = (qa.get("structured_targets") or {}).get(field_name)
            if value is None and fallback_label is not None:
                label = fallback_label
            else:
                label = str(value or "")
            by_label[label].append(qa)

        configured_present = {
            label: items
            for label, items in by_label.items()
            if label in label_weights and items
        }
        if not configured_present:
            return qas

        scale = min(len(items) / max(label_weights[label], 1) for label, items in configured_present.items())
        kept_question_ids = set()

        for label, items in by_label.items():
            if label not in label_weights:
                kept_question_ids.update(qa["question_id"] for qa in items)
                continue
            target_keep = int(label_weights[label] * scale)
            if len(items) <= target_keep:
                kept_question_ids.update(qa["question_id"] for qa in items)
            else:
                kept_question_ids.update(
                    qa["question_id"] for qa in self._select_scene_diverse_subset(items, target_keep)
                )

        return [
            qa
            for qa in qas
            if qa["subtemplate"] != template_id or qa["question_id"] in kept_question_ids
        ]

    def _final_ratio_label(self, qa: Dict, field_name: str) -> str:
        value = (qa.get("structured_targets") or {}).get(field_name)
        if value is None:
            return ""
        return self._normalize_bucket_value(value)

    def _apply_template_final_ratio(self, qas: List[Dict], template_id: str, ratio: float) -> List[Dict]:
        target_items = [qa for qa in qas if qa["subtemplate"] == template_id]
        if not target_items:
            return qas

        field_name = self.BUCKET_FIELD_BY_TEMPLATE.get(template_id)
        exempt_labels = self.TEMPLATE_FINAL_RATIO_EXEMPT_LABELS.get(template_id, set())
        included_labels = self.TEMPLATE_FINAL_RATIO_INCLUDED_LABELS.get(template_id)
        label_ratios = self.TEMPLATE_FINAL_RATIO_BY_LABEL.get(template_id, {})

        if not field_name:
            keep_count = int(round(len(target_items) * float(ratio)))
            if keep_count < 1:
                keep_count = 1
            keep_count = min(len(target_items), keep_count)
            kept_question_ids = {
                qa["question_id"] for qa in self._select_scene_diverse_subset(target_items, keep_count)
            }
            return [
                qa
                for qa in qas
                if qa["subtemplate"] != template_id or qa["question_id"] in kept_question_ids
            ]

        exempt_items: List[Dict] = []
        eligible_by_label: Dict[str, List[Dict]] = defaultdict(list)
        for qa in target_items:
            label = self._final_ratio_label(qa, field_name)
            if label in exempt_labels or (included_labels is not None and label not in included_labels):
                exempt_items.append(qa)
            else:
                eligible_by_label[label].append(qa)

        kept_question_ids = {qa["question_id"] for qa in exempt_items}
        for label, label_items in eligible_by_label.items():
            if not label_items:
                continue
            label_ratio = float(label_ratios.get(label, ratio))
            keep_count = int(round(len(label_items) * label_ratio))
            if keep_count < 1:
                keep_count = 1
            keep_count = min(len(label_items), keep_count)
            kept_question_ids.update(
                qa["question_id"] for qa in self._select_scene_diverse_subset(label_items, keep_count)
            )

        return [
            qa
            for qa in qas
            if qa["subtemplate"] != template_id or qa["question_id"] in kept_question_ids
        ]

    def _freeze_suppression_value(self, value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "None"
        if isinstance(value, float):
            return round(float(value), 3)
        if isinstance(value, (int, str)):
            return value
        if isinstance(value, dict):
            return tuple(
                (str(key), self._freeze_suppression_value(subvalue))
                for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_suppression_value(item) for item in value)
        return str(value)

    def _temporal_suppression_key(self, qa: Dict) -> Tuple:
        template_id = str(qa.get("subtemplate") or "")
        structured = qa.get("structured_targets") or {}
        sample_meta = qa.get("_sample_meta") or {}
        scene_id = str(qa.get("scene_id") or "")
        key_parts: List[Tuple[str, object]] = []

        if template_id == "1_3_2_vehicle_signal_state":
            key_parts.append(("question", str(qa.get("question") or "")))
        else:
            for field in (
                "ref_id",
                "object_id",
                "pedestrian_id",
                "vehicle_id",
                "vru_id",
                "follower_id",
                "leader_id",
                "obj1_id",
                "obj2_id",
                "focus_id",
                "subject_id",
            ):
                value = structured.get(field)
                if value in (None, ""):
                    value = sample_meta.get(field)
                if value not in (None, ""):
                    key_parts.append((field, self._freeze_suppression_value(value)))
            for field in (
                "side",
                "lane_function",
                "crosswalk",
                "stopline_side",
                "direction",
                "object_type",
                "pair_type",
                "turn_direction",
            ):
                value = structured.get(field)
                if value in (None, ""):
                    value = sample_meta.get(field)
                if value not in (None, ""):
                    key_parts.append((field, self._freeze_suppression_value(value)))
            if not key_parts:
                question = str(qa.get("question") or "")
                question = re.sub(r"<[^>]+>", "<REF>", question)
                key_parts.append(("question", question))
        return (scene_id, template_id, *key_parts)

    def _temporal_suppression_state(self, qa: Dict) -> Tuple:
        structured = qa.get("structured_targets") or {}
        if structured:
            return (self._freeze_suppression_value(structured),)
        return (
            self._freeze_suppression_value(qa.get("answer")),
            self._freeze_suppression_value(bool(qa.get("placeholder"))),
        )

    def _apply_temporal_state_suppression(self, qas: List[Dict]) -> List[Dict]:
        target_items = list(qas)
        if not target_items:
            return qas

        ordered_items = sorted(
            target_items,
            key=lambda qa: (
                str(qa.get("scene_id") or ""),
                self._relative_scene_time_seconds(str(qa.get("frame_token") or "")),
                str(qa.get("question_id") or ""),
            ),
        )
        kept_question_ids = set()
        last_kept_by_key: Dict[Tuple, Tuple[Tuple, float]] = {}
        threshold = float(self.TEMPLATE_TEMPORAL_SUPPRESSION_THRESHOLD_SEC)

        for qa in ordered_items:
            key = self._temporal_suppression_key(qa)
            state = self._temporal_suppression_state(qa)
            if key is None or state is None:
                kept_question_ids.add(qa["question_id"])
                continue
            time_value = self._relative_scene_time_seconds(str(qa.get("frame_token") or ""))
            previous = last_kept_by_key.get(key)
            if previous is None:
                kept_question_ids.add(qa["question_id"])
                last_kept_by_key[key] = (state, time_value)
                continue
            previous_state, previous_time = previous
            if state != previous_state or (time_value - previous_time) >= threshold:
                kept_question_ids.add(qa["question_id"])
                last_kept_by_key[key] = (state, time_value)

        return [
            qa
            for qa in qas
            if qa["question_id"] in kept_question_ids
        ]

    def _apply_global_template_caps(self, qas: List[Dict]) -> List[Dict]:
        result = list(qas)
        fine_type_cap = self.TEMPLATE_GLOBAL_BUCKET_CAPS.get("1_1_1_fine_type")
        if fine_type_cap is not None:
            result = self._cap_fine_type_template(result, int(fine_type_cap))
        for template_id, (major_label, minor_label, major_weight, minor_weight) in self.TEMPLATE_GLOBAL_GROUP_RATIOS.items():
            result = self._apply_coarse_group_ratio(
                result,
                template_id,
                major_label,
                minor_label,
                int(major_weight),
                int(minor_weight),
            )
        for template_id, (yes_weight, no_weight) in self.TEMPLATE_GLOBAL_YES_NO_RATIOS.items():
            result = self._apply_template_yes_no_ratio(
                result,
                template_id,
                int(yes_weight),
                int(no_weight),
            )
        for template_id, label_weights in self.TEMPLATE_GLOBAL_LABEL_RATIOS.items():
            if template_id == "1_1_1_fine_type":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "object_type",
                )
            elif template_id == "1_1_3_side_count":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "object_type",
                )
            elif template_id == "1_2_1_size_bucket":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "size_bucket",
                )
            elif template_id == "2_1_1_stopline_distance":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "object_type",
                )
            elif template_id == "2_1_2_ped_to_far_edge":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "crosswalk",
                )
            elif template_id == "2_1_4_nearest_vehicle_to_ped":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "direction",
                )
            elif template_id == "2_2_1_lane_function":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "lane_function",
                )
            elif template_id == "2_2_2_ped_zone":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "ped_zone",
                )
            elif template_id == "2_2_3_left_turn_queue_count":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "side",
                )
            elif template_id == "2_2_4_stopline_back_5m_count":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "side",
                )
            elif template_id == "2_2_5_longest_queue_lane":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "lane_function",
                )
            elif template_id == "3_1_1_current_motion_state":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "motion_state",
                )
            elif template_id == "3_1_2_vehicle_maneuver":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "maneuver",
                )
            elif template_id == "3_3_2_likely_long_queue_lane":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "lane_function",
                )
            elif template_id == "3_2_2_future_region":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "future_region",
                )
            elif template_id == "3_4_3_primary_risk_subject":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "risk_reason",
                )
            elif template_id == "4_1_1_overall_state":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "overall_state",
                )
            elif template_id == "4_1_2_side_motion_status":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "motion_label",
                )
            elif template_id == "4_2_2_notable_abnormal":
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "notable_abnormal",
                )
            elif template_id in {
                "4_3_1_intersection_action",
                "4_3_2_side_action",
                "4_3_3_lane_action",
                "4_3_4_object_action",
            }:
                result = self._apply_template_label_ratio(
                    result,
                    template_id,
                    label_weights,
                    "action_state",
                )
        return result

    def _apply_final_template_ratios(self, qas: List[Dict]) -> List[Dict]:
        result = list(qas)
        for template_id, ratio in self.TEMPLATE_FINAL_RATIOS.items():
            result = self._apply_template_final_ratio(result, template_id, float(ratio))
        return result

    def _public_qa(self, qa: Dict) -> Dict:
        cleaned = dict(qa)
        cleaned.pop("_answer_bucket", None)
        cleaned.pop("_sample_meta", None)
        cleaned.pop("_sampling_priority", None)
        return cleaned

    def _v5_queue_prediction_answer(self, side: str, lane: str, lane_objs: List[Dict]) -> str:
        stopped, slow, moving = self._queue_prediction_counts(lane_objs)
        parts = []
        if stopped:
            parts.append(f"{stopped} stopped vehicles")
        if slow:
            parts.append(f"{slow} creeping vehicles")
        if moving:
            parts.append(f"{moving} moving vehicles")
        evidence = ", ".join(parts) if parts else "no strong queue evidence"
        return f"The {self._lane_function_label(lane)} on the {side} approach is most likely to form a long queue because it already contains {evidence}."

    def _contains(self, x: float, y: float, rect: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]) -> bool:
        x0, x1, y0, y1 = rect
        x_ok = True
        y_ok = True
        if x0 is not None:
            x_ok = x_ok and x >= x0
        if x1 is not None:
            x_ok = x_ok and x < x1
        if y0 is not None:
            y_ok = y_ok and y >= y0
        if y1 is not None:
            y_ok = y_ok and y < y1
        return x_ok and y_ok

    def _point_to_rect_distance(self, x: float, y: float, rect: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]) -> float:
        x0, x1, y0, y1 = rect
        if (x0 is None or x >= x0) and (x1 is None or x < x1):
            dx = 0.0
        elif x0 is None:
            dx = abs(x - x1)
        elif x1 is None:
            dx = abs(x - x0)
        else:
            dx = min(abs(x - x0), abs(x - x1))
        if (y0 is None or y >= y0) and (y1 is None or y < y1):
            dy = 0.0
        elif y0 is None:
            dy = abs(y - y1)
        elif y1 is None:
            dy = abs(y - y0)
        else:
            dy = min(abs(y - y0), abs(y - y1))
        return float(np.hypot(dx, dy))

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-8:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _cross_2d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _segments_intersect(self, a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
        r = a2 - a1
        s = b2 - b1
        denom = self._cross_2d(r, s)
        qp = b1 - a1
        if abs(denom) <= 1e-8:
            if abs(self._cross_2d(qp, r)) > 1e-8:
                return False
            rr = float(np.dot(r, r))
            if rr <= 1e-8:
                return float(np.linalg.norm(a1 - b1)) <= 1e-8
            t0 = float(np.dot(qp, r) / rr)
            t1 = float(np.dot(qp + s, r) / rr)
            lo, hi = sorted((t0, t1))
            return hi >= 0.0 and lo <= 1.0
        t = self._cross_2d(qp, s) / denom
        u = self._cross_2d(qp, r) / denom
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    def _point_in_convex_polygon(self, p: np.ndarray, poly: np.ndarray) -> bool:
        sign = 0
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            cross = self._cross_2d(b - a, p - a)
            if abs(cross) <= 1e-8:
                continue
            current_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = current_sign
            elif sign != current_sign:
                return False
        return True

    def _polygon_min_distance(self, poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        for i in range(len(poly_a)):
            a1 = poly_a[i]
            a2 = poly_a[(i + 1) % len(poly_a)]
            for j in range(len(poly_b)):
                b1 = poly_b[j]
                b2 = poly_b[(j + 1) % len(poly_b)]
                if self._segments_intersect(a1, a2, b1, b2):
                    return 0.0
        if self._point_in_convex_polygon(poly_a[0], poly_b) or self._point_in_convex_polygon(poly_b[0], poly_a):
            return 0.0
        best = float("inf")
        for point in poly_a:
            for j in range(len(poly_b)):
                best = min(best, self._point_to_segment_distance(point, poly_b[j], poly_b[(j + 1) % len(poly_b)]))
        for point in poly_b:
            for i in range(len(poly_a)):
                best = min(best, self._point_to_segment_distance(point, poly_a[i], poly_a[(i + 1) % len(poly_a)]))
        return float(best)

    def _box_min_distance(self, a: Dict, b: Dict) -> float:
        rect_a = self._rect(a["x"], a["y"], a.get("dx", 0.0), a.get("dy", 0.0), a.get("yaw", 0.0))
        rect_b = self._rect(b["x"], b["y"], b.get("dx", 0.0), b.get("dy", 0.0), b.get("yaw", 0.0))
        return self._polygon_min_distance(rect_a, rect_b)

    def _axis_interval_distance(self, value: float, lo: float, hi: float) -> float:
        if lo <= value < hi:
            return 0.0
        if value < lo:
            return lo - value
        return value - hi

    def _ped_crosswalk_region(self, obj: Dict) -> Tuple[str, Optional[str]]:
        x, y = obj["x"], obj["y"]
        for crosswalk_name, config in self.CROSSWALK_CONFIGS.items():
            if self._contains(x, y, config["crosswalk"]):
                return crosswalk_name, "crosswalk"
            for entry_name, _, rect in config["entries"]:
                if self._contains(x, y, rect):
                    return crosswalk_name, "entry zone"
            for _, rect in self.EXIT_REGION_NAMES[crosswalk_name]:
                if self._contains(x, y, rect):
                    return crosswalk_name, "exit zone"
        for label, rect in self.WAITING_ZONES.items():
            if self._contains(x, y, rect) and not self._in_any_crosswalk_related_zone(x, y):
                return self._corner_primary_crosswalk(label), "waiting zone"
        return "none", "none"

    def _in_any_crosswalk_related_zone(self, x: float, y: float) -> bool:
        for config in self.CROSSWALK_CONFIGS.values():
            if self._contains(x, y, config["crosswalk"]):
                return True
            for _, _, rect in config["entries"]:
                if self._contains(x, y, rect):
                    return True
        for items in self.EXIT_REGION_NAMES.values():
            for _, rect in items:
                if self._contains(x, y, rect):
                    return True
        return False

    def _corner_primary_crosswalk(self, corner: str) -> str:
        mapping = {
            "north-west corner": "north",
            "north-east corner": "north",
            "south-west corner": "south",
            "south-east corner": "south",
        }
        return mapping[corner]

    def _nearest_waiting_corner(self, obj: Dict) -> str:
        x, y = obj["x"], obj["y"]
        best_label = "north-west corner"
        best_dist = float("inf")
        for label, rect in self.WAITING_ZONES.items():
            dist = self._point_to_rect_distance(x, y, rect)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def _nearest_crosswalk_entry(self, obj: Dict) -> Optional[Dict]:
        x, y = obj["x"], obj["y"]
        best = None
        best_dist = float("inf")
        for crosswalk_name, config in self.CROSSWALK_CONFIGS.items():
            axis = config["axis"]
            lo, hi = (config["crosswalk"][2], config["crosswalk"][3]) if axis == "y" else (config["crosswalk"][0], config["crosswalk"][1])
            coord = y if axis == "y" else x
            for entry_name, end, rect in config["entries"]:
                if self._contains(x, y, rect):
                    dist = 0.0
                elif self._contains(x, y, config["crosswalk"]):
                    dist = coord - lo if end == "low" else hi - coord
                else:
                    dist = self._axis_interval_distance(coord, rect[2], rect[3]) if axis == "y" else self._axis_interval_distance(coord, rect[0], rect[1])
                if dist < best_dist:
                    best_dist = dist
                    best = {
                        "crosswalk": crosswalk_name,
                        "axis": axis,
                        "end": end,
                        "entry_name": entry_name,
                        "distance": round(float(dist), 1),
                    }
        return best

    def _ped_crosswalk_exit_distance(self, obj: Dict, crosswalk_name: str) -> Optional[float]:
        config = self.CROSSWALK_CONFIGS[crosswalk_name]
        axis = config["axis"]
        future = self._future_snapshot(obj, seconds=3.0)
        if future is not None:
            delta = (future["y"] - obj["y"]) if axis == "y" else (future["x"] - obj["x"])
        else:
            delta = obj["vy"] if axis == "y" else obj["vx"]
        if abs(delta) < 1e-3:
            return None
        target_end = "high" if delta > 0 else "low"
        exit_rects = self.EXIT_REGION_NAMES[crosswalk_name]
        if axis == "y":
            target_rect = max(exit_rects, key=lambda item: item[1][2] + item[1][3]) if target_end == "high" else min(exit_rects, key=lambda item: item[1][2] + item[1][3])
            coord = obj["y"]
            lo, hi = target_rect[1][2], target_rect[1][3]
        else:
            target_rect = max(exit_rects, key=lambda item: item[1][0] + item[1][1]) if target_end == "high" else min(exit_rects, key=lambda item: item[1][0] + item[1][1])
            coord = obj["x"]
            lo, hi = target_rect[1][0], target_rect[1][1]
        return round(self._axis_interval_distance(coord, lo, hi), 1)

    def _distance_to_stopline(self, obj: Dict) -> Optional[float]:
        side = obj["side"]
        rect = self._rect(obj["x"], obj["y"], obj.get("dx", 0.0), obj.get("dy", 0.0), obj.get("yaw", 0.0))
        min_x = float(np.min(rect[:, 0]))
        max_x = float(np.max(rect[:, 0]))
        min_y = float(np.min(rect[:, 1]))
        max_y = float(np.max(rect[:, 1]))
        if side == "north":
            return round(1.0 - max_x, 1) if max_x < 1.0 else None
        if side == "south":
            return round(min_x - 35.2, 1) if min_x > 35.2 else None
        if side == "west":
            return round(-4.4 - max_y, 1) if max_y < -4.4 else None
        if side == "east":
            return round(min_y - 31.8, 1) if min_y > 31.8 else None
        return None

    def _front_stopline_vehicle(self, side: str, objects: List[Dict]) -> Optional[Dict]:
        candidates = [obj for obj in objects if obj["side"] == side and obj["type"] != "pedestrian"]
        valid = [(self._distance_to_stopline(obj), obj) for obj in candidates]
        valid = [(dist, obj) for dist, obj in valid if dist is not None]
        if not valid:
            return None
        return min(valid, key=lambda item: item[0])[1]

    def _lane_queue_count(self, side: str, lane: str, objects: List[Dict]) -> int:
        lane_objs = [obj for obj in objects if obj["side"] == side and obj["lane"] == lane and obj["type"] not in self.VRU_TYPES]
        metrics = self._lane_queue_metrics(lane_objs)
        return int(metrics["queue_size"])

    def _stopline_back_5m_count(self, side: str, objects: List[Dict]) -> int:
        count = 0
        for obj in objects:
            if obj["type"] in self.VRU_TYPES:
                continue
            x, y = obj["x"], obj["y"]
            if side == "north" and -4.0 <= x < 1.0:
                count += 1
            elif side == "south" and 35.2 < x <= 40.2:
                count += 1
            elif side == "west" and -9.4 <= y < -4.4:
                count += 1
            elif side == "east" and 31.8 < y <= 36.8:
                count += 1
        return count

    def _vehicle_on_crosswalk(self, side: str, objects: List[Dict]) -> Optional[Dict]:
        rect = self.CROSSWALK_CONFIGS[side]["crosswalk"]
        for obj in objects:
            if obj["type"] in self.VRU_TYPES:
                continue
            if obj["status"] == "stopped" and self._contains(obj["x"], obj["y"], rect):
                return obj
        return None

    def _history_speed_delta(self, obj: Dict, seconds: float = 3.0) -> Optional[float]:
        history = self.history_cache.get((obj["frame_token"], obj["id"]), [])
        needed = int(round(seconds * self.SOURCE_SCENE_FPS))
        if len(history) < needed:
            return None
        previous = history[-needed]
        return round(self._speed(obj) - self._speed(previous), 3)

    def _future_snapshot(self, obj: Dict, seconds: float = 3.0) -> Optional[Dict]:
        future = self.future_track_cache.get((obj["frame_token"], obj["id"]), [])
        needed = int(round(seconds * self.SOURCE_SCENE_FPS))
        if len(future) < needed:
            return None
        return future[needed - 1]

    def _vehicle_accel_bucket(self, obj: Dict) -> Optional[str]:
        delta = self._history_speed_delta(obj, seconds=3.0)
        if delta is None:
            return None
        if delta >= 0.8:
            return "accelerating"
        if delta <= -0.8:
            return "decelerating"
        return None

    def _vehicle_acceleration_value(self, obj: Dict, seconds: float = 3.0) -> Optional[float]:
        delta = self._history_speed_delta(obj, seconds=seconds)
        if delta is None or seconds <= 0:
            return None
        return round(delta / seconds, 1)

    def _vehicle_motion_state_v5(self, obj: Dict) -> str:
        speed = self._speed(obj)
        delta = self._history_speed_delta(obj, seconds=3.0)
        if speed < 0.3:
            return "stopped"
        if delta is not None and 0.3 <= speed < 2.0 and delta >= 0.8:
            return "starting"
        if delta is not None and speed >= 0.3 and delta <= -0.8:
            return "braking"
        if 0.3 <= speed < 3.0:
            return "creeping"
        return "moving"

    def _pedestrian_motion_state_v5(self, obj: Dict) -> str:
        speed = self._speed(obj)
        if speed < 0.3:
            return "standing"
        if speed >= self.PEDESTRIAN_RUNNING_THRESHOLD:
            return "running"
        return "walking"

    def _current_motion_state_targets(self, obj: Dict, motion_state: str, speed_mps: float, accel_state: Optional[str] = None, acceleration: Optional[float] = None) -> Dict:
        return {
            **self._object_targets(obj, include_type=True, include_side=True),
            "motion_state": motion_state,
            "speed": speed_mps,
            "accel_state": accel_state,
            "acceleration": acceleration,
        }

    def _current_motion_state_answer(self, obj: Dict, motion_state: str, speed_mps: float, accel_state: Optional[str], acceleration: Optional[float]) -> str:
        base = f"{self._object_phrase(obj, capitalized=True)} is {motion_state} at {speed_mps:.1f} m/s"
        if obj["type"] != "pedestrian" and motion_state in {"starting", "braking"} and accel_state in {"accelerating", "decelerating"} and acceleration is not None:
            return f"{base} and {accel_state} at {acceleration:.1f} m/s^2."
        return f"{base}."

    def _vehicle_maneuver_v5(self, obj: Dict) -> str:
        status = obj["status"]
        mapping = {
            "turning-left": "left turn",
            "straight-going": "straight",
            "turning-right": "right turn",
            "lane-changing": "lane change",
            "stopped": "stop-and-wait",
        }
        if status == "speeding":
            future = self._future_snapshot(obj, seconds=3.0)
            if future is not None and future["side"] != obj["side"] and future["side"] in {"north", "south", "east", "west"}:
                turn_map = self._turn_map()
                origin = obj["side"]
                destination = future["side"]
                if destination in turn_map.get(origin, {}):
                    return {"straight-going": "straight", "turning-left": "left turn", "turning-right": "right turn"}[turn_map[origin][destination]]
            return "straight"
        return mapping.get(status, "straight")

    def _lead_vehicle_same_lane(self, obj: Dict, objects: List[Dict]) -> Optional[Dict]:
        if obj["lane"] is None or obj["side"] == "center":
            return None
        candidates = []
        for other in objects:
            if other["id"] == obj["id"] or other["side"] != obj["side"] or other["lane"] != obj["lane"] or other["type"] == "pedestrian":
                continue
            if self._center_dist(other) < self._center_dist(obj):
                candidates.append(other)
        if not candidates:
            return None
        return min(candidates, key=lambda other: self._dist(obj, other))

    def _following_distance_change(self, obj: Dict, lead: Dict) -> str:
        history_obj = self.history_cache.get((obj["frame_token"], obj["id"]), [])
        history_lead = self.history_cache.get((lead["frame_token"], lead["id"]), [])
        needed = int(round(3.0 * self.SOURCE_SCENE_FPS))
        if len(history_obj) < needed or len(history_lead) < needed:
            return "nearly unchanged"
        prev_obj = history_obj[-needed]
        prev_lead = history_lead[-needed]
        prev_dist = self._dist(prev_obj, prev_lead)
        now_dist = self._dist(obj, lead)
        delta = now_dist - prev_dist
        if delta >= 1.0:
            return "widening"
        if delta <= -1.0:
            return "shrinking"
        return "nearly unchanged"

    def _time_headway(self, follower: Dict, leader: Dict) -> float:
        gap = max(self._dist(follower, leader), 0.0)
        speed = max(self._speed(follower), 0.1)
        return gap / speed

    def _history_progress_to_center(self, obj: Dict, seconds: float = 3.0) -> Optional[float]:
        history = self.history_cache.get((obj["frame_token"], obj["id"]), [])
        needed = int(round(seconds * self.SOURCE_SCENE_FPS))
        if len(history) < needed:
            return None
        previous = history[-needed]
        return round(self._center_dist(previous) - self._center_dist(obj), 3)

    def _future_region_label(self, obj: Dict) -> str:
        future = self._future_snapshot(obj, seconds=3.0)
        if future is None:
            return "before stop line"
        if future["side"] == "center":
            return "intersection center"
        if obj["side"] == future["side"]:
            return "before stop line"
        turn_map = self._turn_map()
        if future["side"] in turn_map.get(obj["side"], {}):
            motion = turn_map[obj["side"]][future["side"]]
            return {
                "turning-left": "left-turn exit",
                "straight-going": "through exit",
                "turning-right": "right-turn exit",
            }.get(motion, "intersection center")
        return "intersection center"

    def _ped_will_enter_crosswalk(self, obj: Dict) -> bool:
        if self._ped_crosswalk_region(obj)[1] == "crosswalk":
            return False
        future = self._future_snapshot(obj, seconds=3.0)
        if future is None:
            return False
        for config in self.CROSSWALK_CONFIGS.values():
            if self._contains(future["x"], future["y"], config["crosswalk"]):
                return True
        return False

    def _strongest_conflict(self, objects: List[Dict]) -> Optional[Dict]:
        return self._near_miss(objects)

    def _notable_abnormal(self, objects: List[Dict]) -> Optional[Tuple[str, str, str]]:
        abnormal = self._abnormal_behavior_event(objects)
        if abnormal is not None:
            region = "center" if abnormal["region"] == "intersection" else abnormal["region"]
            if abnormal["kind"] == "speeding":
                return "speeding", region, f"A high-risk speeding event exists {self._region_exists_phrase(region)}."
            return "abnormal_proximity", region, f"A high-risk abnormal proximity interaction exists {self._region_exists_phrase(region)}."
        for side in ("north", "south", "east", "west"):
            if self._vehicle_on_crosswalk(side, objects):
                return "crosswalk_blocking", side, f"A crosswalk-blocking event exists on the {side} approach."
        lingering = [obj for obj in objects if obj["type"] == "pedestrian" and self._ped_crosswalk_region(obj)[1] == "waiting zone" and self._speed(obj) < 0.5]
        if lingering:
            return "lingering_pedestrian", lingering[0]["side"], f"A lingering-pedestrian event exists on the {lingering[0]['side']} approach."
        overrun = [obj for obj in objects if obj["type"] not in self.VRU_TYPES and self._distance_to_stopline(obj) is None and obj["side"] in {"north", "south", "east", "west"}]
        if overrun:
            return "stopline_overrun", overrun[0]["side"], f"A stop-line overrun event exists on the {overrun[0]['side']} approach."
        wrong_way = [obj for obj in objects if obj["type"] in {"bicycle", "motorcycle"} and self._is_wrong_way_two_wheeler(obj)]
        if wrong_way:
            return "wrong_way_two_wheeler", wrong_way[0]["side"], f"A wrong-way two-wheeler event exists on the {wrong_way[0]['side']} approach."
        if self._planning_scene_stats(objects)["L_q"] >= 2 and self._planning_scene_stats(objects)["O"] >= 0.25:
            return "queue_spillback", "center", "A queue-spillback event exists in the center area."
        return None

    def _region_exists_phrase(self, region: str) -> str:
        if region == "center":
            return "in the center area"
        return f"on the {region} approach"

    def _is_wrong_way_two_wheeler(self, obj: Dict) -> bool:
        if obj["side"] == "north":
            return obj.get("vx", 0.0) < -0.5
        if obj["side"] == "south":
            return obj.get("vx", 0.0) > 0.5
        if obj["side"] == "west":
            return obj.get("vy", 0.0) < -0.5
        if obj["side"] == "east":
            return obj.get("vy", 0.0) > 0.5
        return False

    def _risk_region(self, objects: List[Dict]) -> str:
        abnormal = self._abnormal_behavior_event(objects)
        if abnormal is not None:
            region = abnormal["region"]
            return "center" if region == "intersection" else region
        event = self._strongest_conflict(objects)
        if event is not None:
            if event["obj1"]["side"] == event["obj2"]["side"]:
                return event["obj1"]["side"]
            return "center"
        return "center"

    def _flow_imbalance(self, objects: List[Dict]) -> bool:
        by_side = defaultdict(int)
        for obj in objects:
            if obj["side"] in {"north", "south", "east", "west"}:
                by_side[obj["side"]] += 1
        total = sum(by_side.values())
        if total <= 0:
            return False
        return max(by_side.values(), default=0) / total > 0.40

    def _scene_summary(self, objects: List[Dict]) -> str:
        overview = self._format_intersection_overview(objects)
        abnormal = self._abnormal_behavior_event(objects)
        if abnormal is None:
            return f"{overview} No clear abnormal behavior is currently dominant."
        region = abnormal["region"]
        if region == "intersection":
            region_phrase = "around the intersection"
        elif region == "center":
            region_phrase = "in the center"
        else:
            region_phrase = f"on the {region} approach"
        return f"{overview} The most notable current risk is {abnormal['kind']} {region_phrase}."

    def _scene_summary_targets(self, objects: List[Dict]) -> Dict:
        overview_state = self._congestion(objects)
        abnormal = self._abnormal_behavior_event(objects)
        primary_focus = self._risk_region(objects)
        return {
            "summary_type": "scene_summary",
            "overall_state": overview_state,
            "primary_focus": primary_focus,
            "abnormal_or_none": "none" if abnormal is None else abnormal["kind"],
        }

    def _size_bucket(self, obj: Dict) -> str:
        volume = max(obj.get("dx", 0.1), 0.1) * max(obj.get("dy", 0.1), 0.1) * max(obj.get("dz", 0.1), 0.1)
        q1, q2, q3 = self.size_bins
        if volume < q1:
            return "small"
        if volume < q2:
            return "medium"
        if volume < q3:
            return "large"
        return "extra-large"

    def _enabled_templates(self) -> Iterable[V5TemplateSpec]:
        return (spec for spec in self.template_registry.values() if spec.enabled_by_default)

    def _disabled_templates(self) -> Iterable[V5TemplateSpec]:
        return (spec for spec in self.template_registry.values() if not spec.enabled_by_default)

    def _basic_perception_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        frame = self.token_to_frame.get(objects[0]["frame_token"]) if objects else None
        candidates = self._candidates(objects)
        for obj in candidates:
            qas.append(
                self._v5_qa(
                    "1_1_1_fine_type",
                    f"What is the type of {self._ref(obj)}?",
                    f"It is a {obj['type']} located {self._side_location_phrase(obj['side'])}.",
                    1.0,
                    self._object_targets(obj, include_type=True, include_side=True),
                    sample_meta={"object_type": obj["type"], "direction": obj["side"], "object_id": obj["ref_id"]},
                )
            )
            sector_best: Dict[str, Tuple[float, float, Dict]] = {}
            for other in objects:
                if other["id"] == obj["id"]:
                    continue
                center_distance = self._center_distance_between(obj, other)
                if center_distance > 4.0:
                    continue
                rel_dir, angle_delta = self._relative_direction_info(obj, other)
                previous = sector_best.get(rel_dir)
                if previous is None or (angle_delta, center_distance) < (previous[0], previous[1]):
                    sector_best[rel_dir] = (angle_delta, center_distance, other)
            for rel_dir, (_, _, other) in sorted(sector_best.items(), key=lambda item: list(self.RELATIVE_DIRECTION_QUERY_PHRASES).index(item[0])):
                qas.append(
                    self._v5_qa(
                        "1_1_4_relative_neighbor_type",
                        f"What is located {self.RELATIVE_DIRECTION_QUERY_PHRASES[rel_dir]} {self._ref(obj)}?",
                        f"It is a {other['type']} at {self._location_ref(other)}.",
                        0.8,
                        {"ref_id": obj["ref_id"], "ref_raw_tracking_id": obj["id"], "rel_dir": rel_dir, "object_type": other["type"], **self._prefixed_object_targets(other, "target_object", include_type=False)},
                        sample_meta={"rel_dir": rel_dir, "object_id": obj["ref_id"], "direction": obj["side"]},
                    )
                )
            size_bucket = self._size_bucket(obj)
            qas.append(
                self._v5_qa(
                    "1_2_1_size_bucket",
                    f"Which size best matches {self._ref(obj)}?",
                    f"The {obj['type']} {self._side_location_phrase(obj['side'])} is classified as a {size_bucket}-size vehicle.",
                    0.5,
                    {**self._object_targets(obj, include_type=True, include_side=True), "size_bucket": size_bucket},
                    sample_meta={"size_bucket": size_bucket, "object_id": obj["ref_id"]},
                )
            )
        for side in ("north", "south", "east", "west"):
            side_objs = [obj for obj in objects if obj["side"] == side]
            counts = defaultdict(int)
            for obj in side_objs:
                counts[obj["type"]] += 1
            vru_count = sum(counts.get(obj_type, 0) for obj_type in self.VRU_TYPES)
            exists = vru_count > 0
            exists_answer = (
                f"Yes, there {'is' if vru_count == 1 else 'are'} {vru_count} VRU{'s' if vru_count != 1 else ''}."
                if exists
                else "No."
            )
            qas.append(
                self._v5_qa(
                    "1_1_2_side_exists",
                    f"Does the {side} approach contain any vulnerable road user?",
                    exists_answer,
                    float(vru_count) if exists else 0.1,
                    {"side": side, "exists": exists, "count": int(vru_count)},
                    answer_bucket="yes" if exists else "no",
                    sample_meta={"direction": side, "exists": exists, "count": int(vru_count)},
                )
            )
            for obj_type in self.FINE_TYPES:
                count_value = int(counts.get(obj_type, 0))
                if count_value > 0:
                    qas.append(
                        self._v5_qa(
                            "1_1_3_side_count",
                            f"How many {obj_type}s are currently on the {side} approach?",
                            f"The {side} approach currently has {count_value} {self._pluralize(obj_type, count_value)}.",
                            float(count_value),
                            {"side": side, "object_type": obj_type, "count": count_value},
                            sample_meta={"direction": side, "object_type": obj_type},
                        )
                    )
        qas.extend(self._annotated_or_placeholder_basic_perception_qas(frame, objects))
        return qas

    def _frame_manual_annotations(self, frame: Optional[Dict]) -> Dict:
        if not frame:
            return {}
        value = frame.get("v5_manual_annotations")
        return value if isinstance(value, dict) else {}

    def _visibility_answer(self, obj: Dict, visibility: str) -> str:
        return f"The {obj['type']} on the {obj['side']} approach is {visibility}."

    def _join_with_and(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _normalized_sun_glare_sides(self, light: Dict) -> List[str]:
        raw = light.get("sun_glare_sides") if isinstance(light, dict) else None
        if not isinstance(raw, list):
            return []
        seen = set()
        sides: List[str] = []
        for side in raw:
            side_str = str(side).strip().lower()
            if side_str in self.SUN_GLARE_SIDE_LABELS and side_str not in seen:
                sides.append(side_str)
                seen.add(side_str)
        ordered = [side for side in self.SUN_GLARE_SIDE_LABELS if side in seen]
        return ordered if ordered else sides

    def _environment_targets(self, weather: Optional[str], time_of_day: Optional[str], sun_glare: bool, sun_glare_sides: List[str]) -> Dict:
        return {
            "weather": weather,
            "time_of_day": time_of_day,
            "light": {
                "sun_glare": bool(sun_glare),
                "sun_glare_sides": list(sun_glare_sides),
            },
        }

    def _environment_signature(self, weather: Optional[str], time_of_day: Optional[str], sun_glare: bool, sun_glare_sides: List[str]) -> str:
        parts = [
            f"weather:{weather or 'none'}",
            f"time:{time_of_day or 'none'}",
            f"sun_glare:{'yes' if sun_glare else 'no'}",
        ]
        if sun_glare_sides:
            parts.append("glare_views:" + "+".join(sun_glare_sides))
        return "|".join(parts)

    def _environment_answer(self, weather: Optional[str], time_of_day: Optional[str], sun_glare: bool, sun_glare_sides: List[str]) -> Optional[str]:
        base_parts: List[str] = []
        if weather:
            base_parts.append(weather)
        if time_of_day:
            base_parts.append(time_of_day)
        glare_phrase = None
        if sun_glare:
            if sun_glare_sides:
                view_phrase = self._join_with_and(list(sun_glare_sides))
                view_word = "view" if len(sun_glare_sides) == 1 else "views"
                glare_phrase = f"strong sun glare in the {view_phrase} {view_word}"
            else:
                glare_phrase = "strong sun glare"
        if base_parts:
            sentence = "The scene appears to be " + ", ".join(base_parts)
            if glare_phrase:
                sentence += f", with {glare_phrase}"
            return sentence + "."
        if glare_phrase:
            return f"The scene appears to have {glare_phrase}."
        return None

    def _valid_vehicle_signal_state(self, side: str, state: Optional[str]) -> Optional[str]:
        if not isinstance(state, str):
            return None
        normalized = re.sub(r"\s+", " ", state.strip().lower())
        if side in {"north", "south"}:
            valid = set(self.VEHICLE_SIGNAL_LIGHT_STATES)
        else:
            valid = set(self.VEHICLE_SIGNAL_LIGHT_STATES) | set(self.VEHICLE_SIGNAL_ARROW_STATES)
        return normalized if normalized in valid else None

    def _signal_answer(self, state: str) -> str:
        return f"The signal is currently {state}."

    def _annotated_or_placeholder_basic_perception_qas(self, frame: Optional[Dict], objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        annotations = self._frame_manual_annotations(frame)
        visibility_map = annotations.get("visibility_by_track_id") if isinstance(annotations.get("visibility_by_track_id"), dict) else {}
        for obj in self._candidates(objects):
            if obj["side"] not in {"north", "south", "east", "west"}:
                continue
            visibility = visibility_map.get(str(obj["id"]))
            question = f"How visible is {self._ref(obj)}?"
            if visibility:
                qas.append(
                    self._v5_qa(
                        "1_2_2_visibility",
                        question,
                        self._visibility_answer(obj, visibility),
                        0.2,
                        {**self._object_targets(obj, include_type=True, include_side=True), "visibility": visibility},
                        sample_meta={"visibility": visibility, "object_id": obj["ref_id"], "direction": obj["side"]},
                    )
                )
            else:
                qas.append(
                    self._v5_qa(
                        "1_2_2_visibility",
                        question,
                        None,
                        0.0,
                        {**self._object_targets(obj, include_type=True, include_side=True), "visibility": None},
                        sample_meta={"visibility": "unknown", "object_id": obj["ref_id"], "direction": obj["side"]},
                        placeholder=True,
                    )
                )

        weather_value = annotations.get("weather")
        weather = str(weather_value).strip().lower() if isinstance(weather_value, str) and str(weather_value).strip() else None
        time_value = annotations.get("time_of_day")
        time_of_day = str(time_value).strip().lower() if isinstance(time_value, str) and str(time_value).strip() else None
        if time_of_day not in self.TIME_OF_DAY_LABELS:
            time_of_day = None
        light = annotations.get("light") if isinstance(annotations.get("light"), dict) else {}
        sun_glare_sides = self._normalized_sun_glare_sides(light)
        sun_glare = bool(light.get("sun_glare")) or bool(sun_glare_sides)
        environment_answer = self._environment_answer(weather, time_of_day, sun_glare, sun_glare_sides)
        environment_targets = self._environment_targets(weather, time_of_day, sun_glare, sun_glare_sides)
        environment_signature = self._environment_signature(weather, time_of_day, sun_glare, sun_glare_sides)
        if environment_answer is not None:
            qas.append(
                self._v5_qa(
                    "1_3_1_weather",
                    "Please describe the current environmental conditions in the scene.",
                    environment_answer,
                    0.1,
                    environment_targets,
                    sample_meta={
                        "environment": environment_signature,
                        "weather": weather or "unknown",
                        "time_of_day": time_of_day or "unknown",
                        "sun_glare": "yes" if sun_glare else "no",
                    },
                )
            )
        else:
            qas.append(
                self._v5_qa(
                    "1_3_1_weather",
                    "Please describe the current environmental conditions in the scene.",
                    None,
                    0.0,
                    environment_targets,
                    sample_meta={
                        "environment": "unknown",
                        "weather": "unknown",
                        "time_of_day": "unknown",
                        "sun_glare": "unknown",
                    },
                    placeholder=True,
                )
            )

        vehicle_signal_state = annotations.get("vehicle_signal_state") if isinstance(annotations.get("vehicle_signal_state"), dict) else {}
        for side in ("north", "south", "east", "west"):
            for movement in ("left-turn", "through"):
                state = None
                if isinstance(vehicle_signal_state.get(side), dict):
                    state = self._valid_vehicle_signal_state(side, vehicle_signal_state.get(side, {}).get(movement))
                if state:
                    qas.append(self._v5_qa("1_3_2_vehicle_signal_state", f"Please describe the signal state for {movement} traffic on the {side} approach.", self._signal_answer(state), 0.1, {"side": side, "movement": movement, "signal_state": state}, sample_meta={"direction": side, "signal_state": state}))
                else:
                    qas.append(self._v5_qa("1_3_2_vehicle_signal_state", f"Please describe the signal state for {movement} traffic on the {side} approach.", None, 0.0, {"side": side, "movement": movement, "signal_state": None}, sample_meta={"direction": side, "signal_state": f"{movement}_unknown"}, placeholder=True))
        return qas

    def _spatial_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        by_side, by_lane = self._split_by_side_and_lane(objects)
        for side in ("north", "south", "east", "west"):
            vehicle = self._front_stopline_vehicle(side, objects)
            if vehicle is not None:
                distance = self._distance_to_stopline(vehicle)
                if distance is not None:
                    qas.append(self._v5_qa("2_1_1_stopline_distance", f"How far is {self._ref(vehicle)} from its relevant stop line?", f"The {vehicle['type']} is {distance:.1f} m from the stop line on the {side} approach.", 10.0 - distance, {**self._object_targets(vehicle, include_type=True), "distance_m": distance, "stopline_side": side}, sample_meta={"object_type": vehicle["type"], "direction": side, "object_id": vehicle["ref_id"]}))
        ped_candidates = [obj for obj in self._candidates(objects) if obj["type"] == "pedestrian"]
        for ped in ped_candidates:
            crosswalk_name, region_label = self._ped_crosswalk_region(ped)
            if region_label in {"crosswalk", "entry zone"}:
                exit_distance = self._ped_crosswalk_exit_distance(ped, crosswalk_name)
                if exit_distance is not None:
                    qas.append(self._v5_qa("2_1_2_ped_to_far_edge", f"How far is {self._ref(ped)} from the crossing exit area?", f"The pedestrian on the {crosswalk_name} crosswalk is {exit_distance:.1f} m from the exit area.", 5.0, {"pedestrian_id": ped["ref_id"], "pedestrian_raw_tracking_id": ped["id"], "crosswalk": crosswalk_name, "ped_zone": "within the crosswalk" if region_label == "crosswalk" else region_label, "distance_m": exit_distance}, sample_meta={"crosswalk": crosswalk_name, "region": region_label, "object_id": ped["ref_id"]}))
            nearest_vehicle = self._nearest(ped, [obj for obj in objects if obj["type"] != "pedestrian"])
            if nearest_vehicle is not None:
                dist = round(self._dist(ped, nearest_vehicle), 1)
                qas.append(self._v5_qa("2_1_4_nearest_vehicle_to_ped", f"Please identify the vehicle nearest to {self._ref(ped)}.", f"It is a {nearest_vehicle['type']} on the {nearest_vehicle['side']} approach, {dist:.1f} m away.", 10.0 - dist, {"pedestrian_id": ped["ref_id"], "pedestrian_raw_tracking_id": ped["id"], "vehicle_id": nearest_vehicle["ref_id"], "vehicle_raw_tracking_id": nearest_vehicle["id"], "vehicle_type": nearest_vehicle["type"], "side": nearest_vehicle["side"], "distance_m": dist, "direction": ped["side"]}, sample_meta={"direction": ped["side"], "object_id": ped["ref_id"]}))
        pair = self._closest_participant_pair(objects)
        if pair is not None:
            a, b, dist = pair
            if {a["type"], b["type"]} != {"truck", "trailer"}:
                pair_type = f"{a['type']}-{b['type']}" if a["type"] <= b["type"] else f"{b['type']}-{a['type']}"
                qas.append(self._v5_qa("2_1_3_participant_distance", f"What is the distance between {self._ref(a)} and {self._ref(b)}?", f"{self._object_phrase(a, capitalized=True)} is {dist:.1f} m from {self._object_phrase(b)}.", 10.0 - dist, {"obj1_id": a["ref_id"], "obj1_raw_tracking_id": a["id"], "obj2_id": b["ref_id"], "obj2_raw_tracking_id": b["id"], "obj1_type": a["type"], "obj2_type": b["type"], "distance_m": dist, "pair_type": pair_type}, sample_meta={"pair_type": pair_type}))
        for obj in self._candidates(objects):
            if obj["type"] != "pedestrian" and obj["side"] in {"north", "south", "east", "west"} and obj["lane"] is not None:
                lane_label = self._lane_function_label(obj["lane"])
                qas.append(self._v5_qa("2_2_1_lane_function", f"Which lane does {self._ref(obj)} currently occupy?", f"{self._object_phrase(obj, capitalized=True)} is currently in the {lane_label}.", 1.0, {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane_label}, sample_meta={"lane_function": lane_label, "object_type": obj["type"], "direction": obj["side"], "object_id": obj["ref_id"]}))
        for ped in ped_candidates:
            crosswalk_name, region = self._ped_crosswalk_region(ped)
            if region != "none":
                ped_zone_value = "within the crosswalk" if region == "crosswalk" else region
                qas.append(self._v5_qa("2_2_2_ped_zone", f"Which area of the crosswalk is {self._ref(ped)} currently in?", f"The pedestrian is currently {self._ped_zone_phrase(crosswalk_name, region)}.", 1.0, {"pedestrian_id": ped["ref_id"], "pedestrian_raw_tracking_id": ped["id"], "crosswalk": crosswalk_name, "ped_zone": ped_zone_value, **self._object_location_targets(ped, "pedestrian")}, sample_meta={"zone": region, "object_id": ped["ref_id"]}))
        for side in ("north", "south", "east", "west"):
            for lane in (self.LEFT_TURN_LANE, self.STRAIGHT_LANE, self.RIGHT_TURN_LANE):
                lane_label = self._lane_function_label(lane)
                queue_count = self._lane_queue_count(side, lane, objects)
                if queue_count > 0:
                    qas.append(
                        self._v5_qa(
                            "2_2_3_left_turn_queue_count",
                            f"How many queued vehicles are currently in the {lane_label} on the {side} approach?",
                            f"There are {queue_count} queued {self._pluralize('vehicle', queue_count)} in the {lane_label} on the {side} approach.",
                            float(queue_count),
                            {"side": side, "count": queue_count, "queue_vehicle_count": queue_count, "lane_function": lane_label},
                            sample_meta={"direction": side, "lane_function": lane_label},
                        )
                    )
            count_5m = self._stopline_back_5m_count(side, objects)
            if count_5m > 0:
                qas.append(self._v5_qa("2_2_4_stopline_back_5m_count", f"How many vehicles are within 5 meters behind the stop line on the {side} approach?", f"There are {count_5m} vehicles within 5 m behind the stop line on the {side} approach.", float(count_5m), {"side": side, "count": count_5m, "vehicle_count": count_5m}, sample_meta={"direction": side}))
            blocking_vehicle = self._vehicle_on_crosswalk(side, objects)
            blocked = blocking_vehicle is not None
            blocking_answer = (
                f"Yes, a {blocking_vehicle['type']} is currently blocking the crosswalk on the {side} approach."
                if blocked
                else f"No, no vehicle is currently blocking the crosswalk on the {side} approach."
            )
            qas.append(self._v5_qa("2_2_6_crosswalk_blocking", "Is any vehicle blocking the crosswalk?", blocking_answer, 1.0 if blocked else 0.1, {"side": side, "crosswalk_blocked": blocked, "object_type": None if blocking_vehicle is None else blocking_vehicle["type"], "blocking_vehicle_id": None if blocking_vehicle is None else blocking_vehicle["ref_id"], "blocking_vehicle_raw_tracking_id": None if blocking_vehicle is None else blocking_vehicle["id"]}, sample_meta={"direction": side, "crosswalk_blocked": blocked, "object_type": None if blocking_vehicle is None else blocking_vehicle["type"]}))
        global_side, global_lane = self._global_longest_queue_lane(by_lane)
        if global_side is not None and global_lane is not None:
            lane_label = self._lane_function_label(global_lane)
            qas.append(
                self._v5_qa(
                    "2_2_5_longest_queue_lane",
                    "Which lane currently has the longest queue right now?",
                    f"The {lane_label} on the {global_side} approach currently has the longest queue.",
                    1.0,
                    {"lane_function": lane_label, "side": global_side},
                    sample_meta={"direction": global_side, "lane_function": lane_label},
                )
            )
        return qas

    def _temporal_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        candidates = self._candidates(objects)
        vehicle_candidates = [obj for obj in candidates if obj["type"] != "pedestrian"]
        for obj in candidates:
            motion_state = self._pedestrian_motion_state_v5(obj) if obj["type"] == "pedestrian" else self._vehicle_motion_state_v5(obj)
            speed_mps = round(self._speed(obj), 1)
            accel_state = None
            acceleration = None
            if obj["type"] != "pedestrian" and motion_state in {"starting", "braking"}:
                accel_state = self._vehicle_accel_bucket(obj)
                acceleration = self._vehicle_acceleration_value(obj, seconds=3.0)
                if accel_state not in {"accelerating", "decelerating"} or acceleration is None:
                    accel_state = None
                    acceleration = None
            qas.append(self._v5_qa("3_1_1_current_motion_state", f"Please describe the current motion state of {self._ref(obj)}.", self._current_motion_state_answer(obj, motion_state, speed_mps, accel_state, acceleration), 1.0, self._current_motion_state_targets(obj, motion_state, speed_mps, accel_state, acceleration), sample_meta={"motion_state": motion_state, "object_id": obj["ref_id"], "direction": obj["side"]}))
        for obj in vehicle_candidates:
            maneuver = self._vehicle_maneuver_v5(obj)
            maneuver_phrase = {
                "left turn": "making a left turn",
                "straight": "going straight",
                "right turn": "making a right turn",
                "lane change": "changing lanes",
                "stop-and-wait": "stopping and waiting",
            }.get(maneuver, maneuver)
            qas.append(self._v5_qa("3_1_2_vehicle_maneuver", f"What maneuver is {self._ref(obj)} most likely executing?", f"{self._object_phrase(obj, capitalized=True)} is {maneuver_phrase}.", 1.0, {**self._object_targets(obj, include_type=True, include_side=True), "maneuver": maneuver}, sample_meta={"maneuver": maneuver, "object_id": obj["ref_id"], "direction": obj["side"]}))
            future_region = self._future_region_label(obj)
            qas.append(self._v5_qa("3_2_2_future_region", f"Which region is {self._ref(obj)} most likely to enter within the next 3 seconds?", f"{self._object_phrase(obj, capitalized=True)} is likely to {self._future_region_phrase(future_region)}.", 1.0, {"object_id": obj["ref_id"], "object_type": obj["type"], "future_region": future_region}, sample_meta={"future_region": future_region, "object_id": obj["ref_id"], "direction": obj["side"]}))
        straight_pairs = self._straight_following_pairs(objects)
        if straight_pairs:
            follower, leader = straight_pairs[0]
            time_headway = self._time_headway(follower, leader)
            safe = time_headway >= 2.0
            pair_type = f"{follower['type']}-{leader['type']}" if follower["type"] <= leader["type"] else f"{leader['type']}-{follower['type']}"
            pair_distance = round(self._dist(follower, leader), 1)
            safe_answer = (
                f"Yes, the following distance is currently safe, about {pair_distance} m."
                if safe
                else f"No, the following distance is currently too short, about {pair_distance} m."
            )
            qas.append(self._v5_qa("3_3_1_safe_following", f"Are straight-moving {self._ref(follower)} and {self._ref(leader)} maintaining a safe following gap?", safe_answer, 1.0 if safe else 0.1, {"follower_id": follower["ref_id"], "follower_raw_tracking_id": follower["id"], "leader_id": leader["ref_id"], "leader_raw_tracking_id": leader["id"], "distance_m": pair_distance, "time_headway_sec": round(time_headway, 2), "is_safe": safe, "pair_type": pair_type}, sample_meta={"pair_type": pair_type, "is_safe": safe}))
        by_side, by_lane = self._split_by_side_and_lane(objects)
        lane_candidates = []
        for side in ("north", "south", "east", "west"):
            lane_map = {lane: lane_objs for (lane_side, lane), lane_objs in by_lane.items() if lane_side == side}
            if not lane_map:
                continue
            answer, lane_bucket, priority = self._longest_queue_lane_answer(side, lane_map, predictive=True)
            if answer is not None and lane_bucket is not None:
                lane_candidates.append((priority, side, lane_bucket, answer, lane_map[lane_bucket]))
        if lane_candidates:
            _, side, lane_bucket, answer, lane_objs = max(lane_candidates, key=lambda item: item[0])
            lane_label = self._lane_function_label(lane_bucket)
            qas.append(self._v5_qa("3_3_2_likely_long_queue_lane", "Which lane is most likely to form a long queue soon?", self._v5_queue_prediction_answer(side, lane_bucket, lane_objs), 1.0, {"lane_function": lane_label, "side": side, "queue_evidence": self._queue_prediction_targets(lane_bucket, lane_objs)}, sample_meta={"lane_function": lane_label, "direction": side}))
        for obj in candidates:
            waypoints = self._trajectory_waypoints(obj)
            if waypoints is not None:
                qas.append(self._v5_qa("3_2_3_waypoints", f"Please predict the short-term future trajectory for {self._ref(obj)}.", self._format_v5_waypoints(waypoints), 1.0, {**self._object_targets(obj, include_type=True), **self._trajectory_targets(waypoints)}, sample_meta={"object_id": obj["ref_id"], "direction": obj["side"]}))
        return qas

    def _interaction_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        vrus = [obj for obj in objects if obj["type"] in self.VRU_TYPES]
        vehicles = [obj for obj in objects if obj["type"] != "pedestrian"]
        for vru in vrus[:4]:
            nearest_vehicle = self._nearest(vru, vehicles)
            if nearest_vehicle is None:
                continue
            event = self._future_pair_conflict(vru, nearest_vehicle, self.NEAR_MISS_DIST)
            has_conflict = event is not None
            pair_type = f"{nearest_vehicle['type']}-{vru['type']}" if nearest_vehicle["type"] <= vru["type"] else f"{vru['type']}-{nearest_vehicle['type']}"
            conflict_answer = (
                f"Yes, {self._object_phrase(nearest_vehicle)} and {self._object_phrase(vru)} currently show a potential conflict."
                if has_conflict
                else "No."
            )
            qas.append(self._v5_qa("3_4_1_vehicle_ped_conflict", f"Is there a potential conflict between {self._ref(nearest_vehicle)} and {self._ref(vru)}?", conflict_answer, 1.0 if has_conflict else 0.1, {"vehicle_id": nearest_vehicle["ref_id"], "vehicle_raw_tracking_id": nearest_vehicle["id"], "vru_id": vru["ref_id"], "vru_raw_tracking_id": vru["id"], "pair_type": pair_type, "has_conflict": has_conflict}, sample_meta={"pair_type": pair_type, "has_conflict": has_conflict}))
        event = self._strongest_conflict(objects)
        if event is not None:
            focus = event["obj1"]
            conflict_candidates = []
            for other in objects:
                if other["id"] == focus["id"]:
                    continue
                if self._future_pair_conflict(focus, other, self.NEAR_MISS_DIST) is None:
                    continue
                conflict_candidates.append((self._dist(focus, other), other))
            if conflict_candidates:
                _, other = min(conflict_candidates, key=lambda item: (item[0], item[1]["id"]))
                qas.append(
                    self._v5_qa(
                        "3_4_2_nearest_conflict_participant",
                        f"Which participant is most likely to conflict with {self._ref(focus)}?",
                        f"The most probable participant is the {other['type']} at {self._location_ref(other)}.",
                        1.0,
                        {
                            "focus_id": focus["ref_id"],
                            "focus_raw_tracking_id": focus["id"],
                            "focus_position": {
                                "x": round(float(focus["x"]), 1),
                                "y": round(float(focus["y"]), 1),
                            },
                            "conflict_partner_id": other["ref_id"],
                            "conflict_partner_raw_tracking_id": other["id"],
                            "conflict_partner_type": other["type"],
                            **self._object_location_targets(other, "conflict_partner"),
                        },
                        sample_meta={"object_id": focus["ref_id"], "direction": focus["side"]},
                    )
                )
            pattern = self._interaction_pattern_label(event)
            if pattern != "other":
                pattern_side = event["obj1"]["side"]
                side_phrase = f"on the {pattern_side} approach" if pattern_side != "center" else "in the center area"
                qas.append(self._v5_qa("3_4_4_risk_pattern", "What is the dominant conflict interaction pattern in the current scene?", f"The dominant conflict pattern is {pattern} {side_phrase}.", 1.0, {"interaction_pattern": pattern, "side": pattern_side}, sample_meta={"interaction_pattern": pattern}))
        subject, reason = self._primary_risk_subject(objects)
        if subject is not None:
            qas.append(
                self._v5_qa(
                    "3_4_3_primary_risk_subject",
                    "Please identify the key participant associated with the major potential safety hazards.",
                    f"The primary risk subject is the {subject['type']} at {self._location_ref(subject)} because of {self._turn_reason_phrase(reason)}.",
                    1.0,
                    {
                        "subject_id": subject["ref_id"],
                        "subject_raw_tracking_id": subject["id"],
                        "subject_type": subject["type"],
                        "risk_reason": reason,
                        **self._object_location_targets(subject, "subject"),
                    },
                    sample_meta={"risk_reason": reason, "object_id": subject["ref_id"]},
                )
            )
        return qas

    def _scene_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        level = self._congestion(objects)
        vehicles = self._vehicle_objects(objects)
        moving_vehicles = self._moving_vehicle_count(vehicles)
        total_vehicles = len(vehicles)
        qas.append(self._v5_qa("4_1_1_overall_state", "What is the overall traffic condition of the intersection?", f"The intersection is currently {level}, with {moving_vehicles} of {total_vehicles} vehicles moving.", 1.0, {"overall_state": level, "moving_vehicles": moving_vehicles, "total_vehicles": total_vehicles}, sample_meta={"overall_state": level}))
        by_side, by_lane = self._split_by_side_and_lane(objects)
        ext_sides = {side: side_objs for side, side_objs in by_side.items() if side in {"north", "south", "east", "west"}}
        for side in ("east", "north", "south", "west"):
            side_objs = by_side.get(side, [])
            targets = self._side_status_targets(side_objs)
            targets["side"] = side
            qas.append(self._v5_qa("4_1_2_side_motion_status", f"Please describe the motion status of traffic participants on the {side} approach of the intersection.", self._side_status_answer(side, side_objs), 1.0, targets, sample_meta={"direction": side}))
        qas.append(self._v5_qa("4_1_3_scene_summary", "Provide a brief summary of the current intersection scene.", self._scene_summary(objects), 1.0, self._scene_summary_targets(objects)))
        side_counts = {side: len(by_side.get(side, [])) for side in ("north", "south", "east", "west")}
        side_tie_priority = {"north": 0, "south": 1, "east": 2, "west": 3}
        dominant_side = min(side_counts, key=lambda side: (-side_counts[side], side_tie_priority[side])) if side_counts else None
        participant_count = side_counts.get(dominant_side, 0) if dominant_side is not None else 0
        busiest_answer = f"The {dominant_side} approach is the busiest, with {participant_count} traffic participants." if dominant_side is not None else "The north approach is the busiest, with 0 traffic participants."
        qas.append(self._v5_qa("4_1_4_flow_imbalance", "Which approach to the intersection has the heaviest traffic?", busiest_answer, float(participant_count), {"dominant_side": dominant_side, "participant_count": participant_count}, sample_meta={"dominant_side": dominant_side, "participant_count": participant_count}))

        fastest = max([obj for obj in objects if obj["type"] not in self.VRU_TYPES], key=self._speed, default=None)
        if fastest is not None and self._speed(fastest) >= self.VEHICLE_OVERSPEED_THRESHOLD - self.SIDE_SPEEDING_MARGIN:
            answer = f"Yes, a {fastest['type']} {self._side_location_phrase(fastest['side'])} is still moving at about {self._speed(fastest):.1f} m/s."
        else:
            answer = "No."
        speeding_targets = self._side_speeding_targets(fastest if answer.startswith("Yes") else None)
        if fastest is not None and answer.startswith("Yes"):
            speeding_targets["vehicle_id"] = fastest["ref_id"]
            speeding_targets["side"] = fastest["side"]
            speeding_targets["risk_region"] = fastest["side"]
            speeding_targets["evidence"] = f"A {fastest['type']} is still moving at about {self._speed(fastest):.1f} m/s."
        qas.append(self._v5_qa("4_2_1_speeding_risk", "Is there still a risk of speeding at the intersection?", answer, 1.0 if answer.startswith("Yes") else 0.1, speeding_targets, sample_meta={"direction": fastest["side"] if fastest is not None and answer.startswith("Yes") else "none", "has_speeding_risk": answer.startswith("Yes")}))
        notable = self._notable_abnormal(objects)
        notable_label = None
        if notable is not None:
            notable_label, notable_region, notable_reason = notable
            qas.append(self._v5_qa("4_2_2_notable_abnormal", "What is the most notable current abnormal event in the scene?", notable_reason, 1.0, {"notable_abnormal": notable_label, "risk_region": notable_region, "reason": notable_reason, "evidence": notable_reason}, sample_meta={"notable_abnormal": notable_label}))

        intersection_answer, intersection_bucket = self._intersection_safe_action_answer(objects)
        qas.append(self._v5_qa("4_3_1_intersection_action", "How should traffic proceed through the intersection right now?", intersection_answer, 1.0, self._action_state_targets(intersection_bucket), sample_meta={"action_state": intersection_bucket}))
        for side in ("north", "south", "east", "west"):
            side_answer, side_bucket = self._side_safe_action_answer(side, by_side.get(side, []))
            qas.append(self._v5_qa("4_3_2_side_action", f"What precaution should traffic take on the {side} approach of the intersection right now?", side_answer, 1.0, {"side": side, **self._action_state_targets(side_bucket)}, sample_meta={"action_state": side_bucket, "direction": side}))
        for side in ("north", "south", "east", "west"):
            lane_map = {lane: lane_objs for (lane_side, lane), lane_objs in by_lane.items() if lane_side == side}
            for lane in (self.LEFT_TURN_LANE, self.STRAIGHT_LANE, self.RIGHT_TURN_LANE):
                lane_objs = lane_map.get(lane, [])
                lane_answer, lane_bucket = self._lane_safe_action_answer(side, lane, lane_objs, objects)
                lane_label = self._lane_function_label(lane)
                qas.append(self._v5_qa("4_3_3_lane_action", f"How should traffic behave in the {lane_label} on the {side} approach right now?", lane_answer, 1.0, {"side": side, "lane_function": lane_label, **self._action_state_targets(lane_bucket)}, sample_meta={"action_state": lane_bucket, "lane_function": lane_label, "direction": side}))
        for obj in self._candidates(objects):
            object_answer, object_bucket = self._object_safe_action_answer(obj, objects)
            qas.append(self._v5_qa("4_3_4_object_action", f"What is the safest action guidance for {self._ref(obj)}?", object_answer, 1.0, {**self._object_targets(obj, include_type=True), **self._action_state_targets(object_bucket)}, sample_meta={"action_state": object_bucket, "object_id": obj["ref_id"], "object_type": obj["type"]}))
        return qas

    def _lane_function_label(self, lane: Optional[str]) -> str:
        if lane in {self.LEFT_TURN_LANE, self.STRAIGHT_LANE, self.RIGHT_TURN_LANE}:
            return str(lane)
        return "through lane"

    def _ref_id_sort_key(self, obj: Dict) -> Tuple[int, str]:
        ref_id = str(obj.get("ref_id", ""))
        if ref_id.startswith("o") and ref_id[1:].isdigit():
            return (int(ref_id[1:]), ref_id)
        return (10**9, ref_id)

    def _closest_participant_pair(self, objects: List[Dict]) -> Optional[Tuple[Dict, Dict, float]]:
        ordered = sorted(objects, key=self._ref_id_sort_key)
        cross_side_pairs: List[Tuple[Dict, Dict]] = []
        fallback_pairs: List[Tuple[Dict, Dict]] = []
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if {a["type"], b["type"]} == {"truck", "trailer"}:
                    continue
                pair = (a, b)
                if a.get("side") != b.get("side"):
                    cross_side_pairs.append(pair)
                else:
                    fallback_pairs.append(pair)
        selected = cross_side_pairs[0] if cross_side_pairs else fallback_pairs[0] if fallback_pairs else None
        if selected is None:
            return None
        a, b = selected
        return (a, b, round(self._dist(a, b), 1))

    def _qa(
        self,
        chapter: str,
        section: str,
        subtemplate: str,
        question: str,
        answer,
        priority: float,
        answer_bucket: Optional[str] = None,
        structured_targets: Optional[Dict] = None,
        sample_meta: Optional[Dict] = None,
        placeholder: bool = False,
    ) -> Dict:
        qa = {
            "question": question,
            "answer": answer,
            "chapter": chapter,
            "section": section,
            "subtemplate": subtemplate,
            "structured_targets": structured_targets or {},
            "_sampling_priority": float(priority),
            "_sample_meta": sample_meta or {},
            "placeholder": bool(placeholder),
        }
        if answer_bucket is not None:
            qa["_answer_bucket"] = answer_bucket
        return qa

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _print_progress(self, label: str, current: int, total: int, start_time: Optional[float] = None) -> None:
        if total <= 0:
            return
        width = 30
        filled = int(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        message = f"{label}: [{bar}] {current}/{total}"
        if start_time is not None and current > 0:
            elapsed = time.perf_counter() - start_time
            eta = max(0.0, (elapsed / current) * (total - current))
            message += f" | elapsed {self._format_duration(elapsed)} | eta {self._format_duration(eta)}"
        if sys.stdout.isatty():
            sys.stdout.write(f"\x1b[2K\r{message}")
            if current >= total:
                sys.stdout.write("\n")
            sys.stdout.flush()
            return
        if current >= total:
            print(message, flush=True)

    def sample_keyframes(self, frames: List[Dict], keyframe_fps: float) -> List[Dict]:
        step = max(1, int(round(self.SOURCE_SCENE_FPS / keyframe_fps)))
        sampled, buffer, current_scene = [], [], None
        for frame in frames:
            scene = self._scene_id(frame)
            if current_scene is None:
                current_scene = scene
            if scene != current_scene:
                sampled.extend(buffer[::step])
                buffer, current_scene = [frame], scene
            else:
                buffer.append(frame)
        if buffer:
            sampled.extend(buffer[::step])
        return sampled

    def _frame_has_objects(self, frame: Dict) -> bool:
        token = self._token(frame)
        snapshots = self.snapshots_by_token.get(token)
        return bool(snapshots)

    def _nearest(self, ref_obj: Dict, objects: List[Dict], max_dist: Optional[float] = None) -> Optional[Dict]:
        candidates = []
        for obj in objects:
            if obj["id"] == ref_obj["id"]:
                continue
            dist = self._dist(ref_obj, obj)
            if max_dist is not None and dist > max_dist:
                continue
            candidates.append((dist, obj))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _rect(self, x: float, y: float, dx: float, dy: float, yaw: float) -> np.ndarray:
        hx, hy = max(dx, 0.1) / 2.0, max(dy, 0.1) / 2.0
        local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        world = local @ rot.T
        world[:, 0] += x
        world[:, 1] += y
        return world

    def _overlap(self, a: Dict, b: Dict, t: float) -> bool:
        c1 = self._rect(a["x"] + a["vx"] * t, a["y"] + a["vy"] * t, a.get("dx", 0.0), a.get("dy", 0.0), a.get("yaw", 0.0))
        c2 = self._rect(b["x"] + b["vx"] * t, b["y"] + b["vy"] * t, b.get("dx", 0.0), b.get("dy", 0.0), b.get("yaw", 0.0))
        axes = []
        for corners in (c1, c2):
            for i in range(4):
                edge = corners[(i + 1) % 4] - corners[i]
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm > 1e-8:
                    axes.append(axis / norm)
        for axis in axes:
            p1 = c1 @ axis
            p2 = c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def _pair_priority(self, a: Dict, b: Dict) -> int:
        return int((a["type"] in self.VRU_TYPES and b["type"] in self.VEHICLE_TYPES) or (b["type"] in self.VRU_TYPES and a["type"] in self.VEHICLE_TYPES))

    def _future_pair_conflict(self, a: Dict, b: Dict, threshold: float) -> Optional[Dict]:
        frame_cache = self._active_frame_cache
        if frame_cache is not None:
            cache_key = (self._pair_key(a, b), float(threshold))
            cache = frame_cache["future_conflict"]
            if cache_key in cache:
                return cache[cache_key]
        if {a["type"], b["type"]} == {"truck", "trailer"}:
            if frame_cache is not None:
                frame_cache["future_conflict"][(self._pair_key(a, b), float(threshold))] = None
            return None
        best_dist, best_time = float("inf"), 0.0
        t = 0.0
        while t <= self.PRED_HORIZON + 1e-6:
            future_a = dict(a)
            future_b = dict(b)
            future_a["x"] = a["x"] + a["vx"] * t
            future_a["y"] = a["y"] + a["vy"] * t
            future_b["x"] = b["x"] + b["vx"] * t
            future_b["y"] = b["y"] + b["vy"] * t
            d = self._box_min_distance(future_a, future_b)
            if d < best_dist:
                best_dist, best_time = d, round(t, 2)
            t += self.PRED_STEP
        if best_dist > threshold:
            result = None
        else:
            result = {
            "obj1": a,
            "obj2": b,
            "time": best_time,
            "distance": round(best_dist, 2),
            "priority": self._pair_priority(a, b),
            "trigger": "distance",
        }
        if frame_cache is not None:
            frame_cache["future_conflict"][(self._pair_key(a, b), float(threshold))] = result
        return result

    def _near_miss(self, objects: List[Dict], focus: Optional[Dict] = None) -> Optional[Dict]:
        frame_cache = self._active_frame_cache
        cache_key = None
        if frame_cache is not None:
            cache_key = ("near_miss", self._object_ids_key(objects), None if focus is None else str(focus["id"]))
            events_cache = frame_cache["events"]
            if cache_key in events_cache:
                return events_cache[cache_key]
        candidates = []
        pool = [focus] if focus is not None else objects
        for a in pool:
            for b in objects:
                if a is None or a["id"] == b["id"]:
                    continue
                if focus is None and a["id"] > b["id"]:
                    continue
                event = self._future_pair_conflict(a, b, self.NEAR_MISS_DIST)
                if event is not None:
                    candidates.append(event)
        if not candidates:
            result = None
        else:
            candidates.sort(key=lambda item: (-item["priority"], item["time"], item["distance"]))
            result = candidates[0]
        if frame_cache is not None and cache_key is not None:
            frame_cache["events"][cache_key] = result
        return result

    def _trajectory_waypoints(self, obj: Dict) -> Optional[List[Dict[str, float]]]:
        if self.active_keyframe_fps <= 0:
            return None
        step_frames = int(round(self.SOURCE_SCENE_FPS / self.active_keyframe_fps))
        if step_frames <= 0:
            return None
        required_indices = [step_frames * idx - 1 for idx in range(1, 5)]
        future = self.future_track_cache.get((obj["frame_token"], obj["id"]), [])
        required_length = step_frames * 4
        if len(future) < required_length:
            return None
        if any(idx >= len(future) or future[idx] is None for idx in required_indices):
            return None
        waypoints = []
        for idx in required_indices:
            snapshot = future[idx]
            time_offset = (idx + 1) / self.SOURCE_SCENE_FPS
            waypoints.append(
                {
                    "time": round(time_offset, 2),
                    "dx": round(snapshot["x"] - obj["x"], 2),
                    "dy": round(snapshot["y"] - obj["y"], 2),
                }
            )
        return waypoints

    def _moving_vehicle_count(self, vehicles: List[Dict]) -> int:
        return sum(1 for obj in vehicles if self._speed(obj) >= 2.0 and obj["status"] != "stopped")

    def _vehicle_objects(self, objects: List[Dict]) -> List[Dict]:
        return [obj for obj in objects if obj["type"] in self.VEHICLE_TYPES]

    def _slow_vehicle_objects(self, objects: List[Dict]) -> List[Dict]:
        return [obj for obj in self._vehicle_objects(objects) if obj["status"] != "stopped" and self.MOVING_THRESHOLD <= self._speed(obj) < 2.0]

    def _fast_vehicle_objects(self, objects: List[Dict]) -> List[Dict]:
        return [obj for obj in self._vehicle_objects(objects) if obj["status"] != "stopped" and self._speed(obj) >= 8.0]

    def _center_occupancy_ratio(self, objects: List[Dict]) -> float:
        center_area = (28.0 - 7.4) * (23.4 - 3.0)
        if center_area <= 0:
            return 0.0
        occupied = 0.0
        for obj in self._vehicle_objects(objects):
            if obj["side"] != "center":
                continue
            occupied += max(obj.get("dx", 0.1), 0.1) * max(obj.get("dy", 0.1), 0.1)
        return max(0.0, min(1.0, occupied / center_area))

    def _congestion(self, objects: List[Dict]) -> str:
        vehicles = [o for o in objects if o["type"] in self.VEHICLE_TYPES]
        vehicle_count = len(vehicles)
        moving_count = self._moving_vehicle_count(vehicles)
        score10 = 13 * vehicle_count - 10 * moving_count
        if score10 <= 33:
            return "free-flowing"
        if score10 <= 46:
            return "light traffic"
        if score10 <= 60:
            return "slightly congested"
        if score10 <= 77:
            return "moderately congested"
        return "heavily congested"

    def _split_by_side_and_lane(self, objects: List[Dict]):
        by_side, by_lane = defaultdict(list), defaultdict(list)
        for obj in objects:
            by_side[obj["side"]].append(obj)
            if obj["lane"] is not None:
                by_lane[(obj["side"], obj["lane"])].append(obj)
        return by_side, by_lane

    def _join(self, parts: List[str]) -> str:
        parts = [p for p in parts if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _pluralize(self, noun: str, count: int) -> str:
        if count == 1:
            return noun
        if noun == "bus":
            return "buses"
        if noun.endswith(("s", "x", "z", "ch", "sh")):
            return noun + "es"
        return noun + "s"

    def _count_word(self, count: int) -> str:
        words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        return words.get(count, str(count))

    def _group_breakdown_clause(self, vehicles: int, vrus: int) -> str:
        parts = []
        if vehicles > 0:
            parts.append(f"{vehicles} {'vehicle' if vehicles == 1 else 'vehicles'}")
        if vrus > 0:
            parts.append(f"{vrus} {'vulnerable road user' if vrus == 1 else 'vulnerable road users'}")
        return f", including {self._join(parts)}" if parts else ""

    def _group_counts(self, objects: List[Dict]) -> Tuple[int, int, int]:
        total = len(objects)
        vehicles = sum(1 for obj in objects if obj["type"] in self.COARSE_VEHICLE_TYPES)
        vrus = sum(1 for obj in objects if obj["type"] in self.VRU_TYPES)
        return total, vehicles, vrus

    def _format_group_overview(self, objects: List[Dict], location_phrase: str) -> str:
        total, vehicles, vrus = self._group_counts(objects)
        participant_word = "traffic participant" if total == 1 else "traffic participants"
        breakdown = self._group_breakdown_clause(vehicles, vrus)
        return f"There are currently {total} {participant_word} {location_phrase}{breakdown}."

    def _format_intersection_overview(self, objects: List[Dict]) -> str:
        return self._format_group_overview(objects, "at the intersection")

    def _side_rank(self, side: str, objects: List[Dict]) -> Tuple[int, int, int]:
        total, vehicles, _ = self._group_counts(objects)
        order = self.SIDE_ORDER.index(side) if side in self.SIDE_ORDER else len(self.SIDE_ORDER)
        return total, vehicles, -order

    def _lane_rank(self, lane: str, objects: List[Dict]) -> Tuple[int, int, int]:
        total, vehicles, _ = self._group_counts(objects)
        order = self.LANE_ORDER.index(lane) if lane in self.LANE_ORDER else len(self.LANE_ORDER)
        return total, vehicles, -order

    def _queue_prediction_counts(self, lane_objs: List[Dict]) -> Tuple[int, int, int]:
        stopped = sum(1 for obj in lane_objs if obj["type"] in self.VEHICLE_TYPES and obj["status"] == "stopped")
        slow = sum(1 for obj in lane_objs if obj["type"] in self.VEHICLE_TYPES and obj["status"] != "stopped" and self.MOVING_THRESHOLD <= self._speed(obj) < 2.0)
        moving = sum(1 for obj in lane_objs if obj["type"] in self.VEHICLE_TYPES and self._speed(obj) >= 2.0 and obj["status"] != "stopped")
        return stopped, slow, moving

    def _queue_prediction_rank(self, lane: str, objects: List[Dict]) -> Tuple[float, int, int, int, int]:
        stopped, slow, moving = self._queue_prediction_counts(objects)
        score = 4.0 * stopped + 2.5 * slow + 1.0 * moving
        order = self.LANE_ORDER.index(lane) if lane in self.LANE_ORDER else len(self.LANE_ORDER)
        return score, stopped, slow, moving, -order

    def _heaviest_side_answer(self, by_side: Dict[str, List[Dict]]) -> Tuple[str, str]:
        side, side_objs = max(by_side.items(), key=lambda item: self._side_rank(item[0], item[1]))
        total = len(side_objs)
        return f"The {side} approach currently has the heaviest traffic pressure, with the largest participant count ({total}).", side

    def _longest_queue_lane_answer(self, side: str, side_lanes: Dict[str, List[Dict]], predictive: bool) -> Tuple[Optional[str], Optional[str], float]:
        if predictive:
            lane, lane_objs = max(side_lanes.items(), key=lambda item: self._queue_prediction_rank(item[0], item[1]))
        else:
            lane, lane_objs = max(side_lanes.items(), key=lambda item: self._lane_rank(item[0], item[1]))
        total, vehicles, vrus = self._group_counts(lane_objs)
        breakdown = self._group_breakdown_clause(vehicles, vrus)
        if predictive:
            stopped, slow, moving = self._queue_prediction_counts(lane_objs)
            score = 4.0 * stopped + 2.5 * slow + 1.0 * moving
            parts = []
            if stopped:
                parts.append(f"{self._count_word(stopped)} stopped vehicle{'s' if stopped != 1 else ''}")
            if slow:
                parts.append(f"{self._count_word(slow)} slow-moving vehicle{'s' if slow != 1 else ''}")
            if not parts:
                return None, None, score
            if moving:
                parts.append(f"{self._count_word(moving)} moving vehicle{'s' if moving != 1 else ''}")
            return f"The {lane} lane is most likely to form a long queue, as it already contains {self._join(parts)}, indicating a clear queuing trend.", lane, score
        return f"The {lane} lane has the longest queue, with {total} traffic participants{breakdown}.", lane, float(total)

    def _global_longest_queue_lane(self, by_lane: Dict[Tuple[str, str], List[Dict]]) -> Tuple[Optional[str], Optional[str]]:
        lane_candidates = [
            ((side, lane), lane_objs)
            for (side, lane), lane_objs in by_lane.items()
            if side in {"north", "south", "east", "west"} and lane is not None
        ]
        if not lane_candidates:
            return None, None
        (side, lane), _ = max(lane_candidates, key=lambda item: self._lane_rank(item[0][1], item[1]))
        return side, lane

    def _direction_vector(self, obj: Dict) -> np.ndarray:
        direction = np.array([obj.get("vx", 0.0), obj.get("vy", 0.0)], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            yaw = float(obj.get("yaw", 0.0))
            direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
            norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.zeros(2, dtype=float)
        return direction / norm

    def _is_crossing_like(self, vru: Dict, vehicle: Dict, event: Dict) -> bool:
        if vru["side"] == "center":
            return True
        if vru["type"] not in {"pedestrian", "bicycle"}:
            return False
        if vru["status"] not in {"walking", "running", "moving"}:
            return False
        if vehicle["side"] != vru["side"]:
            return True
        vru_dir = self._direction_vector(vru)
        veh_dir = self._direction_vector(vehicle)
        if float(np.linalg.norm(vru_dir)) < 1e-6 or float(np.linalg.norm(veh_dir)) < 1e-6:
            return False
        return abs(float(np.dot(vru_dir, veh_dir))) <= 0.5

    def _abnormal_proximity_events(self, objects: List[Dict], threshold: float = ABNORMAL_PROXIMITY_THRESHOLD) -> List[Dict]:
        frame_cache = self._active_frame_cache
        cache_key = None
        if frame_cache is not None:
            cache_key = ("abnormal_proximity", self._object_ids_key(objects), float(threshold))
            events_cache = frame_cache["events"]
            if cache_key in events_cache:
                return events_cache[cache_key]
        events = []
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                if a["type"] not in self.VEHICLE_TYPES and b["type"] not in self.VEHICLE_TYPES:
                    continue
                event = self._future_pair_conflict(a, b, threshold)
                if event is None:
                    continue
                severity = 3.0 + event["priority"]
                severity += max(0.0, threshold - event["distance"])
                enriched = dict(event)
                enriched["severity"] = round(severity, 3)
                events.append(enriched)
        events.sort(key=lambda item: (-item["severity"], item["time"], item["distance"], item["obj1"]["id"], item["obj2"]["id"]))
        if frame_cache is not None and cache_key is not None:
            frame_cache["events"][cache_key] = events
        return events

    def _irregular_crossing_events(self, objects: List[Dict]) -> List[Dict]:
        frame_cache = self._active_frame_cache
        cache_key = None
        if frame_cache is not None:
            cache_key = ("irregular_crossing", self._object_ids_key(objects))
            events_cache = frame_cache["events"]
            if cache_key in events_cache:
                return events_cache[cache_key]
        events = []
        vehicles = self._vehicle_objects(objects)
        for obj in objects:
            if obj["type"] not in {"pedestrian", "bicycle"}:
                continue
            best_candidate = None
            for vehicle in vehicles:
                if obj["id"] == vehicle["id"]:
                    continue
                event = self._future_pair_conflict(obj, vehicle, self.IRREGULAR_CROSSING_THRESHOLD)
                if event is None or not self._is_crossing_like(obj, vehicle, event):
                    continue
                severity = 4.0 + event["priority"]
                severity += max(0.0, self.IRREGULAR_CROSSING_THRESHOLD - event["distance"])
                candidate = {"vru": obj, "vehicle": vehicle, "distance": event["distance"], "time": event["time"], "trigger": event["trigger"], "severity": round(severity, 3)}
                if best_candidate is None or (candidate["severity"], -candidate["time"], -candidate["distance"]) > (best_candidate["severity"], -best_candidate["time"], -best_candidate["distance"]):
                    best_candidate = candidate
            if best_candidate is not None:
                events.append(best_candidate)
        events.sort(key=lambda item: (-item["severity"], item["time"], item["distance"], item["vru"]["id"], item["vehicle"]["id"]))
        if frame_cache is not None and cache_key is not None:
            frame_cache["events"][cache_key] = events
        return events

    def _improper_stopping_events(self, objects: List[Dict]) -> List[Dict]:
        return [{"object": obj} for obj in self._vehicle_objects(objects) if obj["status"] == "stopped" and obj["side"] == "center"]

    def _abnormal_behavior_event(self, objects: List[Dict]) -> Optional[Dict]:
        candidates = []
        proximity_events = self._abnormal_proximity_events(objects, threshold=self.ABNORMAL_PROXIMITY_THRESHOLD)
        if proximity_events:
            event = proximity_events[0]
            side = event["obj1"]["side"] if event["obj1"]["side"] == event["obj2"]["side"] else "intersection"
            candidates.append({"kind": "abnormal proximity", "region": side, "event": event, "severity": event["severity"], "priority": 3})
        speeding = [o for o in objects if o["status"] == "speeding"]
        if speeding:
            obj = max(speeding, key=self._speed)
            candidates.append({"kind": "speeding", "region": obj["side"], "event": {"object": obj}, "severity": round(1.0 + max(0.0, self._speed(obj) - self.VEHICLE_OVERSPEED_THRESHOLD), 3), "priority": 1})
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item["severity"], -item["priority"]))
        return candidates[0]

    def _lane_queue_metrics(self, lane_objs: List[Dict]) -> Dict[str, float]:
        vehicles = self._vehicle_objects(lane_objs)
        stopped = sum(1 for obj in vehicles if obj["status"] == "stopped")
        slow = sum(1 for obj in vehicles if obj["status"] != "stopped" and self.MOVING_THRESHOLD <= self._speed(obj) < 2.0)
        fast = sum(1 for obj in vehicles if obj["status"] != "stopped" and self._speed(obj) >= 8.0)
        normal = sum(1 for obj in vehicles if obj["status"] != "stopped" and 2.0 <= self._speed(obj) < 8.0)
        queued = sorted([obj for obj in vehicles if obj["status"] == "stopped" or self.MOVING_THRESHOLD <= self._speed(obj) < 2.0], key=self._center_dist)
        if queued:
            longest_segment = 1
            current_segment = 1
            gap_count = 0
            for previous, current in zip(queued, queued[1:]):
                if self._dist(previous, current) <= 2.5:
                    current_segment += 1
                else:
                    gap_count += 1
                    current_segment = 1
                longest_segment = max(longest_segment, current_segment)
        else:
            longest_segment = 0
            gap_count = 0
        queue_size = stopped + slow
        queue_score = 4.0 * stopped + 2.0 * slow - 1.0 * fast + 2.0 * longest_segment - 1.0 * gap_count
        return {"vehicle_count": len(vehicles), "stopped": stopped, "slow": slow, "fast": fast, "normal": normal, "queue_size": queue_size, "continuous_segment": longest_segment, "gap_count": gap_count, "queue_score": queue_score}

    def _lane_is_queue_like(self, metrics: Dict[str, float]) -> bool:
        return metrics["queue_size"] >= 2 and metrics["queue_score"] >= 6.0

    def _planning_scene_stats(self, objects: List[Dict]) -> Dict[str, float]:
        vehicles = self._vehicle_objects(objects)
        vrus = [obj for obj in objects if obj["type"] in self.VRU_TYPES]
        proximity_events = self._abnormal_proximity_events(objects, threshold=self.ABNORMAL_PROXIMITY_THRESHOLD)
        crossing_events = self._irregular_crossing_events(objects)
        improper_events = self._improper_stopping_events(objects)
        _, by_lane = self._split_by_side_and_lane(objects)
        lane_metrics = [self._lane_queue_metrics(lane_objs) for lane_objs in by_lane.values()]
        queue_like_count = sum(1 for metrics in lane_metrics if self._lane_is_queue_like(metrics))
        q_max = max((metrics["queue_size"] for metrics in lane_metrics), default=0)
        n_v = len(vehicles)
        n_s = sum(1 for obj in vehicles if obj["status"] == "stopped")
        n_l = len(self._slow_vehicle_objects(objects))
        n_f = len(self._fast_vehicle_objects(objects))
        n_spd = sum(1 for obj in vehicles if self._speed(obj) >= self.VEHICLE_OVERSPEED_THRESHOLD - self.SIDE_SPEEDING_MARGIN)
        return {"N": len(objects), "N_v": n_v, "N_vru": len(vrus), "N_s": n_s, "N_l": n_l, "N_f": n_f, "N_p": len(proximity_events), "N_c": len(crossing_events), "N_i": len(improper_events), "N_spd": n_spd, "L_q": queue_like_count, "Q_max": q_max, "O": self._center_occupancy_ratio(objects), "R_s": n_s / max(n_v, 1), "R_sl": (n_s + n_l) / max(n_v, 1)}

    def _planning_side_stats(self, side: str, side_objs: List[Dict]) -> Dict[str, float]:
        vehicles = self._vehicle_objects(side_objs)
        vrus = [obj for obj in side_objs if obj["type"] in self.VRU_TYPES]
        proximity_events = self._abnormal_proximity_events(side_objs, threshold=self.ABNORMAL_PROXIMITY_THRESHOLD)
        crossing_events = self._irregular_crossing_events(side_objs)
        improper_events = self._improper_stopping_events(side_objs)
        _, by_lane = self._split_by_side_and_lane(side_objs)
        lane_metrics = [self._lane_queue_metrics(lane_objs) for lane_objs in by_lane.values()]
        queue_like_count = sum(1 for metrics in lane_metrics if self._lane_is_queue_like(metrics))
        q_max = max((metrics["queue_size"] for metrics in lane_metrics), default=0)
        n_v = len(vehicles)
        n_s = sum(1 for obj in vehicles if obj["status"] == "stopped")
        n_l = len(self._slow_vehicle_objects(side_objs))
        n_f = len(self._fast_vehicle_objects(side_objs))
        return {"side": side, "N_v": n_v, "N_vru": len(vrus), "N_s": n_s, "N_l": n_l, "N_f": n_f, "N_p": len(proximity_events), "N_c": len(crossing_events), "N_i": len(improper_events), "L_q": queue_like_count, "Q_max": q_max, "R_s": n_s / max(n_v, 1), "R_sl": (n_s + n_l) / max(n_v, 1)}

    def _lane_change_attempts(self, side: str, lane: str, objects: List[Dict]) -> int:
        count = 0
        for obj in objects:
            if obj["type"] not in self.VEHICLE_TYPES or obj["side"] != side:
                continue
            target_lane = self.motion_cache.get((obj["frame_token"], obj["id"]), {}).get("target_lane")
            if obj["status"] == "lane-changing" and (obj["lane"] == lane or target_lane == lane):
                count += 1
        return count

    def _planning_lane_stats(self, side: str, lane: str, lane_objs: List[Dict], objects: List[Dict]) -> Dict[str, float]:
        metrics = self._lane_queue_metrics(lane_objs)
        vehicles = self._vehicle_objects(lane_objs)
        proximity_events = []
        for event in self._abnormal_proximity_events(objects, threshold=self.ABNORMAL_PROXIMITY_THRESHOLD):
            if (event["obj1"]["side"] == side and event["obj1"]["lane"] == lane) or (event["obj2"]["side"] == side and event["obj2"]["lane"] == lane):
                proximity_events.append(event)
        vru_ids = set()
        for vru in objects:
            if vru["type"] in self.VRU_TYPES and any(self._dist(vru, vehicle) <= 3.0 for vehicle in vehicles):
                vru_ids.add(vru["id"])
        n_v = len(vehicles)
        return {"N_v": n_v, "N_s": metrics["stopped"], "N_l": metrics["slow"], "N_f": metrics["fast"], "N_lc": self._lane_change_attempts(side, lane, objects), "N_p": len(proximity_events), "N_vru": len(vru_ids), "C_q": metrics["continuous_segment"], "G_q": metrics["gap_count"], "Qscore": metrics["queue_score"], "R_s": metrics["stopped"] / max(n_v, 1), "R_sl": (metrics["stopped"] + metrics["slow"]) / max(n_v, 1)}

    def _object_forward_clearance(self, obj: Dict, objects: List[Dict]) -> float:
        same_lane_ahead = [other for other in objects if other["id"] != obj["id"] and other["side"] == obj["side"] and other["lane"] == obj["lane"] and self._center_dist(other) < self._center_dist(obj)]
        if same_lane_ahead:
            return min(self._dist(obj, other) for other in same_lane_ahead)
        direction = np.array([obj.get("vx", 0.0), obj.get("vy", 0.0)], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            yaw = float(obj.get("yaw", 0.0))
            direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
            norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return float("inf")
        direction = direction / norm
        candidates = []
        for other in objects:
            if other["id"] == obj["id"]:
                continue
            offset = np.array([other["x"] - obj["x"], other["y"] - obj["y"]], dtype=float)
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                continue
            if float(np.dot(direction, offset / distance)) >= 0.707:
                candidates.append(distance)
        return min(candidates) if candidates else float("inf")

    def _object_relative_speed(self, obj: Dict, objects: List[Dict]) -> float:
        peers = [self._speed(other) for other in objects if other["id"] != obj["id"] and other["type"] in self.VEHICLE_TYPES and other["side"] == obj["side"] and other["lane"] == obj["lane"]]
        if not peers:
            return 0.0
        return self._speed(obj) - float(np.median(peers))

    def _planning_object_stats(self, obj: Dict, objects: List[Dict]) -> Dict[str, float]:
        others = [other for other in objects if other["id"] != obj["id"]]
        d_min = min((self._dist(obj, other) for other in others), default=float("inf"))
        vru_others = [other for other in others if other["type"] in self.VRU_TYPES]
        d_vru = min((self._dist(obj, other) for other in vru_others), default=float("inf"))
        n_nbr = sum(1 for other in others if self._dist(obj, other) <= 3.0)
        proximity_count = sum(1 for event in self._abnormal_proximity_events(objects, threshold=self.ABNORMAL_PROXIMITY_THRESHOLD) if obj["id"] in {event["obj1"]["id"], event["obj2"]["id"]})
        crossing_count = sum(1 for event in self._irregular_crossing_events(objects) if obj["id"] in {event["vru"]["id"], event["vehicle"]["id"]})
        projected_near_miss = 1 if self._near_miss(objects, obj) is not None else 0
        return {"d_min": d_min, "d_vru": d_vru, "v": self._speed(obj), "delta_v": self._object_relative_speed(obj, objects), "n_nbr": n_nbr, "I_c": proximity_count + crossing_count + projected_near_miss, "T_f": self._object_forward_clearance(obj, objects), "L_c": 1 if obj["status"] == "lane-changing" else 0, "Z_center": 1 if obj["side"] == "center" else 0}

    def _intersection_guidance_state(self, objects: List[Dict]) -> str:
        stats = self._planning_scene_stats(objects)
        if stats["O"] >= 0.45 or (stats["L_q"] >= 2 and stats["R_sl"] >= 0.50) or (stats["Q_max"] >= 4 and stats["N_s"] >= 3) or (stats["N_s"] >= 3 and stats["R_s"] >= 0.35):
            return "QUEUE_MANAGEMENT"
        if (stats["N_p"] + stats["N_c"] >= 1) or (stats["N_vru"] >= 1 and stats["N_spd"] >= 1) or (stats["N_i"] >= 1):
            return "CONFLICT_SUPPRESSION"
        if stats["R_s"] < 0.15 and stats["R_sl"] < 0.30 and stats["L_q"] == 0 and stats["N_p"] == 0 and stats["N_c"] == 0 and stats["N_i"] == 0 and stats["N_spd"] == 0 and stats["O"] < 0.20:
            return "FLOW_STABLE"
        return "FLOW_CALMING"

    def _side_guidance_state(self, side: str, objects: List[Dict]) -> str:
        stats = self._planning_side_stats(side, objects)
        if stats["N_p"] >= 1 or (stats["N_vru"] >= 1 and stats["N_v"] >= 2) or (stats["N_p"] + stats["N_c"] >= 1):
            return "SIDE_CLEARANCE_PROTECTION"
        if stats["N_c"] >= 1 or (stats["N_vru"] >= 1 and stats["N_c"] >= 1) or stats["N_vru"] >= 2:
            return "SIDE_CROSSING_AWARENESS"
        if (stats["N_f"] >= 1 and stats["N_p"] == 0 and stats["N_c"] == 0) or stats["N_f"] >= 2:
            return "SIDE_SPEED_MODERATION"
        if (stats["L_q"] >= 1 and stats["Q_max"] >= 3) or stats["R_sl"] >= 0.50 or stats["N_s"] >= 3:
            return "SIDE_QUEUE_STABILIZATION"
        return "SIDE_GENERAL_CAUTION"

    def _lane_guidance_state(self, side: str, lane: str, lane_objs: List[Dict], objects: List[Dict]) -> str:
        stats = self._planning_lane_stats(side, lane, lane_objs, objects)
        if stats["N_p"] >= 1 or stats["N_vru"] >= 1 or (stats["N_p"] >= 1 and stats["N_f"] >= 1):
            return "LANE_CLEARANCE_MAINTENANCE"
        if stats["N_s"] >= 2 or (stats["Qscore"] >= 8 and stats["N_f"] == 0) or (stats["C_q"] >= 2 and stats["G_q"] <= 1):
            return "LANE_PREPARE_TO_STOP"
        if stats["Qscore"] >= 10 or stats["C_q"] >= 3 or stats["R_sl"] >= 0.60:
            return "LANE_QUEUE_PRESERVATION"
        if (stats["N_f"] >= 1 and stats["R_sl"] < 0.40) or stats["N_f"] >= 2:
            return "LANE_SPEED_REDUCTION"
        return "LANE_GENERAL_ORDER"

    def _object_guidance_state(self, obj: Dict, objects: List[Dict]) -> str:
        stats = self._planning_object_stats(obj, objects)
        if (stats["I_c"] >= 1 and stats["d_min"] < 5.5) or stats["d_vru"] < 6.5 or (stats["Z_center"] == 1 and stats["I_c"] >= 1) or stats["I_c"] >= 2:
            return "OBJECT_YIELD_NOW"
        if stats["T_f"] < 11.0 or (stats["Z_center"] == 1 and stats["T_f"] < 8.0) or (stats["I_c"] >= 1 and stats["T_f"] < 11.0):
            return "OBJECT_PREPARE_TO_STOP"
        if stats["delta_v"] > 3.8 or (stats["v"] > 6.5 and stats["Z_center"] == 1) or (stats["v"] > 6.0 and stats["d_min"] < 4.5):
            return "OBJECT_SLOW_DOWN"
        return "OBJECT_PROCEED_CAUTIOUSLY"

    def _intersection_safe_action_answer(self, objects: List[Dict]) -> Tuple[str, str]:
        state = self._intersection_guidance_state(objects)
        mapping = {
            "FLOW_STABLE": "Traffic should continue in a coordinated and orderly way.",
            "FLOW_CALMING": "Traffic should move more calmly and with reduced speed.",
            "QUEUE_MANAGEMENT": "Traffic should proceed in a more orderly and tightly managed way.",
            "CONFLICT_SUPPRESSION": "Traffic should reduce aggressive movement and prioritize conflict avoidance.",
        }
        return mapping[state], state

    def _side_safe_action_answer(self, side: str, objects: List[Dict]) -> Tuple[str, str]:
        state = self._side_guidance_state(side, objects)
        mapping = {
            "SIDE_CLEARANCE_PROTECTION": "Traffic should keep safer local spacing.",
            "SIDE_CROSSING_AWARENESS": "Traffic should yield clearly to crossing activity.",
            "SIDE_SPEED_MODERATION": "Traffic should moderate speed and maintain safer spacing.",
            "SIDE_QUEUE_STABILIZATION": "Traffic should stabilize queue movement and avoid unnecessary disruption.",
            "SIDE_GENERAL_CAUTION": "Traffic should proceed cautiously and remain orderly.",
        }
        return mapping[state], state

    def _lane_safe_action_answer(self, side: str, lane: str, objects: List[Dict], all_objects: List[Dict]) -> Tuple[str, str]:
        state = self._lane_guidance_state(side, lane, objects, all_objects)
        lane_label = self._lane_function_label(lane)
        mapping = {
            "LANE_CLEARANCE_MAINTENANCE": "Traffic should maintain clearance and avoid tight local conflicts.",
            "LANE_PREPARE_TO_STOP": "Traffic should prepare to stop and avoid pressing forward.",
            "LANE_QUEUE_PRESERVATION": "Traffic should preserve queue order and avoid disruption.",
            "LANE_SPEED_REDUCTION": "Traffic should reduce speed and proceed more conservatively.",
            "LANE_GENERAL_ORDER": "Traffic should proceed in an orderly manner.",
        }
        return mapping[state], state

    def _object_safe_action_answer(self, obj: Dict, objects: List[Dict]) -> Tuple[str, str]:
        state = self._object_guidance_state(obj, objects)
        subject = self._object_phrase(obj, capitalized=True)
        mapping = {
            "OBJECT_YIELD_NOW": f"{subject} should yield now.",
            "OBJECT_SLOW_DOWN": f"{subject} should slow down and keep safer local spacing.",
            "OBJECT_PREPARE_TO_STOP": f"{subject} should prepare to stop.",
            "OBJECT_PROCEED_CAUTIOUSLY": f"{subject} should proceed cautiously.",
        }
        return mapping[state], state

    def _motion_summary_label(self, stopped: int, moving: int) -> str:
        total = stopped + moving
        if total == 0:
            return "mixed movement"
        if stopped / total >= 0.8:
            return "mostly stopped"
        if moving / total >= 0.8:
            return "mostly moving"
        return "mixed movement"

    def _ordered_participant_status_items(self, objects: List[Dict], limit: int = 4) -> List[Tuple[Tuple[str, str], int]]:
        counts = defaultdict(int)
        for obj in objects:
            counts[(obj["status"], obj["type"])] += 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))[:limit]

    def _participant_status_clauses(self, objects: List[Dict], limit: int = 4) -> List[str]:
        ordered = self._ordered_participant_status_items(objects, limit=limit)
        clauses = []
        for (status, obj_type), count in ordered:
            noun = self._pluralize(obj_type, count)
            verb = "is" if count == 1 else "are"
            nearby = " nearby" if obj_type == "pedestrian" else ""
            clauses.append(f"{count} {noun} {verb} {status}{nearby}")
        return clauses

    def _side_status_answer(self, side: str, side_objs: List[Dict]) -> str:
        stopped = sum(1 for o in side_objs if o["type"] in self.COARSE_VEHICLE_TYPES and o["status"] == "stopped")
        moving = sum(1 for o in side_objs if o["type"] in self.COARSE_VEHICLE_TYPES and o["status"] not in {"stopped", "speeding"})
        motion_label = self._motion_summary_label(stopped, moving)
        if stopped <= 0 and moving <= 0:
            return f"The {side} approach is {motion_label}."
        return f"The {side} approach is {motion_label}, with {moving} vehicles moving and {stopped} stopped."

    def _side_status_targets(self, side_objs: List[Dict]) -> Dict:
        stopped = sum(1 for obj in side_objs if obj["type"] in self.COARSE_VEHICLE_TYPES and obj["status"] == "stopped")
        moving = sum(1 for obj in side_objs if obj["type"] in self.COARSE_VEHICLE_TYPES and obj["status"] not in {"stopped", "speeding"})
        status_targets = [obj for obj in side_objs if obj["type"] not in self.COARSE_VEHICLE_TYPES]
        extra = [{"type": obj_type, "status": status, "count": int(count)} for (status, obj_type), count in self._ordered_participant_status_items(status_targets, limit=4)]
        return {"motion_label": self._motion_summary_label(stopped, moving), "counts": {"stopped_vehicles": int(stopped), "moving_vehicles": int(moving)}, "extra_status_counts": extra}

    def _candidates(self, objects: List[Dict]) -> List[Dict]:
        return sorted(objects, key=lambda obj: (self._center_dist(obj), obj["id"]))

    def _trajectory_targets(self, waypoints: List[Dict[str, float]]) -> Dict:
        return {"trajectory": {"times_sec": [item["time"] for item in waypoints], "waypoints_xy": [{"dx": item["dx"], "dy": item["dy"]} for item in waypoints]}}

    def _object_targets(self, obj: Dict, include_type: bool = True, include_side: bool = False) -> Dict:
        targets = {"object_id": obj["ref_id"], "raw_tracking_id": obj["id"]}
        if include_type:
            targets["object_type"] = obj["type"]
        if include_side:
            targets["side"] = obj["side"]
        targets.update(self._object_location_targets(obj, "object"))
        return targets

    def _prefixed_object_targets(self, obj: Dict, prefix: str, include_type: bool = False) -> Dict:
        targets = {f"{prefix}_id": obj["ref_id"], f"{prefix}_raw_tracking_id": obj["id"]}
        if include_type:
            targets[f"{prefix}_type"] = obj["type"]
        targets.update(self._object_location_targets(obj, prefix))
        return targets

    def _object_location_targets(self, obj: Dict, prefix: str) -> Dict:
        cam_refs = list(obj.get("cam_refs") or [])
        if not cam_refs:
            cam_refs = [
                {
                    "cam_view": obj.get("cam_view", "north_image"),
                    "x1": obj.get("x1"),
                    "y1": obj.get("y1"),
                }
            ]
        image_refs = []
        for cam_ref in cam_refs:
            x1 = -1.0 if cam_ref.get("x1") is None else round(float(cam_ref["x1"]), 1)
            y1 = -1.0 if cam_ref.get("y1") is None else round(float(cam_ref["y1"]), 1)
            image_refs.append(
                {
                    "image_name": str(cam_ref.get("cam_view", "north_image")),
                    "x1": x1,
                    "y1": y1,
                }
            )
        return {
            f"{prefix}_position": {
                "x": round(float(obj["x"]), 1),
                "y": round(float(obj["y"]), 1),
            },
            f"{prefix}_image_refs": image_refs,
        }

    def _queue_prediction_targets(self, lane: str, lane_objs: List[Dict]) -> Dict:
        stopped, slow, moving = self._queue_prediction_counts(lane_objs)
        return {"winning_lane": lane, "queue_evidence": {"stopped_vehicles": int(stopped), "slow_vehicles": int(slow), "moving_vehicles": int(moving)}}

    def _action_state_targets(self, state: str) -> Dict:
        return {"action_state": state}

    def _side_speeding_targets(self, fastest: Optional[Dict]) -> Dict:
        if fastest is None:
            return {"has_speeding_risk": False, "vehicle_id": None, "vehicle_raw_tracking_id": None, "vehicle_type": None, "side": None, "risk_region": None, "evidence": None, "numeric_targets": {"speed_mps": None}}
        return {"has_speeding_risk": True, "vehicle_id": fastest["ref_id"], "vehicle_raw_tracking_id": fastest["id"], "vehicle_type": fastest["type"], "side": fastest["side"], "risk_region": fastest["side"], "evidence": f"A {fastest['type']} is still moving at about {self._speed(fastest):.1f} m/s.", "numeric_targets": {"speed_mps": round(self._speed(fastest), 1)}}

    def _straight_following_pairs(self, objects: List[Dict]) -> List[Tuple[Dict, Dict]]:
        pairs = []
        for follower in objects:
            if follower["type"] == "pedestrian" or follower["status"] != "straight-going" or follower["lane"] is None or follower["side"] == "center":
                continue
            lead = self._lead_vehicle_same_lane(follower, objects)
            if lead is not None and lead["status"] == "straight-going":
                pairs.append((follower, lead))
        pairs.sort(key=lambda pair: self._dist(pair[0], pair[1]))
        return pairs

    def _interaction_pattern_label(self, event: Dict) -> str:
        types = {event["obj1"]["type"], event["obj2"]["type"]}
        if "pedestrian" in types and any(obj_type not in self.VRU_TYPES for obj_type in types):
            return "left-turn_pedestrian" if {event["obj1"]["status"], event["obj2"]["status"]} & {"turning-left"} else "other"
        if {event["obj1"]["status"], event["obj2"]["status"]} & {"lane-changing"}:
            return "straight_lane_change"
        if any(obj_type in {"bicycle", "motorcycle", "golf cart"} for obj_type in types) and {event["obj1"]["status"], event["obj2"]["status"]} & {"turning-right"}:
            return "right-turn_two-wheeler"
        return "rear-end" if event["obj1"]["side"] == event["obj2"]["side"] and event["obj1"]["lane"] == event["obj2"]["lane"] else "other"

    def _primary_risk_subject(self, objects: List[Dict]) -> Tuple[Optional[Dict], Optional[str]]:
        best_obj = None
        best_reason = None
        best_score = float("-inf")
        for obj in objects:
            stats = self._planning_object_stats(obj, objects)
            score = stats["I_c"] * 3.0 + max(0.0, 5.0 - stats["d_min"]) + (3.0 if obj["status"] == "speeding" else 0.0)
            if stats["d_vru"] < 6.5 and obj["type"] not in self.VRU_TYPES:
                score += 2.0
            reason = "proximity"
            if obj["status"] == "speeding":
                reason = "overspeed"
            elif obj["status"] == "lane-changing":
                reason = "lane_change_conflict"
            elif obj["type"] in self.VRU_TYPES and stats["I_c"] > 0:
                reason = "vru_conflict"
            elif stats["I_c"] > 0:
                reason = "path_crossing"
            if score > best_score:
                best_obj = obj
                best_reason = reason
                best_score = score
        return best_obj, best_reason

    def generate_frame(self, frame: Dict, max_per_type: int) -> List[Dict]:
        objects = self.extract_objects(frame)
        if not objects:
            return []
        token = self._token(frame)
        scene_id = self._scene_id(frame)
        media_paths = self._frame_media_paths(frame)
        self._active_frame_cache = self._make_frame_cache(token)
        try:
            all_items = (
                self._basic_perception_qas(objects)
                + self._spatial_qas(objects)
                + self._temporal_qas(objects)
                + self._interaction_qas(objects)
                + self._scene_qas(objects)
            )
            template_groups: Dict[str, List[Dict]] = defaultdict(list)
            for qa in all_items:
                template_groups[qa["subtemplate"]].append(qa)
            out: List[Dict] = []
            for template_id in sorted(template_groups):
                out.extend(self._select_template_items(template_id, template_groups[template_id], max_per_type))
            out.sort(key=lambda qa: (self.CHAPTER_ORDER.index(qa["chapter"]), qa["subtemplate"], self._rank_tuple(qa)))
            for qa in out:
                qa["frame_token"] = token
                qa["scene_id"] = scene_id
                qa.update(media_paths)
            for idx, qa in enumerate(out, 1):
                qa["question_id"] = f"{token}_{idx:04d}"
            return out
        finally:
            self._active_frame_cache = None

    def generate_dataset(
        self,
        output_path: str,
        max_frames: Optional[int] = None,
        keyframe_fps: float = DEFAULT_KEYFRAME_FPS,
        max_per_type: int = DEFAULT_MAX_PER_TYPE,
        num_workers: int = 1,
    ) -> List[Dict]:
        self.active_keyframe_fps = keyframe_fps
        frames = self.infos[:max_frames] if max_frames is not None else self.infos
        sampled = self.sample_keyframes(frames, keyframe_fps)
        eligible_sampled = [frame for frame in sampled if self._frame_has_objects(frame)]
        eligible_indices = [self.frame_index_by_token[self._token(frame)] for frame in eligible_sampled]
        print(
            f"Source frames: {len(frames)} | "
            f"Sampled keyframes: {len(sampled)} | "
            f"Eligible keyframes: {len(eligible_sampled)}"
        )

        qas: List[Dict] = []
        generation_start = time.perf_counter()
        worker_count = max(1, int(num_workers))
        if worker_count == 1 or len(eligible_indices) <= 1:
            for index, frame in enumerate(eligible_sampled, 1):
                frame_qas = self.generate_frame(frame, max_per_type)
                qas.extend(frame_qas)
                self._print_progress("Generating QA V5", index, len(eligible_sampled), generation_start)
        else:
            ordered_results: Dict[int, List[Dict]] = {}
            max_workers = min(worker_count, len(eligible_indices))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("fork"),
                initializer=_init_v5_worker,
                initargs=(str(self.pkl_path), self.subtemplate_patch_style),
            ) as executor:
                future_to_order = {
                    executor.submit(_generate_v5_frame_worker, frame_index, max_per_type, keyframe_fps): order
                    for order, frame_index in enumerate(eligible_indices)
                }
                completed = 0
                for future in as_completed(future_to_order):
                    order = future_to_order[future]
                    ordered_results[order] = future.result()
                    completed += 1
                    self._print_progress("Generating QA V5", completed, len(eligible_indices), generation_start)
            for order in range(len(eligible_indices)):
                qas.extend(ordered_results[order])

        pre_scene_filter_qas = list(qas)
        (
            pre_scene_filter_chapter_stats,
            pre_scene_filter_section_stats,
            pre_scene_filter_template_stats,
            pre_scene_filter_bucket_stats,
        ) = self._compute_statistics(pre_scene_filter_qas)

        qas = self._apply_temporal_state_suppression(qas)

        post_scene_filter_qas = list(qas)
        (
            post_scene_filter_chapter_stats,
            post_scene_filter_section_stats,
            post_scene_filter_template_stats,
            post_scene_filter_bucket_stats,
        ) = self._compute_statistics(post_scene_filter_qas)

        post_balance_qas = self._apply_global_template_caps(qas)
        (
            post_balance_chapter_stats,
            post_balance_section_stats,
            post_balance_template_stats,
            post_balance_bucket_stats,
        ) = self._compute_statistics(post_balance_qas)

        post_ratio_qas = self._apply_final_template_ratios(post_balance_qas)
        (
            chapter_stats,
            section_stats,
            template_stats,
            bucket_stats,
        ) = self._compute_statistics(post_ratio_qas)

        public_qas = [self._public_qa(qa) for qa in post_ratio_qas]

        payload = {
            "metadata": {
                "version": "v5",
                "source_total_frames": len(frames),
                "sampled_keyframes": len(sampled),
                "eligible_keyframes": len(eligible_sampled),
                "source_scene_fps": self.SOURCE_SCENE_FPS,
                "keyframe_fps": keyframe_fps,
                "num_workers": worker_count,
                "parallel_generation_enabled": bool(worker_count > 1 and len(eligible_indices) > 1),
                "max_per_type_per_frame": max_per_type,
                "pre_scene_filter_total_qas": len(pre_scene_filter_qas),
                "post_scene_filter_total_qas": len(post_scene_filter_qas),
                "pre_balance_total_qas": len(post_scene_filter_qas),
                "post_balance_total_qas": len(post_balance_qas),
                "post_ratio_total_qas": len(post_ratio_qas),
                "total_qas": len(post_ratio_qas),
                "pre_scene_filter_chapter_statistics": pre_scene_filter_chapter_stats,
                "pre_scene_filter_section_statistics": pre_scene_filter_section_stats,
                "pre_scene_filter_template_statistics": pre_scene_filter_template_stats,
                "pre_scene_filter_template_bucket_statistics": pre_scene_filter_bucket_stats,
                "post_scene_filter_chapter_statistics": post_scene_filter_chapter_stats,
                "post_scene_filter_section_statistics": post_scene_filter_section_stats,
                "post_scene_filter_template_statistics": post_scene_filter_template_stats,
                "post_scene_filter_template_bucket_statistics": post_scene_filter_bucket_stats,
                "pre_balance_chapter_statistics": post_scene_filter_chapter_stats,
                "pre_balance_section_statistics": post_scene_filter_section_stats,
                "pre_balance_template_statistics": post_scene_filter_template_stats,
                "pre_balance_template_bucket_statistics": post_scene_filter_bucket_stats,
                "post_balance_chapter_statistics": post_balance_chapter_stats,
                "post_balance_section_statistics": post_balance_section_stats,
                "post_balance_template_statistics": post_balance_template_stats,
                "post_balance_template_bucket_statistics": post_balance_bucket_stats,
                "post_ratio_chapter_statistics": dict(sorted(chapter_stats.items())),
                "post_ratio_section_statistics": dict(sorted(section_stats.items())),
                "post_ratio_template_statistics": dict(sorted(template_stats.items())),
                "post_ratio_template_bucket_statistics": bucket_stats,
                "chapter_statistics": dict(sorted(chapter_stats.items())),
                "section_statistics": dict(sorted(section_stats.items())),
                "template_statistics": dict(sorted(template_stats.items())),
                "template_bucket_statistics": bucket_stats,
                "template_registry": {k: asdict(v) for k, v in sorted(self.template_registry.items())},
                "prompt_metadata": self._prompt_metadata(),
                "structured_targets_enabled": True,
                "geometry_rules": {
                    "boundary_rule": "left_closed_right_open",
                    "side_regions": {
                        "north": "x < 7.5",
                        "south": "x > 28.6",
                        "west": "y < 3.1",
                        "east": "y > 23.8",
                        "center": "7.5 <= x < 28.6 and 3.1 <= y < 23.8",
                    },
                    "stoplines": {
                        "north": "x = 1.0",
                        "south": "x = 35.2",
                        "west": "y = -4.4",
                        "east": "y = 31.8",
                    },
                    "distance_rules": {
                        "participant_distance": "minimum_ground_plane_box_distance",
                        "crosswalk_entry_distance": "crosswalk_principal_axis",
                        "crosswalk_far_edge_distance": "crosswalk_principal_axis",
                    },
                    "timing_rules": {
                        "default_temporal_window_sec": 3.0,
                        "waypoint_times_sec": [0.5, 1.0, 1.5, 2.0],
                    },
                },
            },
            "qa_pairs": public_qas,
        }
        print("Writing JSON...")
        Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(post_ratio_qas)} QA pairs to {output_path}")
        return public_qas
