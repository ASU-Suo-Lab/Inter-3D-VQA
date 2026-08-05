from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from _qa_v5_runtime import IntersectionQAGeneratorV5Runtime, V5TemplateSpec


_WORKER_RUNTIME: Optional["IntersectionQAGeneratorV6Runtime"] = None


def _init_v6_worker(pkl_path: str, subtemplate_patch_style: str) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = IntersectionQAGeneratorV6Runtime(pkl_path, subtemplate_patch_style=subtemplate_patch_style)


def _generate_v6_frame_worker(frame_index: int, max_per_type: int, keyframe_fps: float) -> List[Dict]:
    if _WORKER_RUNTIME is None:
        raise RuntimeError("V6 worker runtime is not initialized.")
    _WORKER_RUNTIME.active_keyframe_fps = keyframe_fps
    frame = _WORKER_RUNTIME.infos[frame_index]
    return _WORKER_RUNTIME.generate_frame(frame, max_per_type)


class IntersectionQAGeneratorV6Runtime(IntersectionQAGeneratorV5Runtime):
    DEFAULT_OUTPUT_NAME = "intersection_qa_pairs_v6.json"
    PROMPT_METADATA_VERSION = "v6_natural_prompt_v1"

    BUCKET_FIELD_BY_TEMPLATE = {
        "1_1_1_lane_first_object_type": "object_type",
        "1_1_2_front_neighbor_type": "rel_dir",
        "1_1_3_approach_vru_exists": "exists",
        "1_1_4_approach_type_count": "object_type",
        "1_2_1_size_bucket": "size_bucket",
        "1_3_1_environment": "weather",
        "1_3_2_vehicle_signal_state": "signal_state",
        "2_1_1_stopline_distance": "object_type",
        "2_1_2_ped_to_far_edge": "crosswalk",
        "2_1_3_participant_distance": "pair_type",
        "2_1_4_nearest_vehicle": "direction",
        "2_2_1_ped_zone": "ped_zone",
        "2_2_2_lane_queue_count": "lane_function",
        "2_2_3_stopline_back_5m_count": "side",
        "2_2_4_longest_queue_lane": "lane_function",
        "2_2_5_crosswalk_blocking": "crosswalk_blocked",
        "3_1_1_current_motion_state": "motion_state",
        "3_1_2_vehicle_maneuver": "maneuver",
        "3_2_1_waypoints": "trajectory",
        "3_2_2_future_region": "future_region",
        "3_3_1_safe_following": "is_safe",
        "3_3_2_likely_long_queue_lane": "lane_function",
        "3_4_1_pair_conflict": "has_conflict",
        "3_4_2_nearest_conflict_participant": "conflict_partner_type",
        "3_4_3_primary_risk_subject": "risk_reason",
        "3_4_4_risk_pattern": "interaction_pattern",
        "4_1_1_overall_state": "overall_state",
        "4_1_2_approach_motion_status": "motion_label",
        "4_1_3_scene_summary": "summary_type",
        "4_1_4_heaviest_traffic_approach": "dominant_side",
        "4_2_1_speeding_risk": "has_speeding_risk",
        "4_2_2_notable_abnormal": "notable_abnormal",
        "4_3_1_intersection_action": "action_state",
        "4_3_2_approach_action": "action_state",
        "4_3_3_lane_action": "action_state",
        "4_3_4_object_action": "action_state",
    }
    PLACEHOLDER_TEMPLATE_IDS = {
        "1_2_2_visibility",
        "1_3_1_environment",
        "1_3_2_vehicle_signal_state",
    }
    TEMPLATE_GLOBAL_BUCKET_CAPS = {}
    TEMPLATE_GLOBAL_GROUP_RATIOS = {}
    TEMPLATE_GLOBAL_YES_NO_RATIOS = {
        "1_1_3_approach_vru_exists": (7, 3),
        "2_2_5_crosswalk_blocking": (7, 3),
        "3_3_1_safe_following": (1, 1),
        "3_4_1_pair_conflict": (7, 3),
        "4_2_1_speeding_risk": (7, 3),
    }
    TEMPLATE_GLOBAL_LABEL_RATIOS = {
        "1_1_1_lane_first_object_type": {("construction_vehicle" if label == "construction vehicle" else label): 1 for label in IntersectionQAGeneratorV5Runtime.FINE_TYPES},
        "1_1_4_approach_type_count": {label: 1 for label in IntersectionQAGeneratorV5Runtime.FINE_TYPES},
        "1_2_1_size_bucket": {
            "small": 1,
            "medium": 1,
            "large": 1,
            "extra-large": 1,
        },
        "2_1_1_stopline_distance": {"car": 1, "truck": 1, "van": 1, "bus": 1},
        "2_1_2_ped_to_far_edge": {"north": 1, "south": 1, "east": 1, "west": 1},
        "2_1_4_nearest_vehicle": {"north": 1, "south": 1, "east": 1, "west": 1},
        "2_2_1_ped_zone": {
            "within the crosswalk": 3,
            "waiting zone": 2,
            "entry zone": 1,
            "exit zone": 1,
        },
        "2_2_2_lane_queue_count": {
            "left-turn lane": 1,
            "through lane": 1,
            "right-turn lane": 1,
        },
        "2_2_4_longest_queue_lane": {
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
        "3_2_2_future_region": {
            "before stop line": 1,
            "intersection center": 1,
            "left-turn exit": 1,
            "through exit": 1,
            "right-turn exit": 1,
        },
        "3_3_2_likely_long_queue_lane": {
            "left-turn lane": 1,
            "through lane": 1,
            "right-turn lane": 1,
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
        "4_1_2_approach_motion_status": {
            "mostly moving": 1,
            "mostly stopped": 1,
            "mixed movement": 1,
        },
        "4_2_2_notable_abnormal": {
            "abnormal_proximity": 1,
            "speeding": 1,
            "crosswalk_blocking": 1,
            "lingering_pedestrian": 1,
            "stopline_overrun": 1,
            "wrong_way_two_wheeler": 1,
            "queue_spillback": 1,
        },
        "4_3_1_intersection_action": {
            "QUEUE_MANAGEMENT": 1,
            "CONFLICT_SUPPRESSION": 1,
            "FLOW_STABLE": 1,
            "FLOW_CALMING": 1,
        },
        "4_3_2_approach_action": {
            "SIDE_CLEARANCE_PROTECTION": 1,
            "SIDE_CROSSING_AWARENESS": 1,
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
        "1_1_1_lane_first_object_type": 0.5,
        "1_1_2_front_neighbor_type": 0.2,
        "1_1_3_approach_vru_exists": 0.5,
        "1_1_4_approach_type_count": 1.0,
        "1_2_1_size_bucket": 0.05,
        "1_2_2_visibility": 0.05,
        "1_3_1_environment": 0.10,
        "1_3_2_vehicle_signal_state": 0.02,
        "2_1_1_stopline_distance": 1.0,
        "2_1_2_ped_to_far_edge": 1.0,
        "2_1_3_participant_distance": 1.0,
        "2_1_4_nearest_vehicle": 1.0,
        "2_2_1_ped_zone": 1.0,
        "2_2_2_lane_queue_count": 0.8,
        "2_2_3_stopline_back_5m_count": 1.0,
        "2_2_4_longest_queue_lane": 1.0,
        "2_2_5_crosswalk_blocking": 1.0,
        "3_1_1_current_motion_state": 0.1,
        "3_1_2_vehicle_maneuver": 0.6,
        "3_2_1_waypoints": 0.05,
        "3_2_2_future_region": 1.0,
        "3_3_1_safe_following": 0.5,
        "3_3_2_likely_long_queue_lane": 1.0,
        "3_4_1_pair_conflict": 1.0,
        "3_4_2_nearest_conflict_participant": 0.5,
        "3_4_3_primary_risk_subject": 1.0,
        "3_4_4_risk_pattern": 1.0,
        "4_1_1_overall_state": 0.5,
        "4_1_2_approach_motion_status": 0.5,
        "4_1_3_scene_summary": 0.4,
        "4_1_4_heaviest_traffic_approach": 0.5,
        "4_2_1_speeding_risk": 0.5,
        "4_2_2_notable_abnormal": 1.0,
        "4_3_1_intersection_action": 1.0,
        "4_3_2_approach_action": 0.5,
        "4_3_3_lane_action": 1.0,
        "4_3_4_object_action": 0.1,
    }
    TEMPLATE_FINAL_RATIO_EXEMPT_LABELS = {
        "1_1_1_lane_first_object_type": {
            "construction_vehicle",
        },
        "1_3_1_environment": {"cloudy", "rainy", ""},
        "3_1_1_current_motion_state": {"starting"},
        "3_2_2_future_region": {"right-turn exit"},
        "4_1_4_heaviest_traffic_approach": {"north", "south"},
        "4_2_2_notable_abnormal": {
            "crosswalk_blocking",
            "lingering_pedestrian",
            "wrong_way_two_wheeler",
        },
        "4_3_2_approach_action": {"SIDE_CROSSING_AWARENESS"},
    }
    TEMPLATE_FINAL_RATIO_INCLUDED_LABELS = {
        "1_1_2_front_neighbor_type": {"left", "right"},
        "1_3_2_vehicle_signal_state": {"red light", "green light", "yellow light"},
        "3_4_2_nearest_conflict_participant": {"car", "truck"},
        "4_1_2_approach_motion_status": {"mixed movement", "mostly moving"},
        "4_3_2_approach_action": {
            "SIDE_GENERAL_CAUTION",
            "SIDE_SPEED_MODERATION",
            "SIDE_CLEARANCE_PROTECTION",
        },
    }
    TEMPLATE_FINAL_RATIO_BY_LABEL = {
        "1_1_1_lane_first_object_type": {
            "car": 0.05,
            "truck": 0.10,
            "trailer": 0.50,
        },
        "1_1_2_front_neighbor_type": {
            "left": 0.40,
            "right": 0.40,
        },
        "1_1_4_approach_type_count": {
            "car": 0.10,
            "truck": 0.20,
        },
        "1_3_1_environment": {"sunny": 0.10},
        "1_3_2_vehicle_signal_state": {
            "red light": 0.04,
            "green light": 0.04,
            "yellow light": 0.30,
        },
        "2_1_1_stopline_distance": {
            "trailer": 0.20,
        },
        "2_1_3_participant_distance": {
            "bicycle-car": 0.30,
            "bus-car": 0.20,
            "car-car": 0.02,
            "car-golf cart": 0.20,
            "car-motorcycle": 0.20,
            "car-trailer": 0.10,
            "car-truck": 0.01,
            "car-van": 0.10,
            "golf cart-truck": 0.60,
            "truck-truck": 0.05,
            "truck-van": 0.10,
        },
        "2_2_3_stopline_back_5m_count": {
            "east": 0.50,
            "west": 0.50,
        },
        "3_4_2_nearest_conflict_participant": {
            "car": 0.10,
            "truck": 0.20,
        },
        "4_1_2_approach_motion_status": {
            "mixed movement": 0.10,
            "mostly moving": 0.10,
        },
        "4_3_2_approach_action": {
            "SIDE_GENERAL_CAUTION": 0.20,
            "SIDE_SPEED_MODERATION": 0.20,
            "SIDE_CLEARANCE_PROTECTION": 0.30,
        },
    }

    def _build_template_registry(self) -> Dict[str, V5TemplateSpec]:
        specs = [
            V5TemplateSpec("1_1_1_lane_first_object_type", "1_base_perception", "1_1_object_identify", "Lane Ranked Object Type", "What is the type of the {second} object in the {left-turn} lane on the {east} approach?", "It is a {car}.", ("3d_boxes", "lane_side", "tracking"), True),
            V5TemplateSpec("1_1_2_front_neighbor_type", "1_base_perception", "1_1_object_identify", "Relative Neighbor Type", "What is located {to the front-left of} the {first} {van} in the {left-turn} lane on the {east} approach?", "It is a {car}.", ("3d_boxes", "tracking"), True),
            V5TemplateSpec("1_1_3_approach_vru_exists", "1_base_perception", "1_1_object_identify", "Approach VRU Presence", "Does the {east} approach contain any vulnerable road user?", "Yes, there are {3} VRUs. / No.", ("3d_boxes",), True),
            V5TemplateSpec("1_1_4_approach_type_count", "1_base_perception", "1_1_object_identify", "Approach Type Count", "How many {type}s are currently on the {approach} approach?", "The east approach currently has {3} trucks.", ("3d_boxes",), True),
            V5TemplateSpec("1_2_1_size_bucket", "1_base_perception", "1_2_appearance_attributes", "Size Bucket", "Which size best matches the {first} {van} in the {left-turn} lane on the {east} approach?", "The {van} is classified as a {large-size} vehicle.", ("3d_boxes",), True),
            V5TemplateSpec("1_2_2_visibility", "1_base_perception", "1_2_appearance_attributes", "Visibility", "How visible is the {first} {van} in the {left-turn} lane on the {east} approach?", "The {van} is {fully visible}.", ("images", "visibility"), True),
            V5TemplateSpec("1_3_1_environment", "1_base_perception", "1_3_scene_conditions_and_signals", "Environment", "Please describe the current environmental conditions in the scene.", "The scene appears to be {sunny}, {daytime}, with {strong sun glare} in the {east} view.", ("scene_metadata",), True),
            V5TemplateSpec("1_3_2_vehicle_signal_state", "1_base_perception", "1_3_scene_conditions_and_signals", "Vehicle Signal State", "Please describe the signal state for {movement} traffic on the {approach} approach.", "The signal is currently {green arrow}.", ("signals", "images"), True),
            V5TemplateSpec("2_1_1_stopline_distance", "2_spatial_reasoning", "2_1_geometric_localization", "Distance to Stop Line", "How far is the {first} {van} in the {left-turn} lane on the {east} approach from its relevant stop line?", "The {van} is {3.2 m} from the stop line.", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_1_2_ped_to_far_edge", "2_spatial_reasoning", "2_1_geometric_localization", "Pedestrian to Crossing Exit Area", "How far is the {nearest} pedestrian on the {east} crosswalk from the crossing exit area?", "The pedestrian is {6.4 m} from the exit area.", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("2_1_3_participant_distance", "2_spatial_reasoning", "2_1_geometric_localization", "Participant Pair Distance", "What is the distance between the {first} {van} in the {left-turn} lane on the {east} approach and the {second} {bus} in the {right-turn} lane on the {west} approach?", "The {van} is {2.4 m} from the {bus}.", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_1_4_nearest_vehicle", "2_spatial_reasoning", "2_1_geometric_localization", "Nearest Vehicle", "Please identify the vehicle nearest to the {first} {van} in the {left-turn} lane on the {east} approach.", "It is a {car}, {3.1 m} away.", ("3d_boxes",), True),
            V5TemplateSpec("2_2_1_ped_zone", "2_spatial_reasoning", "2_2_topological_relations", "Pedestrian Zone", "Which area of the crosswalk is the {nearest} pedestrian on the {east} crosswalk currently in?", "The pedestrian is currently in the {entry zone/waiting zone/within the crosswalk}.", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("2_2_2_lane_queue_count", "2_spatial_reasoning", "2_2_topological_relations", "Lane Queue Count", "How many queued vehicles are currently in the {north} {left-turn} lane?", "There are {4} queued vehicles in the {north} {left-turn} lane.", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_2_3_stopline_back_5m_count", "2_spatial_reasoning", "2_2_topological_relations", "Vehicles Within 5m Behind Stop Line", "How many vehicles are currently within 5 m behind the {east} stop line?", "There are {3} vehicles within 5 m behind the {east} stop line.", ("3d_boxes",), True),
            V5TemplateSpec("2_2_4_longest_queue_lane", "2_spatial_reasoning", "2_2_topological_relations", "Longest Queue Lane", "Which lane currently has the longest queue right now?", "The {through lane} on the {north} approach currently has the longest queue.", ("3d_boxes", "lane_side"), True),
            V5TemplateSpec("2_2_5_crosswalk_blocking", "2_spatial_reasoning", "2_2_topological_relations", "Crosswalk Blocking", "Is any vehicle blocking the crosswalk?", "Yes, a {truck} is currently blocking the {east} crosswalk. / No.", ("3d_boxes", "crosswalk_geometry"), True),
            V5TemplateSpec("3_1_1_current_motion_state", "3_temporal_reasoning", "3_1_motion_state_recognition", "Current Motion State", "Please describe the current motion state of the {first} {van} in the {left-turn} lane on the {east} approach.", "The {van} is {braking} at {2.0} m/s and {decelerating} at {-1.2} m/s^2.", ("history_tracks",), True),
            V5TemplateSpec("3_1_2_vehicle_maneuver", "3_temporal_reasoning", "3_1_motion_state_recognition", "Vehicle Maneuver", "What maneuver is the {first} {van} in the {left-turn} lane on the {east} approach most likely executing?", "The {van} is making a {left turn/straight/right turn/lane change/stop-and-wait}.", ("history_tracks", "future_tracks"), True),
            V5TemplateSpec("3_2_1_waypoints", "3_temporal_reasoning", "3_2_trajectory_trend_judgment", "Trajectory Prediction", "Please predict the short-term future trajectory for the {first} {van} in the {left-turn} lane on the {east} approach.", "Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)", ("future_tracks",), True),
            V5TemplateSpec("3_2_2_future_region", "3_temporal_reasoning", "3_2_trajectory_trend_judgment", "Future Region", "Which region is the {first} {van} in the {left-turn} lane on the {east} approach most likely to enter within the next 3 seconds?", "The {van} is likely to move into the {intersection center/before stop line/left-turn exit/through exit/right-turn exit}.", ("future_tracks", "lane_side"), True),
            V5TemplateSpec("3_3_1_safe_following", "3_temporal_reasoning", "3_3_following_and_queue_dynamics", "Safe Following", "Are the first and second vehicles in the {through} lane on the {east} approach maintaining a safe following gap?", "Yes, the following distance is currently safe, about {3.0} m. / No, the following distance is currently too short, about {1.0} m.", ("history_tracks", "lane_side"), True),
            V5TemplateSpec("3_3_2_likely_long_queue_lane", "3_temporal_reasoning", "3_3_following_and_queue_dynamics", "Likely Long Queue Lane", "Which lane is most likely to form a long queue soon?", "The {through} lane on the {east} approach is most likely to form a long queue because it already contains {3} stopped vehicles.", ("history_tracks", "lane_side"), True),
            V5TemplateSpec("3_4_1_pair_conflict", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Pair Conflict", "Is there a potential conflict between the {first} {van} in the {left-turn} lane on the {east} approach and the {first} {bus} in the {left-turn} lane on the {north} approach right now?", "Yes, the queried pair currently shows a potential conflict. / No.", ("future_tracks",), True),
            V5TemplateSpec("3_4_2_nearest_conflict_participant", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Nearest Conflict Participant", "Which participant is most likely to conflict with the {first} {van} in the {left-turn} lane on the {east} approach?", "The most probable participant is the car in the {left-turn} lane on the {east} approach / the car in the center of the intersection.", ("future_tracks",), True),
            V5TemplateSpec("3_4_3_primary_risk_subject", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Primary Risk Subject", "Please identify the key participant associated with the major potential safety hazards.", "The primary risk subject is the car in the {left-turn} lane on the {east} approach / the car in the center of the intersection because of {path crossing/proximity/vru_conflict/overspeed/lane_change_conflict}.", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("3_4_4_risk_pattern", "3_temporal_reasoning", "3_4_potential_conflict_detection", "Risk Interaction Pattern", "What is the dominant conflict interaction pattern in the current scene?", "The dominant conflict pattern is {rear-end/left-turn_pedestrian/right-turn_two-wheeler/straight_lane_change} on the {east} approach.", ("future_tracks",), True),
            V5TemplateSpec("4_1_1_overall_state", "4_scene_understanding", "4_1_overall_intersection_state", "Overall Traffic Condition", "What is the overall traffic condition of the intersection?", "The intersection is currently {moderately congested/free-flowing/heavily congested/light traffic/slightly congested}, with only {2} of {9} vehicles moving.", ("history_tracks", "3d_boxes"), True),
            V5TemplateSpec("4_1_2_approach_motion_status", "4_scene_understanding", "4_1_overall_intersection_state", "Approach Motion Status", "Please describe the motion status of traffic participants on the {east} approach of the intersection.", "The east approach is {mostly moving/mixed movement/mostly stopped}, with {5} vehicles moving and {1} stopped.", ("3d_boxes", "history_tracks"), True),
            V5TemplateSpec("4_1_3_scene_summary", "4_scene_understanding", "4_1_overall_intersection_state", "Scene Summary", "Provide a brief summary of the current intersection scene.", "two-sentence scene summary", ("3d_boxes", "history_tracks"), True),
            V5TemplateSpec("4_1_4_heaviest_traffic_approach", "4_scene_understanding", "4_1_overall_intersection_state", "Heaviest Traffic Approach", "Which approach to the intersection has the heaviest traffic?", "The {east} approach is the busiest, with {6} traffic participants.", ("3d_boxes",), True),
            V5TemplateSpec("4_2_1_speeding_risk", "4_scene_understanding", "4_2_abnormal_events", "Speeding Risk", "Is there still a risk of speeding at the intersection?", "Yes. A {car} on the {north} approach appears to be traveling at high speed at about {22.4} m/s. / No.", ("future_tracks",), True),
            V5TemplateSpec("4_2_2_notable_abnormal", "4_scene_understanding", "4_2_abnormal_events", "Most Notable Abnormal Event", "What is the most notable current abnormal event in the scene?", "A high-risk abnormal {proximity interaction/crosswalk_blocking/lingering_pedestrian/speeding/stopline_overrun/wrong_way_two_wheeler} exists on the {east} approach.", ("future_tracks", "history_tracks", "crosswalk_geometry"), True),
            V5TemplateSpec("4_3_1_intersection_action", "4_scene_understanding", "4_3_planning_guidance", "Intersection Guidance", "How should traffic proceed through the intersection right now?", "Traffic should proceed conservatively and give extra priority to suppressing risky interactions, especially around vulnerable or conflicted movement.", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_2_approach_action", "4_scene_understanding", "4_3_planning_guidance", "Approach Guidance", "What precaution should traffic take on the west approach of the intersection right now?", "Traffic should pay close attention to crossing movement and remain ready to yield if needed.", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_3_lane_action", "4_scene_understanding", "4_3_planning_guidance", "Lane Guidance", "How should traffic behave in the through lane on the north approach right now?", "Traffic should maintain orderly progression, remain in lane, and avoid unnecessary lane reorganization.", ("future_tracks", "history_tracks"), True),
            V5TemplateSpec("4_3_4_object_action", "4_scene_understanding", "4_3_planning_guidance", "Center Object Guidance", "What is the safest action for the highest-risk vehicle in the center of the intersection?", "That van should yield momentarily and allow the nearby interaction to clear before continuing.", ("future_tracks", "history_tracks"), True),
        ]
        return {spec.template_id: spec for spec in specs}

    def _build_sampling_policy_registry(self) -> Dict[str, Dict]:
        return {
            "1_1_1_lane_first_object_type": {"coverage_priority": ("object_type", "direction", "lane_function")},
            "1_1_2_front_neighbor_type": {"coverage_priority": ("rel_dir", "direction", "lane_function")},
            "1_1_3_approach_vru_exists": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "1_1_4_approach_type_count": {"coverage_priority": ("object_type", "direction")},
            "1_2_1_size_bucket": {"coverage_priority": ("size_bucket",)},
            "1_2_2_visibility": {"coverage_priority": ("visibility",), "placeholder": True},
            "1_3_1_environment": {"coverage_priority": ("weather", "time_of_day", "sun_glare"), "placeholder": True},
            "1_3_2_vehicle_signal_state": {"coverage_priority": ("signal_state", "movement", "direction"), "placeholder": True},
            "2_1_1_stopline_distance": {"coverage_priority": ("direction", "lane_function")},
            "2_1_2_ped_to_far_edge": {"coverage_priority": ("crosswalk",)},
            "2_1_3_participant_distance": {"coverage_priority": ("pair_type",)},
            "2_1_4_nearest_vehicle": {"coverage_priority": ("direction",)},
            "2_2_1_ped_zone": {"coverage_priority": ("zone", "crosswalk")},
            "2_2_2_lane_queue_count": {"coverage_priority": ("direction", "lane_function")},
            "2_2_3_stopline_back_5m_count": {"coverage_priority": ("direction",)},
            "2_2_4_longest_queue_lane": {"coverage_priority": ("direction", "lane_function")},
            "2_2_5_crosswalk_blocking": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "3_1_1_current_motion_state": {"coverage_priority": ("motion_state", "object_id", "direction")},
            "3_1_2_vehicle_maneuver": {"coverage_priority": ("maneuver", "object_id", "direction")},
            "3_2_1_waypoints": {"coverage_priority": ("object_id", "direction")},
            "3_2_2_future_region": {"coverage_priority": ("future_region", "object_id", "direction")},
            "3_3_1_safe_following": {"coverage_priority": ("pair_type",), "yes_no_ratio": (5, 5)},
            "3_3_2_likely_long_queue_lane": {"coverage_priority": ("lane_function", "direction")},
            "3_4_1_pair_conflict": {"coverage_priority": ("has_conflict", "pair_type"), "yes_no_ratio": (7, 3)},
            "3_4_2_nearest_conflict_participant": {"coverage_priority": ("object_id", "direction")},
            "3_4_3_primary_risk_subject": {"coverage_priority": ("risk_reason", "object_id")},
            "3_4_4_risk_pattern": {"coverage_priority": ("interaction_pattern",)},
            "4_1_1_overall_state": {"coverage_priority": ("overall_state",), "single_instance_global": True},
            "4_1_2_approach_motion_status": {"coverage_priority": ("direction",)},
            "4_1_3_scene_summary": {"single_instance_global": True, "fixed_cap": 1},
            "4_1_4_heaviest_traffic_approach": {"coverage_priority": ("dominant_side",)},
            "4_2_1_speeding_risk": {"coverage_priority": ("direction",), "yes_no_ratio": (7, 3)},
            "4_2_2_notable_abnormal": {"coverage_priority": ("notable_abnormal",)},
            "4_3_1_intersection_action": {"coverage_priority": ("action_state",), "single_instance_global": True},
            "4_3_2_approach_action": {"coverage_priority": ("action_state", "direction")},
            "4_3_3_lane_action": {"coverage_priority": ("action_state", "lane_function", "direction")},
            "4_3_4_object_action": {"coverage_priority": ("action_state", "object_id")},
        }

    def _image_only_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.\n\n"
            "You are given four synchronized camera images from the same timestamp:\n"
            "- north_image\n"
            "- south_image\n"
            "- east_image\n"
            "- west_image\n\n"
            "Use only the four images as evidence in image-only mode.\n"
            "Do not hallucinate unsupported objects, states, or interactions.\n\n"
            "Natural selector rules:\n"
            "- first / second inside one approach lane are ranked by distance to that approach's relevant stop line.\n"
            "- typed vehicle selectors such as first van or second bus mean rank within the same approach, same lane, and same type.\n"
            "- relative-direction phrases such as in front of, behind, to the left of, to the right of, and the four diagonal directions use the referenced object's own heading frame, not the global north-up frame.\n"
            "- nearest pedestrian on a named crosswalk is the one whose current position is closest to the intersection center.\n"
            "- center-region object guidance targets the highest-risk vehicle currently in the center.\n"
            "- Safe-following questions use the automatically selected front and following vehicles in the queried lane.\n\n"
            "Spatial wording rules:\n"
            "- Use approach names north, south, east, west, and center of the intersection consistently.\n"
            "- Use lane names left-turn lane, through lane, and right-turn lane.\n"
            "- Use crosswalk, entry zone, exit zone, and waiting zone consistently.\n\n"
            "Answering rules:\n"
            "- Answer only the queried task.\n"
            "- Keep the answer concise and grounded in the visible scene.\n"
            "- Keep numbers and units when the task is about counts, distances, speeds, or trajectories.\n"
        )

    def _pointcloud_plus_image_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.\n\n"
            "You are given synchronized multimodal evidence from the same timestamp:\n"
            "- four camera images: north_image, south_image, east_image, west_image\n"
            "- global intersection point cloud: point_cloud\n\n"
            "Use the point cloud primarily for geometry, global position, and metric distances.\n"
            "Use the images primarily for appearance, visibility, and visible signal evidence.\n\n"
            "Natural selector rules:\n"
            "- first / second inside one approach lane are ranked by distance to that approach's relevant stop line.\n"
            "- typed vehicle selectors such as first van or second bus mean rank within the same approach, same lane, and same type.\n"
            "- relative-direction phrases such as in front of, behind, to the left of, to the right of, and the four diagonal directions use the referenced object's own heading frame.\n"
            "- nearest pedestrian on a named crosswalk is the one whose current position is closest to the intersection center.\n"
            "- center-region object guidance targets the highest-risk vehicle currently in the center.\n"
            "- Safe-following questions use the automatically selected front and following vehicles in the queried lane.\n\n"
            "Spatial wording rules:\n"
            "- Use approach names north, south, east, west, and center of the intersection consistently.\n"
            "- Use lane names left-turn lane, through lane, and right-turn lane.\n"
            "- Distances between participants are defined on the ground plane as the minimum distance between their 3D box footprints, ignoring z.\n"
            "- Relative directions such as front, behind, left, and right are defined by the referenced object's own heading.\n\n"
            "Answering rules:\n"
            "- Answer only the queried task.\n"
            "- Keep the answer concise and grounded in the provided multimodal evidence.\n"
            "- Keep numbers and units when the task is about counts, distances, speeds, or trajectories.\n"
        )

    def _strict_answer_schemas(self) -> Dict[str, str]:
        return {
            "3_2_1_waypoints": "Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)",
        }

    def _subtemplate_patches(self) -> Dict[str, str]:
        return {
            "1_1_1_lane_first_object_type": "Answer only with the fine-grained type of the selected ranked lane object.",
            "1_1_2_front_neighbor_type": "Use the referenced vehicle's own heading frame to interpret the queried relative direction among front, behind, left, right, front-left, rear-left, front-right, and rear-right, and answer only with the target object's type.",
            "1_1_3_approach_vru_exists": "Treat VRUs as pedestrians, bicycles, motorcycles, and golf carts only. If yes, include the count concisely.",
            "1_1_4_approach_type_count": "Return only the count for the queried fine type on that approach.",
            "1_2_1_size_bucket": "Judge the selected vehicle's overall 3D size and answer only with the size category.",
            "1_2_2_visibility": "Judge the selected vehicle's visibility level from the images only.",
            "1_3_1_environment": "Answer only with the available environment fields: weather, time of day, and visible strong sun glare.",
            "1_3_2_vehicle_signal_state": "Use only directly visible signal evidence and answer only with the normalized signal-state label.",
            "2_1_1_stopline_distance": "Return only the selected vehicle's distance to its relevant stop line.",
            "2_1_2_ped_to_far_edge": "Use the selected crosswalk pedestrian's current crossing direction and answer only with the distance to the exit area.",
            "2_1_3_participant_distance": "Answer only with the distance between the two queried participants.",
            "2_1_4_nearest_vehicle": "Answer only with the nearest vehicle type and the distance.",
            "2_2_1_ped_zone": "Answer only with the normalized crosswalk-zone label.",
            "2_2_2_lane_queue_count": "Count queued vehicles only in the queried approach-lane pair.",
            "2_2_3_stopline_back_5m_count": "Count vehicles only within the 5 m strip behind the named stop line.",
            "2_2_4_longest_queue_lane": "Return only the winning lane and approach.",
            "2_2_5_crosswalk_blocking": "If yes, mention the blocking vehicle type and crosswalk briefly; otherwise keep the answer brief.",
            "3_1_1_current_motion_state": "Report the selected object's current motion state and speed. Only starting and braking should include acceleration wording and m/s^2.",
            "3_1_2_vehicle_maneuver": "Answer only with the dominant current maneuver of the selected vehicle.",
            "3_2_1_waypoints": "Follow the exact Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4) format. Each point is an XY offset in meters relative to the selected object's current position.",
            "3_2_2_future_region": "Answer only with the most likely next region for the selected vehicle.",
            "3_3_1_safe_following": "The queried pair is the automatically selected front and following vehicles in the named lane. Answer only with the safety judgment and the current distance.",
            "3_3_2_likely_long_queue_lane": "Return the winning lane, approach, and concise queue evidence only.",
            "3_4_1_pair_conflict": "Judge only the queried pair. Do not replace it with another conflict pair in the scene.",
            "3_4_2_nearest_conflict_participant": "Return only the most likely conflict partner using a natural location phrase.",
            "3_4_3_primary_risk_subject": "Identify one primary risk subject and one dominant risk reason only.",
            "3_4_4_risk_pattern": "Return only the dominant conflict interaction pattern and its associated approach or center region.",
            "4_1_1_overall_state": "Answer only with the overall traffic condition label and the moving-versus-total vehicle count.",
            "4_1_2_approach_motion_status": "Answer only with the queried approach's motion label and moving/stopped counts.",
            "4_1_3_scene_summary": "Keep the summary to two short sentences.",
            "4_1_4_heaviest_traffic_approach": "Return only the busiest approach and its participant count.",
            "4_2_1_speeding_risk": "If yes, mention the risky object type, approach, and speed; otherwise answer No.",
            "4_2_2_notable_abnormal": "Answer only with the dominant abnormal-event label and its main location.",
            "4_3_1_intersection_action": "Focus on the recommended action itself and keep it concise.",
            "4_3_2_approach_action": "Keep the guidance local to the queried approach.",
            "4_3_3_lane_action": "Keep the guidance local to the queried approach-lane pair.",
            "4_3_4_object_action": "Focus only on the highest-risk center vehicle and what it should do now.",
        }

    def _canonical_lanes(self) -> Tuple[str, str, str]:
        return (self.LEFT_TURN_LANE, self.STRAIGHT_LANE, self.RIGHT_TURN_LANE)

    def _canonical_lane_objects(self, objects: List[Dict]) -> List[Dict]:
        return [obj for obj in objects if obj["side"] in {"north", "south", "east", "west"} and obj["lane"] in self._canonical_lanes()]

    def _ranked_lane_non_ped_objects(
        self,
        side: str,
        lane: str,
        objects: List[Dict],
        obj_type: Optional[str] = None,
    ) -> List[Dict]:
        return [
            obj
            for obj in self._ranked_lane_objects(side, lane, objects, obj_type=obj_type)
            if obj["type"] != "pedestrian"
        ]

    def _canonical_lane_non_ped_objects(self, objects: List[Dict]) -> List[Dict]:
        participants: List[Dict] = []
        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                participants.extend(self._ranked_lane_non_ped_objects(side, lane, objects))
        return participants

    def _ordinal_label(self, rank: int) -> str:
        return {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(rank, f"{rank}th")

    def _lane_rank_sort_key(self, obj: Dict) -> Tuple[float, Tuple[int, str]]:
        distance = self._distance_to_stopline(obj)
        fallback = self._center_dist(obj)
        key_distance = float(distance) if distance is not None else 1e6 + fallback
        return (key_distance, self._ref_id_sort_key(obj))

    def _ranked_lane_objects(
        self,
        side: str,
        lane: str,
        objects: List[Dict],
        obj_type: Optional[str] = None,
        non_vru_only: bool = False,
    ) -> List[Dict]:
        candidates = []
        for obj in objects:
            if obj["side"] != side or obj.get("lane") != lane:
                continue
            if obj_type is not None and obj["type"] != obj_type:
                continue
            if non_vru_only and obj["type"] in self.VRU_TYPES:
                continue
            candidates.append(obj)
        return sorted(candidates, key=self._lane_rank_sort_key)

    def _object_rank_in_lane(self, target: Dict, objects: List[Dict], type_specific: bool) -> Optional[int]:
        obj_type = target["type"] if type_specific else None
        ranked = self._ranked_lane_objects(target["side"], target["lane"], objects, obj_type=obj_type)
        for index, obj in enumerate(ranked, 1):
            if obj["id"] == target["id"]:
                return index
        return None

    def _lane_selector_phrase(
        self,
        side: str,
        lane: str,
        rank: Optional[int] = None,
        obj_type: Optional[str] = None,
        noun: str = "object",
    ) -> str:
        rank_phrase = f"{self._ordinal_label(rank)} " if rank is not None else ""
        noun_phrase = obj_type if obj_type is not None else noun
        return f"the {rank_phrase}{noun_phrase} in the {lane} on the {side} approach"

    def _typed_lane_selector_for_object(self, obj: Dict, objects: List[Dict]) -> str:
        rank = self._object_rank_in_lane(obj, objects, type_specific=True) or 1
        return self._lane_selector_phrase(obj["side"], obj["lane"], rank=rank, obj_type=obj["type"])

    def _generic_lane_selector_for_object(self, obj: Dict, objects: List[Dict]) -> str:
        rank = self._object_rank_in_lane(obj, objects, type_specific=False) or 1
        return self._lane_selector_phrase(obj["side"], obj["lane"], rank=rank)

    def _natural_object_location_phrase(self, obj: Dict) -> str:
        if obj["side"] == "center":
            return "in the center of the intersection"
        lane = obj.get("lane")
        if lane in self._canonical_lanes():
            return f"in the {lane} on the {obj['side']} approach"
        return f"on the {obj['side']} approach"

    def _natural_object_phrase(self, obj: Dict, capitalized: bool = False) -> str:
        phrase = f"the {obj['type']} {self._natural_object_location_phrase(obj)}"
        if capitalized:
            return phrase[:1].upper() + phrase[1:]
        return phrase

    def _answer_subject(self, obj: Dict) -> str:
        return f"The {obj['type']}"

    def _pair_type_key(self, a: Dict, b: Dict) -> str:
        first, second = sorted((a["type"], b["type"]))
        return f"{first}-{second}"

    def _best_relative_neighbors(self, obj: Dict, objects: List[Dict]) -> Dict[str, Dict]:
        sector_best: Dict[str, Tuple[float, float, Tuple[int, str], Dict]] = {}
        for other in objects:
            if other["id"] == obj["id"]:
                continue
            center_distance = self._center_distance_between(obj, other)
            if center_distance > 4.0:
                continue
            rel_dir, angle_delta = self._relative_direction_info(obj, other)
            ref_key = self._ref_id_sort_key(other)
            key = (angle_delta, center_distance, ref_key)
            previous = sector_best.get(rel_dir)
            if previous is None or key < previous[:3]:
                sector_best[rel_dir] = (angle_delta, center_distance, ref_key, other)
        return {rel_dir: item[3] for rel_dir, item in sector_best.items()}

    def _nearest_pedestrian_on_crosswalk(self, crosswalk: str, objects: List[Dict]) -> Optional[Dict]:
        candidates = []
        for obj in objects:
            if obj["type"] != "pedestrian":
                continue
            crosswalk_name, region = self._ped_crosswalk_region(obj)
            if crosswalk_name != crosswalk or region == "none":
                continue
            candidates.append(obj)
        if not candidates:
            return None
        return min(candidates, key=lambda obj: (self._center_dist(obj), self._lane_rank_sort_key(obj)))

    def _stopline_back_10m_count(self, side: str, objects: List[Dict]) -> int:
        return self._stopline_back_5m_count(side, objects)

    def _highest_risk_center_vehicle(self, objects: List[Dict]) -> Optional[Dict]:
        candidates = [obj for obj in objects if obj["side"] == "center" and obj["type"] != "pedestrian"]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda obj: (
                self._planning_object_stats(obj, objects)["I_c"],
                -self._planning_object_stats(obj, objects)["d_min"],
                self._speed(obj),
                -self._ref_id_sort_key(obj)[0],
            ),
        )

    def _center_location_phrase(self, region: str) -> str:
        if region == "center":
            return "in the center of the intersection"
        return f"on the {region} approach"

    def _visibility_answer(self, obj: Dict, visibility: str) -> str:
        return f"The {obj['type']} is {visibility}."

    def _signal_answer(self, state: str) -> str:
        return f"The signal is currently {state}."

    def _current_motion_state_answer(self, obj: Dict, motion_state: str, speed_mps: float, accel_state: Optional[str], acceleration: Optional[float]) -> str:
        base = f"The {obj['type']} is {motion_state} at {speed_mps:.1f} m/s"
        if obj["type"] != "pedestrian" and motion_state in {"starting", "braking"} and accel_state in {"accelerating", "decelerating"} and acceleration is not None:
            return f"{base} and {accel_state} at {acceleration:.1f} m/s^2."
        return f"{base}."

    def _maneuver_answer(self, obj: Dict, maneuver: str) -> str:
        phrase = {
            "left turn": "making a left turn",
            "straight": "going straight",
            "right turn": "making a right turn",
            "lane change": "making a lane change",
            "stop-and-wait": "stop-and-wait",
        }.get(maneuver, maneuver)
        if phrase == "stop-and-wait":
            return f"The {obj['type']} is executing a stop-and-wait."
        return f"The {obj['type']} is {phrase}."

    def _future_region_answer(self, obj: Dict, future_region: str) -> str:
        return f"The {obj['type']} is likely to {self._future_region_phrase(future_region)}."

    def _safe_following_answer(self, safe: bool, pair_distance: float) -> str:
        if safe:
            return f"Yes, the following distance is currently safe, about {pair_distance:.1f} m."
        return f"No, the following distance is currently too short, about {pair_distance:.1f} m."

    def _pair_conflict_answer(self, a: Dict, b: Dict, has_conflict: bool) -> str:
        if has_conflict:
            return f"Yes, the {a['type']} and the {b['type']} currently show a potential conflict."
        return "No."

    def _most_likely_conflict_partner_answer(self, partner: Dict) -> str:
        return f"The most probable participant is {self._natural_object_phrase(partner)}."

    def _primary_risk_subject_answer(self, subject: Dict, reason: str) -> str:
        return f"The primary risk subject is {self._natural_object_phrase(subject)} because of {self._turn_reason_phrase(reason)}."

    def _risk_pattern_answer(self, pattern: str, side: str) -> str:
        if side == "center":
            return f"The dominant conflict pattern is {pattern} in the center of the intersection."
        return f"The dominant conflict pattern is {pattern} on the {side} approach."

    def _notable_abnormal_answer(self, label: str, region: str) -> str:
        label_text = {
            "abnormal_proximity": "proximity interaction",
            "crosswalk_blocking": "crosswalk_blocking",
            "lingering_pedestrian": "lingering_pedestrian",
            "speeding": "speeding",
            "stopline_overrun": "stopline_overrun",
            "wrong_way_two_wheeler": "wrong_way_two_wheeler",
            "queue_spillback": "queue_spillback",
        }.get(label, label)
        return f"A high-risk abnormal {label_text} exists {self._center_location_phrase(region)}."

    def _intersection_safe_action_answer(self, objects: List[Dict]) -> Tuple[str, str]:
        state = self._intersection_guidance_state(objects)
        mapping = {
            "FLOW_STABLE": "Traffic should continue orderly progression while maintaining normal caution.",
            "FLOW_CALMING": "Traffic should proceed more calmly with moderated speed and extra spacing.",
            "QUEUE_MANAGEMENT": "Traffic should proceed in a tightly managed and orderly way to prevent queue growth.",
            "CONFLICT_SUPPRESSION": "Traffic should proceed conservatively and give extra priority to suppressing risky interactions, especially around vulnerable or conflicted movement.",
        }
        return mapping[state], state

    def _side_safe_action_answer(self, side: str, objects: List[Dict]) -> Tuple[str, str]:
        state = self._side_guidance_state(side, objects)
        mapping = {
            "SIDE_CLEARANCE_PROTECTION": "Traffic should keep safer spacing and protect local clearance around nearby participants.",
            "SIDE_CROSSING_AWARENESS": "Traffic should pay close attention to crossing movement and remain ready to yield if needed.",
            "SIDE_SPEED_MODERATION": "Traffic should moderate speed and maintain safer local spacing.",
            "SIDE_QUEUE_STABILIZATION": "Traffic should stabilize queue movement and avoid unnecessary disruption.",
            "SIDE_GENERAL_CAUTION": "Traffic should proceed cautiously and remain orderly.",
        }
        return mapping[state], state

    def _lane_safe_action_answer(self, side: str, lane: str, objects: List[Dict], all_objects: List[Dict]) -> Tuple[str, str]:
        state = self._lane_guidance_state(side, lane, objects, all_objects)
        mapping = {
            "LANE_CLEARANCE_MAINTENANCE": "Traffic should maintain local clearance and avoid tight conflicts in this lane.",
            "LANE_PREPARE_TO_STOP": "Traffic should prepare to stop and avoid pressing forward in this lane.",
            "LANE_QUEUE_PRESERVATION": "Traffic should preserve queue order and avoid unnecessary lane reorganization.",
            "LANE_SPEED_REDUCTION": "Traffic should reduce speed and proceed more conservatively in this lane.",
            "LANE_GENERAL_ORDER": "Traffic should maintain orderly progression, remain in lane, and avoid unnecessary lane reorganization.",
        }
        return mapping[state], state

    def _object_safe_action_answer(self, obj: Dict, objects: List[Dict]) -> Tuple[str, str]:
        state = self._object_guidance_state(obj, objects)
        mapping = {
            "OBJECT_YIELD_NOW": f"That {obj['type']} should yield momentarily and allow the nearby interaction to clear before continuing.",
            "OBJECT_PREPARE_TO_STOP": f"That {obj['type']} should prepare to stop and wait for nearby movement to clear.",
            "OBJECT_SLOW_DOWN": f"That {obj['type']} should slow down and create more local safety margin.",
            "OBJECT_PROCEED_CAUTIOUSLY": f"That {obj['type']} should proceed cautiously while monitoring nearby participants.",
        }
        return mapping[state], state

    def _basic_perception_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        frame = self.token_to_frame.get(objects[0]["frame_token"]) if objects else None
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
                    "1_1_3_approach_vru_exists",
                    f"Does the {side} approach contain any vulnerable road user?",
                    exists_answer,
                    float(vru_count) if exists else 0.1,
                    {"side": side, "exists": exists, "count": int(vru_count)},
                    sample_meta={"direction": side, "exists": exists, "count": int(vru_count)},
                )
            )
            for obj_type in self.FINE_TYPES:
                count_value = int(counts.get(obj_type, 0))
                if count_value <= 0:
                    continue
                qas.append(
                    self._v5_qa(
                        "1_1_4_approach_type_count",
                        f"How many {obj_type}s are currently on the {side} approach?",
                        f"The {side} approach currently has {count_value} {self._pluralize(obj_type, count_value)}.",
                        float(count_value),
                        {"side": side, "object_type": obj_type, "count": count_value},
                        sample_meta={"direction": side, "object_type": obj_type},
                    )
                )

            for lane in self._canonical_lanes():
                ranked_all = self._ranked_lane_objects(side, lane, objects)
                for rank, obj in enumerate(ranked_all, 1):
                    object_type_bucket = (
                        "construction_vehicle" if obj["type"] == "construction vehicle" else obj["type"]
                    )
                    qas.append(
                        self._v5_qa(
                            "1_1_1_lane_first_object_type",
                            f"What is the type of {self._generic_lane_selector_for_object(obj, objects)}?",
                            f"It is a {obj['type']}.",
                            1.0,
                            {
                                **self._object_targets(obj, include_type=True, include_side=True),
                                "object_type": object_type_bucket,
                                "lane_function": lane,
                            },
                            sample_meta={
                                "object_type": object_type_bucket,
                                "direction": side,
                                "lane_function": lane,
                                "object_id": obj["ref_id"],
                                "rank": rank,
                                "ordinal": self._ordinal_label(rank),
                            },
                        )
                    )
                for vehicle in self._ranked_lane_non_ped_objects(side, lane, objects):
                    relative_neighbors = self._best_relative_neighbors(vehicle, objects)
                    for rel_dir in self.RELATIVE_DIRECTION_QUERY_PHRASES:
                        target_obj = relative_neighbors.get(rel_dir)
                        if target_obj is None:
                            continue
                        qas.append(
                            self._v5_qa(
                                "1_1_2_front_neighbor_type",
                                f"What is located {self.RELATIVE_DIRECTION_QUERY_PHRASES[rel_dir]} {self._typed_lane_selector_for_object(vehicle, objects)}?",
                                f"It is a {target_obj['type']}.",
                                0.8,
                                {
                                    **self._object_targets(vehicle, include_type=True, include_side=True),
                                    "lane_function": lane,
                                    "rel_dir": rel_dir,
                                    "target_object_type": target_obj["type"],
                                    **self._prefixed_object_targets(target_obj, "target_object", include_type=False),
                                },
                                sample_meta={"rel_dir": rel_dir, "direction": side, "lane_function": lane, "object_id": vehicle["ref_id"]},
                            )
                        )
                    size_bucket = self._size_bucket(vehicle)
                    qas.append(
                        self._v5_qa(
                            "1_2_1_size_bucket",
                            f"Which size best matches {self._typed_lane_selector_for_object(vehicle, objects)}?",
                            f"The {vehicle['type']} is classified as a {size_bucket}-size vehicle.",
                            0.5,
                            {**self._object_targets(vehicle, include_type=True, include_side=True), "lane_function": lane, "size_bucket": size_bucket},
                            sample_meta={"size_bucket": size_bucket, "direction": side, "lane_function": lane, "object_id": vehicle["ref_id"]},
                        )
                    )
        qas.extend(self._annotated_or_placeholder_basic_perception_qas(frame, objects))
        return qas

    def _annotated_or_placeholder_basic_perception_qas(self, frame: Optional[Dict], objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        annotations = self._frame_manual_annotations(frame)
        visibility_map = annotations.get("visibility_by_track_id") if isinstance(annotations.get("visibility_by_track_id"), dict) else {}
        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                for obj in self._ranked_lane_non_ped_objects(side, lane, objects):
                    visibility = visibility_map.get(str(obj["id"]))
                    question = f"How visible is {self._typed_lane_selector_for_object(obj, objects)}?"
                    targets = {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane, "visibility": visibility}
                    sample_meta = {"visibility": visibility or "unknown", "object_id": obj["ref_id"], "direction": side, "lane_function": lane}
                    if visibility:
                        qas.append(self._v5_qa("1_2_2_visibility", question, self._visibility_answer(obj, visibility), 0.2, targets, sample_meta=sample_meta))
                    else:
                        qas.append(self._v5_qa("1_2_2_visibility", question, None, 0.0, targets, sample_meta=sample_meta, placeholder=True))

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
                    "1_3_1_environment",
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
                    "1_3_1_environment",
                    "Please describe the current environmental conditions in the scene.",
                    None,
                    0.0,
                    environment_targets,
                    sample_meta={"environment": "unknown", "weather": "unknown", "time_of_day": "unknown", "sun_glare": "unknown"},
                    placeholder=True,
                )
            )

        vehicle_signal_state = annotations.get("vehicle_signal_state") if isinstance(annotations.get("vehicle_signal_state"), dict) else {}
        for side in ("north", "south", "east", "west"):
            for movement in ("left-turn", "through"):
                state = None
                if isinstance(vehicle_signal_state.get(side), dict):
                    state = self._valid_vehicle_signal_state(side, vehicle_signal_state.get(side, {}).get(movement))
                question = f"Please describe the signal state for {movement} traffic on the {side} approach."
                targets = {"side": side, "movement": movement, "signal_state": state}
                sample_meta = {"direction": side, "movement": movement, "signal_state": state or f"{movement}_unknown"}
                if state:
                    qas.append(self._v5_qa("1_3_2_vehicle_signal_state", question, self._signal_answer(state), 0.1, targets, sample_meta=sample_meta))
                else:
                    qas.append(self._v5_qa("1_3_2_vehicle_signal_state", question, None, 0.0, targets, sample_meta=sample_meta, placeholder=True))
        return qas

    def _spatial_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        by_side, by_lane = self._split_by_side_and_lane(objects)
        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                for vehicle in self._ranked_lane_non_ped_objects(side, lane, objects):
                    distance = self._distance_to_stopline(vehicle)
                    if distance is not None:
                        qas.append(
                            self._v5_qa(
                                "2_1_1_stopline_distance",
                                f"How far is {self._typed_lane_selector_for_object(vehicle, objects)} from its relevant stop line?",
                                f"The {vehicle['type']} is {distance:.1f} m from the stop line.",
                                10.0 - distance,
                                {**self._object_targets(vehicle, include_type=True, include_side=True), "lane_function": lane, "distance_m": distance, "stopline_side": side},
                                sample_meta={"direction": side, "lane_function": lane, "object_type": vehicle["type"], "object_id": vehicle["ref_id"]},
                            )
                        )
                    nearest_vehicle = self._nearest(vehicle, [obj for obj in objects if obj["type"] != "pedestrian" and obj["id"] != vehicle["id"]])
                    if nearest_vehicle is not None:
                        dist = round(self._dist(vehicle, nearest_vehicle), 1)
                        qas.append(
                            self._v5_qa(
                                "2_1_4_nearest_vehicle",
                                f"Please identify the vehicle nearest to {self._typed_lane_selector_for_object(vehicle, objects)}.",
                                f"It is a {nearest_vehicle['type']}, {dist:.1f} m away.",
                                10.0 - dist,
                                {
                                    "focus_id": vehicle["ref_id"],
                                    "focus_raw_tracking_id": vehicle["id"],
                                    "focus_type": vehicle["type"],
                                    "side": vehicle["side"],
                                    "lane_function": lane,
                                    "vehicle_id": nearest_vehicle["ref_id"],
                                    "vehicle_raw_tracking_id": nearest_vehicle["id"],
                                    "vehicle_type": nearest_vehicle["type"],
                                    "distance_m": dist,
                                    **self._object_location_targets(nearest_vehicle, "vehicle"),
                                },
                                sample_meta={"direction": side, "object_id": vehicle["ref_id"]},
                            )
                        )
        lane_vehicles = self._canonical_lane_non_ped_objects(objects)
        for index, obj1 in enumerate(lane_vehicles):
            for obj2 in lane_vehicles[index + 1 :]:
                pair_type = self._pair_type_key(obj1, obj2)
                if pair_type == "trailer-truck":
                    continue
                dist = round(self._dist(obj1, obj2), 1)
                qas.append(
                    self._v5_qa(
                        "2_1_3_participant_distance",
                        f"What is the distance between {self._typed_lane_selector_for_object(obj1, objects)} and {self._typed_lane_selector_for_object(obj2, objects)}?",
                        f"The {obj1['type']} is {dist:.1f} m from the {obj2['type']}.",
                        10.0 - dist,
                        {
                            "obj1_id": obj1["ref_id"],
                            "obj1_raw_tracking_id": obj1["id"],
                            "obj2_id": obj2["ref_id"],
                            "obj2_raw_tracking_id": obj2["id"],
                            "obj1_type": obj1["type"],
                            "obj2_type": obj2["type"],
                            "distance_m": dist,
                            "pair_type": pair_type,
                            "obj1_side": obj1["side"],
                            "obj2_side": obj2["side"],
                            "obj1_lane_function": obj1["lane"],
                            "obj2_lane_function": obj2["lane"],
                        },
                        sample_meta={"pair_type": pair_type, "direction": obj1["side"], "object_id": obj1["ref_id"]},
                    )
                )
        for crosswalk in ("north", "south", "east", "west"):
            ped = self._nearest_pedestrian_on_crosswalk(crosswalk, objects)
            if ped is None:
                continue
            exit_distance = self._ped_crosswalk_exit_distance(ped, crosswalk)
            if exit_distance is not None:
                qas.append(
                    self._v5_qa(
                        "2_1_2_ped_to_far_edge",
                        f"How far is the nearest pedestrian on the {crosswalk} crosswalk from the crossing exit area?",
                        f"The pedestrian is {exit_distance:.1f} m from the exit area.",
                        5.0,
                        {"pedestrian_id": ped["ref_id"], "pedestrian_raw_tracking_id": ped["id"], "crosswalk": crosswalk, "distance_m": exit_distance, **self._object_location_targets(ped, "pedestrian")},
                        sample_meta={"crosswalk": crosswalk, "object_id": ped["ref_id"]},
                    )
                )
            crosswalk_name, region = self._ped_crosswalk_region(ped)
            if region != "none":
                ped_zone_value = "within the crosswalk" if region == "crosswalk" else region
                qas.append(
                    self._v5_qa(
                        "2_2_1_ped_zone",
                        f"Which area of the crosswalk is the nearest pedestrian on the {crosswalk} crosswalk currently in?",
                        f"The pedestrian is currently in the {ped_zone_value}.",
                        1.0,
                        {"pedestrian_id": ped["ref_id"], "pedestrian_raw_tracking_id": ped["id"], "crosswalk": crosswalk_name, "ped_zone": ped_zone_value, **self._object_location_targets(ped, "pedestrian")},
                        sample_meta={"zone": ped_zone_value, "crosswalk": crosswalk_name, "object_id": ped["ref_id"]},
                    )
                )

        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                queue_count = self._lane_queue_count(side, lane, objects)
                if queue_count > 0:
                    qas.append(
                        self._v5_qa(
                            "2_2_2_lane_queue_count",
                            f"How many queued vehicles are currently in the {side} {lane}?",
                            f"There are {queue_count} queued vehicles in the {side} {lane}.",
                            float(queue_count),
                            {"side": side, "count": queue_count, "queue_vehicle_count": queue_count, "lane_function": lane},
                            sample_meta={"direction": side, "lane_function": lane},
                        )
                    )
            count_5m = self._stopline_back_10m_count(side, objects)
            if count_5m > 0:
                qas.append(
                    self._v5_qa(
                        "2_2_3_stopline_back_5m_count",
                        f"How many vehicles are currently within 5 m behind the {side} stop line?",
                        f"There are {count_5m} vehicles within 5 m behind the {side} stop line.",
                        float(count_5m),
                        {"side": side, "count": count_5m, "vehicle_count": count_5m},
                        sample_meta={"direction": side},
                    )
                )
        blocking_side = None
        blocking_vehicle = None
        for side in ("north", "south", "east", "west"):
            candidate = self._vehicle_on_crosswalk(side, objects)
            if candidate is not None:
                blocking_side = side
                blocking_vehicle = candidate
                break
        blocked = blocking_vehicle is not None
        blocking_answer = f"Yes, a {blocking_vehicle['type']} is currently blocking the {blocking_side} crosswalk." if blocked else "No."
        qas.append(
            self._v5_qa(
                "2_2_5_crosswalk_blocking",
                "Is any vehicle blocking the crosswalk?",
                blocking_answer,
                1.0 if blocked else 0.1,
                {
                    "side": blocking_side,
                    "crosswalk_blocked": blocked,
                    "object_type": None if blocking_vehicle is None else blocking_vehicle["type"],
                    "blocking_vehicle_id": None if blocking_vehicle is None else blocking_vehicle["ref_id"],
                    "blocking_vehicle_raw_tracking_id": None if blocking_vehicle is None else blocking_vehicle["id"],
                },
                sample_meta={"direction": blocking_side or "none", "crosswalk_blocked": blocked, "object_type": None if blocking_vehicle is None else blocking_vehicle["type"]},
            )
        )
        canonical_by_lane = {
            (lane_side, lane): lane_objs
            for (lane_side, lane), lane_objs in by_lane.items()
            if lane_side in {"north", "south", "east", "west"} and lane in self._canonical_lanes()
        }
        global_side, global_lane = self._global_longest_queue_lane(canonical_by_lane)
        if global_side is not None and global_lane is not None:
            qas.append(
                self._v5_qa(
                    "2_2_4_longest_queue_lane",
                    "Which lane currently has the longest queue right now?",
                    f"The {global_lane} on the {global_side} approach currently has the longest queue.",
                    1.0,
                    {"lane_function": global_lane, "side": global_side},
                    sample_meta={"direction": global_side, "lane_function": global_lane},
                )
            )
        return qas

    def _temporal_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                for obj in self._ranked_lane_non_ped_objects(side, lane, objects):
                    motion_state = self._vehicle_motion_state_v5(obj)
                    speed_mps = round(self._speed(obj), 1)
                    accel_state = None
                    acceleration = None
                    if motion_state in {"starting", "braking"}:
                        accel_state = self._vehicle_accel_bucket(obj)
                        acceleration = self._vehicle_acceleration_value(obj, seconds=3.0)
                        if accel_state not in {"accelerating", "decelerating"} or acceleration is None:
                            accel_state = None
                            acceleration = None
                    qas.append(
                        self._v5_qa(
                            "3_1_1_current_motion_state",
                            f"Please describe the current motion state of {self._typed_lane_selector_for_object(obj, objects)}.",
                            self._current_motion_state_answer(obj, motion_state, speed_mps, accel_state, acceleration),
                            1.0,
                            {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane, "motion_state": motion_state, "speed": speed_mps, "accel_state": accel_state, "acceleration": acceleration},
                            sample_meta={"motion_state": motion_state, "object_id": obj["ref_id"], "direction": obj["side"]},
                        )
                    )
                    maneuver = self._vehicle_maneuver_v5(obj)
                    qas.append(
                        self._v5_qa(
                            "3_1_2_vehicle_maneuver",
                            f"What maneuver is {self._typed_lane_selector_for_object(obj, objects)} most likely executing?",
                            self._maneuver_answer(obj, maneuver),
                            1.0,
                            {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane, "maneuver": maneuver},
                            sample_meta={"maneuver": maneuver, "object_id": obj["ref_id"], "direction": obj["side"]},
                        )
                    )
                    future_region = self._future_region_label(obj)
                    qas.append(
                        self._v5_qa(
                            "3_2_2_future_region",
                            f"Which region is {self._typed_lane_selector_for_object(obj, objects)} most likely to enter within the next 3 seconds?",
                            self._future_region_answer(obj, future_region),
                            1.0,
                            {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane, "future_region": future_region},
                            sample_meta={"future_region": future_region, "object_id": obj["ref_id"], "direction": obj["side"]},
                        )
                    )
                    waypoints = self._trajectory_waypoints(obj)
                    if waypoints is not None:
                        qas.append(
                            self._v5_qa(
                                "3_2_1_waypoints",
                                f"Please predict the short-term future trajectory for {self._typed_lane_selector_for_object(obj, objects)}.",
                                self._format_v5_waypoints(waypoints),
                                1.0,
                                {**self._object_targets(obj, include_type=True, include_side=True), "lane_function": lane, **self._trajectory_targets(waypoints)},
                                sample_meta={"object_id": obj["ref_id"], "direction": obj["side"]},
                            )
                        )
                pair = self._same_lane_following_pair(side, lane, objects)
                if pair is not None:
                    follower, leader = pair
                    time_headway = self._time_headway(follower, leader)
                    safe = time_headway >= 2.0
                    pair_distance = round(self._dist(follower, leader), 1)
                    qas.append(
                        self._v5_qa(
                            "3_3_1_safe_following",
                            f"Are the first and second vehicles in the {lane} on the {side} approach maintaining a safe following gap?",
                            self._safe_following_answer(safe, pair_distance),
                            1.0 if safe else 0.1,
                            {
                                "follower_id": follower["ref_id"],
                                "follower_raw_tracking_id": follower["id"],
                                "leader_id": leader["ref_id"],
                                "leader_raw_tracking_id": leader["id"],
                                "distance_m": pair_distance,
                                "time_headway_sec": round(time_headway, 2),
                                "is_safe": safe,
                                "pair_type": self._pair_type_key(follower, leader),
                                "side": side,
                                "lane_function": lane,
                            },
                            sample_meta={"pair_type": self._pair_type_key(follower, leader), "is_safe": safe, "direction": side, "lane_function": lane},
                        )
                    )
        by_side, by_lane = self._split_by_side_and_lane(objects)
        lane_candidates = []
        for side in ("north", "south", "east", "west"):
            lane_map = {
                lane: lane_objs
                for (lane_side, lane), lane_objs in by_lane.items()
                if lane_side == side and lane in self._canonical_lanes()
            }
            if not lane_map:
                continue
            answer, lane_bucket, priority = self._longest_queue_lane_answer(side, lane_map, predictive=True)
            if answer is not None and lane_bucket is not None:
                lane_candidates.append((priority, side, lane_bucket, lane_map[lane_bucket]))
        if lane_candidates:
            _, side, lane_bucket, lane_objs = max(lane_candidates, key=lambda item: item[0])
            qas.append(
                self._v5_qa(
                    "3_3_2_likely_long_queue_lane",
                    "Which lane is most likely to form a long queue soon?",
                    self._v5_queue_prediction_answer(side, lane_bucket, lane_objs),
                    1.0,
                    {"lane_function": lane_bucket, "side": side, "queue_evidence": self._queue_prediction_targets(lane_bucket, lane_objs)},
                    sample_meta={"lane_function": lane_bucket, "direction": side},
                )
            )
        return qas

    def _same_lane_following_pair(self, side: str, lane: str, objects: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
        ranked = self._ranked_lane_non_ped_objects(side, lane, objects)
        if len(ranked) < 2:
            return None
        leader = ranked[0]
        follower = ranked[1]
        return follower, leader

    def _interaction_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        vehicle_candidates = self._canonical_lane_non_ped_objects(objects)
        for idx, a in enumerate(vehicle_candidates):
            for b in vehicle_candidates[idx + 1:]:
                event = self._future_pair_conflict(a, b, self.NEAR_MISS_DIST)
                has_conflict = event is not None
                qas.append(
                    self._v5_qa(
                        "3_4_1_pair_conflict",
                        f"Is there a potential conflict between {self._typed_lane_selector_for_object(a, objects)} and {self._typed_lane_selector_for_object(b, objects)} right now?",
                        self._pair_conflict_answer(a, b, has_conflict),
                        1.0 if has_conflict else 0.1,
                        {
                            "obj1_id": a["ref_id"],
                            "obj1_raw_tracking_id": a["id"],
                            "obj2_id": b["ref_id"],
                            "obj2_raw_tracking_id": b["id"],
                            "pair_type": self._pair_type_key(a, b),
                            "has_conflict": has_conflict,
                        },
                        sample_meta={"pair_type": self._pair_type_key(a, b), "has_conflict": has_conflict},
                    )
                )

        for side in ("north", "south", "east", "west"):
            for lane in self._canonical_lanes():
                for focus in self._ranked_lane_non_ped_objects(side, lane, objects):
                    conflict_candidates = []
                    for other in objects:
                        if other["id"] == focus["id"]:
                            continue
                        event = self._future_pair_conflict(focus, other, self.NEAR_MISS_DIST)
                        if event is None:
                            continue
                        conflict_candidates.append((self._dist(focus, other), other))
                    if not conflict_candidates:
                        continue
                    _, partner = min(conflict_candidates, key=lambda item: (item[0], item[1]["id"]))
                    qas.append(
                        self._v5_qa(
                            "3_4_2_nearest_conflict_participant",
                            f"Which participant is most likely to conflict with {self._typed_lane_selector_for_object(focus, objects)}?",
                            self._most_likely_conflict_partner_answer(partner),
                            1.0,
                            {
                                "focus_id": focus["ref_id"],
                                "focus_raw_tracking_id": focus["id"],
                                "focus_type": focus["type"],
                                "side": focus["side"],
                                "lane_function": focus["lane"],
                                "conflict_partner_id": partner["ref_id"],
                                "conflict_partner_raw_tracking_id": partner["id"],
                                "conflict_partner_type": partner["type"],
                                **self._object_location_targets(partner, "conflict_partner"),
                            },
                            sample_meta={"object_id": focus["ref_id"], "direction": focus["side"]},
                        )
                    )

        subject, reason = self._primary_risk_subject(objects)
        if subject is not None:
            qas.append(
                self._v5_qa(
                    "3_4_3_primary_risk_subject",
                    "Please identify the key participant associated with the major potential safety hazards.",
                    self._primary_risk_subject_answer(subject, reason),
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
        event = self._strongest_conflict(objects)
        if event is not None:
            pattern = self._interaction_pattern_label(event)
            if pattern != "other":
                pattern_side = event["obj1"]["side"] if event["obj1"]["side"] == event["obj2"]["side"] else "center"
                qas.append(
                    self._v5_qa(
                        "3_4_4_risk_pattern",
                        "What is the dominant conflict interaction pattern in the current scene?",
                        self._risk_pattern_answer(pattern, pattern_side),
                        1.0,
                        {"interaction_pattern": pattern, "side": pattern_side},
                        sample_meta={"interaction_pattern": pattern},
                    )
                )
        return qas

    def _scene_qas(self, objects: List[Dict]) -> List[Dict]:
        qas: List[Dict] = []
        level = self._congestion(objects)
        vehicles = self._vehicle_objects(objects)
        moving_vehicles = self._moving_vehicle_count(vehicles)
        total_vehicles = len(vehicles)
        qas.append(
            self._v5_qa(
                "4_1_1_overall_state",
                "What is the overall traffic condition of the intersection?",
                f"The intersection is currently {level}, with only {moving_vehicles} of {total_vehicles} vehicles moving.",
                1.0,
                {"overall_state": level, "moving_vehicles": moving_vehicles, "total_vehicles": total_vehicles},
                sample_meta={"overall_state": level},
            )
        )
        by_side, by_lane = self._split_by_side_and_lane(objects)
        for side in ("east", "north", "south", "west"):
            side_objs = by_side.get(side, [])
            targets = self._side_status_targets(side_objs)
            targets["side"] = side
            qas.append(
                self._v5_qa(
                    "4_1_2_approach_motion_status",
                    f"Please describe the motion status of traffic participants on the {side} approach of the intersection.",
                    self._side_status_answer(side, side_objs),
                    1.0,
                    targets,
                    sample_meta={"direction": side},
                )
            )
        qas.append(
            self._v5_qa(
                "4_1_3_scene_summary",
                "Provide a brief summary of the current intersection scene.",
                self._scene_summary(objects),
                1.0,
                self._scene_summary_targets(objects),
            )
        )
        side_counts = {side: len(by_side.get(side, [])) for side in ("north", "south", "east", "west")}
        side_tie_priority = {"north": 0, "south": 1, "east": 2, "west": 3}
        dominant_side = min(side_counts, key=lambda side: (-side_counts[side], side_tie_priority[side])) if side_counts else None
        participant_count = side_counts.get(dominant_side, 0) if dominant_side is not None else 0
        busiest_answer = f"The {dominant_side} approach is the busiest, with {participant_count} traffic participants." if dominant_side is not None else "The north approach is the busiest, with 0 traffic participants."
        qas.append(
            self._v5_qa(
                "4_1_4_heaviest_traffic_approach",
                "Which approach to the intersection has the heaviest traffic?",
                busiest_answer,
                float(participant_count),
                {"dominant_side": dominant_side, "participant_count": participant_count},
                sample_meta={"dominant_side": dominant_side, "participant_count": participant_count},
            )
        )
        fastest = max([obj for obj in objects if obj["type"] != "pedestrian"], key=self._speed, default=None)
        if fastest is not None and self._speed(fastest) >= self.VEHICLE_OVERSPEED_THRESHOLD - self.SIDE_SPEEDING_MARGIN:
            location_phrase = (
                "in the center of the intersection"
                if fastest["side"] == "center"
                else f"on the {fastest['side']} approach"
            )
            answer = f"Yes. A {fastest['type']} {location_phrase} appears to be traveling at high speed at about {self._speed(fastest):.1f} m/s."
        else:
            answer = "No."
        speeding_targets = self._side_speeding_targets(fastest if answer.startswith("Yes") else None)
        if fastest is not None and answer.startswith("Yes"):
            speeding_targets["vehicle_id"] = fastest["ref_id"]
            speeding_targets["side"] = fastest["side"]
            speeding_targets["risk_region"] = fastest["side"]
            speeding_targets["evidence"] = f"A {fastest['type']} is still moving at about {self._speed(fastest):.1f} m/s."
        qas.append(
            self._v5_qa(
                "4_2_1_speeding_risk",
                "Is there still a risk of speeding at the intersection?",
                answer,
                1.0 if answer.startswith("Yes") else 0.1,
                speeding_targets,
                sample_meta={"direction": fastest["side"] if fastest is not None and answer.startswith("Yes") else "none", "has_speeding_risk": answer.startswith("Yes")},
            )
        )
        notable = self._notable_abnormal(objects)
        if notable is not None:
            notable_label, notable_region, _ = notable
            qas.append(
                self._v5_qa(
                    "4_2_2_notable_abnormal",
                    "What is the most notable current abnormal event in the scene?",
                    self._notable_abnormal_answer(notable_label, notable_region),
                    1.0,
                    {"notable_abnormal": notable_label, "risk_region": notable_region, "reason": self._notable_abnormal_answer(notable_label, notable_region), "evidence": self._notable_abnormal_answer(notable_label, notable_region)},
                    sample_meta={"notable_abnormal": notable_label},
                )
            )
        intersection_answer, intersection_bucket = self._intersection_safe_action_answer(objects)
        qas.append(
            self._v5_qa(
                "4_3_1_intersection_action",
                "How should traffic proceed through the intersection right now?",
                intersection_answer,
                1.0,
                self._action_state_targets(intersection_bucket),
                sample_meta={"action_state": intersection_bucket},
            )
        )
        for side in ("north", "south", "east", "west"):
            side_answer, side_bucket = self._side_safe_action_answer(side, by_side.get(side, []))
            qas.append(
                self._v5_qa(
                    "4_3_2_approach_action",
                    f"What precaution should traffic take on the {side} approach of the intersection right now?",
                    side_answer,
                    1.0,
                    {"side": side, **self._action_state_targets(side_bucket)},
                    sample_meta={"action_state": side_bucket, "direction": side},
                )
            )
        for side in ("north", "south", "east", "west"):
            lane_map = {lane: lane_objs for (lane_side, lane), lane_objs in by_lane.items() if lane_side == side}
            for lane in self._canonical_lanes():
                lane_objs = lane_map.get(lane, [])
                lane_answer, lane_bucket = self._lane_safe_action_answer(side, lane, lane_objs, objects)
                qas.append(
                    self._v5_qa(
                        "4_3_3_lane_action",
                        f"How should traffic behave in the {lane} on the {side} approach right now?",
                        lane_answer,
                        1.0,
                        {"side": side, "lane_function": lane, **self._action_state_targets(lane_bucket)},
                        sample_meta={"action_state": lane_bucket, "lane_function": lane, "direction": side},
                    )
                )
        center_obj = self._highest_risk_center_vehicle(objects)
        if center_obj is not None:
            object_answer, object_bucket = self._object_safe_action_answer(center_obj, objects)
            qas.append(
                self._v5_qa(
                    "4_3_4_object_action",
                    "What is the safest action for the highest-risk vehicle in the center of the intersection?",
                    object_answer,
                    1.0,
                    {**self._object_targets(center_obj, include_type=True, include_side=True), **self._action_state_targets(object_bucket)},
                    sample_meta={"action_state": object_bucket, "object_id": center_obj["ref_id"], "object_type": center_obj["type"]},
                )
            )
        return qas

    def generate_dataset(
        self,
        output_path: str,
        max_frames: Optional[int] = None,
        keyframe_fps: float = IntersectionQAGeneratorV5Runtime.DEFAULT_KEYFRAME_FPS,
        max_per_type: int = IntersectionQAGeneratorV5Runtime.DEFAULT_MAX_PER_TYPE,
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
                self._print_progress("Generating QA V6", index, len(eligible_sampled), generation_start)
        else:
            ordered_results: Dict[int, List[Dict]] = {}
            max_workers = min(worker_count, len(eligible_indices))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("fork"),
                initializer=_init_v6_worker,
                initargs=(str(self.pkl_path), self.subtemplate_patch_style),
            ) as executor:
                future_to_order = {
                    executor.submit(_generate_v6_frame_worker, frame_index, max_per_type, keyframe_fps): order
                    for order, frame_index in enumerate(eligible_indices)
                }
                completed = 0
                for future in as_completed(future_to_order):
                    order = future_to_order[future]
                    ordered_results[order] = future.result()
                    completed += 1
                    self._print_progress("Generating QA V6", completed, len(eligible_indices), generation_start)
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
                "version": "v6",
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
                        "crosswalk_exit_distance": "crosswalk_principal_axis",
                    },
                    "timing_rules": {
                        "default_temporal_window_sec": 3.0,
                        "waypoint_times_sec": [0.5, 1.0, 1.5, 2.0],
                    },
                    "natural_selector_rules": {
                        "lane_rank_order": "distance_to_relevant_stop_line",
                        "typed_rank_scope": "same_approach_same_lane_same_type",
                        "front_relation": "referenced_object_heading_frame",
                        "nearest_crosswalk_pedestrian": "closest_to_intersection_center",
                        "center_object_guidance": "highest_risk_vehicle_in_center",
                    },
                },
            },
            "qa_pairs": public_qas,
        }
        print("Writing JSON...")
        Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(post_ratio_qas)} QA pairs to {output_path}")
        return public_qas
