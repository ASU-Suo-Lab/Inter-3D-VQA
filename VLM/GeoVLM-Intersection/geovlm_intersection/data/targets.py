from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from geovlm_intersection.data.v5_io import PreparedSample

OBJECT_TYPE_VOCAB = [
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "golf cart",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
    "van",
]
SIDE_VOCAB = ["center", "east", "north", "south", "west"]
MOTION_STATE_VOCAB = ["braking", "creeping", "moving", "running", "standing", "starting", "stopped", "walking"]
RISK_REASON_VOCAB = ["lane_change_conflict", "overspeed", "path_crossing", "proximity", "vru_conflict"]
INTERSECTION_ACTION_VOCAB = [
    "CONFLICT_SUPPRESSION",
    "FLOW_CALMING",
    "FLOW_STABLE",
    "QUEUE_MANAGEMENT",
]
SIDE_ACTION_VOCAB = [
    "SIDE_CLEARANCE_PROTECTION",
    "SIDE_CROSSING_AWARENESS",
    "SIDE_GENERAL_CAUTION",
    "SIDE_QUEUE_STABILIZATION",
    "SIDE_SPEED_MODERATION",
]
LANE_ACTION_VOCAB = [
    "LANE_CLEARANCE_MAINTENANCE",
    "LANE_GENERAL_ORDER",
    "LANE_PREPARE_TO_STOP",
    "LANE_QUEUE_PRESERVATION",
    "LANE_SPEED_REDUCTION",
]
OBJECT_ACTION_VOCAB = [
    "OBJECT_PREPARE_TO_STOP",
    "OBJECT_PROCEED_CAUTIOUSLY",
    "OBJECT_SLOW_DOWN",
    "OBJECT_YIELD_NOW",
]
LANE_FUNCTION_VOCAB = ["left-turn lane", "through lane", "right-turn lane"]
CAMERA_VOCAB = ["north_image", "south_image", "east_image", "west_image"]

POSITION_3D_NORM_X = 120.0
POSITION_3D_NORM_Y = 160.0
IMAGE_REF_NORM_X = 1920.0
IMAGE_REF_NORM_Y = 1080.0
SPEEDING_RISK_SPEED_THRESHOLD_MPS = 8.0

OBJECT_TYPE_TO_INDEX = {value: idx for idx, value in enumerate(OBJECT_TYPE_VOCAB)}
SIDE_TO_INDEX = {value: idx for idx, value in enumerate(SIDE_VOCAB)}
MOTION_STATE_TO_INDEX = {value: idx for idx, value in enumerate(MOTION_STATE_VOCAB)}
RISK_REASON_TO_INDEX = {value: idx for idx, value in enumerate(RISK_REASON_VOCAB)}
INTERSECTION_ACTION_TO_INDEX = {value: idx for idx, value in enumerate(INTERSECTION_ACTION_VOCAB)}
SIDE_ACTION_TO_INDEX = {value: idx for idx, value in enumerate(SIDE_ACTION_VOCAB)}
LANE_ACTION_TO_INDEX = {value: idx for idx, value in enumerate(LANE_ACTION_VOCAB)}
OBJECT_ACTION_TO_INDEX = {value: idx for idx, value in enumerate(OBJECT_ACTION_VOCAB)}
LANE_FUNCTION_TO_INDEX = {value: idx for idx, value in enumerate(LANE_FUNCTION_VOCAB)}
CAMERA_TO_INDEX = {value: idx for idx, value in enumerate(CAMERA_VOCAB)}

SUPPORTED_STRUCTURED_SUBTEMPLATES = {
    "1_1_1_fine_type",
    "1_1_2_side_exists",
    "1_1_3_side_count",
    "1_1_4_relative_neighbor_type",
    "2_1_2_ped_to_far_edge",
    "2_1_4_nearest_vehicle_to_ped",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "4_2_1_speeding_risk",
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

SUBTEMPLATE_VOCAB = [
    "1_1_1_fine_type",
    "1_1_2_side_exists",
    "1_1_3_side_count",
    "1_1_4_relative_neighbor_type",
    "2_1_2_ped_to_far_edge",
    "2_1_4_nearest_vehicle_to_ped",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "4_2_1_speeding_risk",
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
]
SUBTEMPLATE_TO_INDEX = {value: idx for idx, value in enumerate(SUBTEMPLATE_VOCAB)}

OBJECT_CENTRIC_SUBTEMPLATES = {
    "1_1_1_fine_type",
    "1_1_4_relative_neighbor_type",
    "2_1_4_nearest_vehicle_to_ped",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "4_2_1_speeding_risk",
    "4_3_4_object_action",
}

RELATION_SUBTEMPLATES = {
    "1_1_4_relative_neighbor_type",
    "2_1_4_nearest_vehicle_to_ped",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
}

SINGLE_OBJECT_SUBTEMPLATES = OBJECT_CENTRIC_SUBTEMPLATES - RELATION_SUBTEMPLATES

GLOBAL_ROUTE_SUBTEMPLATES = {
    "1_1_2_side_exists",
    "1_1_3_side_count",
    "2_1_2_ped_to_far_edge",
}

INTERSECTION_ROUTE_SUBTEMPLATES = {"4_3_1_intersection_action"}
SIDE_ROUTE_SUBTEMPLATES = {"4_3_2_side_action"}
LANE_ROUTE_SUBTEMPLATES = {"4_3_3_lane_action"}

FOCUS_SUBTEMPLATES_STAGE3 = {
    "1_1_1_fine_type",
    "1_1_4_relative_neighbor_type",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "4_2_1_speeding_risk",
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
}

OBJECT_SELECTION_MATCH_MAX_DISTANCE_M = 6.0

LION_CLASS_NAMES = [
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
]


@dataclass(frozen=True)
class GeoVLMSupervision:
    subtemplate: str
    subtemplate_index: int
    object_type_index: int | None = None
    side_index: int | None = None
    motion_state_index: int | None = None
    risk_reason_index: int | None = None
    intersection_action_index: int | None = None
    side_action_index: int | None = None
    lane_action_index: int | None = None
    object_action_index: int | None = None
    object_selection_index: int | None = None
    lane_function_index: int | None = None
    binary_answer: float | None = None
    count_value: float | None = None
    distance_value: float | None = None
    speed_value: float | None = None
    acceleration_value: float | None = None
    position_3d: tuple[float, float] | None = None
    camera_index: int | None = None
    image_ref: tuple[float, float] | None = None
    target_object_type_name: str | None = None
    target_raw_tracking_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtemplate": self.subtemplate,
            "subtemplate_index": self.subtemplate_index,
            "object_type_index": self.object_type_index,
            "side_index": self.side_index,
            "motion_state_index": self.motion_state_index,
            "risk_reason_index": self.risk_reason_index,
            "intersection_action_index": self.intersection_action_index,
            "side_action_index": self.side_action_index,
            "lane_action_index": self.lane_action_index,
            "object_action_index": self.object_action_index,
            "object_selection_index": self.object_selection_index,
            "lane_function_index": self.lane_function_index,
            "binary_answer": self.binary_answer,
            "count_value": self.count_value,
            "distance_value": self.distance_value,
            "speed_value": self.speed_value,
            "acceleration_value": self.acceleration_value,
            "position_3d": list(self.position_3d) if self.position_3d is not None else None,
            "camera_index": self.camera_index,
            "image_ref": list(self.image_ref) if self.image_ref is not None else None,
            "target_object_type_name": self.target_object_type_name,
            "target_raw_tracking_id": self.target_raw_tracking_id,
        }


def _index_or_none(value: str | None, mapping: dict[str, int], field_name: str) -> int | None:
    if value in (None, ""):
        return None
    if value not in mapping:
        raise KeyError(f"Unsupported {field_name} value: {value}")
    return mapping[value]


def _extract_first_image_ref(image_refs: Any) -> tuple[int | None, tuple[float, float] | None]:
    if not isinstance(image_refs, list) or not image_refs:
        return None, None
    first = image_refs[0]
    if not isinstance(first, dict):
        return None, None
    image_name = first.get("image_name")
    x1 = first.get("x1")
    y1 = first.get("y1")
    if image_name in (None, "") or x1 is None or y1 is None:
        return None, None
    return _index_or_none(str(image_name), CAMERA_TO_INDEX, "image_name"), (float(x1), float(y1))


def _extract_xy(position: Any) -> tuple[float, float] | None:
    if not isinstance(position, dict):
        return None
    if "x" not in position or "y" not in position:
        return None
    return float(position["x"]), float(position["y"])


def _resolve_tracking_target(sample: PreparedSample, raw_tracking_id: str | None) -> tuple[tuple[float, float] | None, str | None]:
    if raw_tracking_id in (None, ""):
        return None, None
    tracking_ids = sample.info_record.get("tracking_id")
    gt_boxes = sample.info_record.get("gt_boxes")
    gt_names = sample.info_record.get("gt_names")
    if not isinstance(tracking_ids, list) or gt_boxes is None or gt_names is None:
        return None, None
    try:
        target_index = tracking_ids.index(str(raw_tracking_id))
    except ValueError:
        return None, None
    if target_index >= len(gt_boxes) or target_index >= len(gt_names):
        return None, None
    box = gt_boxes[target_index]
    return (float(box[0]), float(box[1])), str(gt_names[target_index])


def map_lion_label_to_object_type(label_value: float | int) -> str | None:
    label_index = int(round(float(label_value))) - 1
    if label_index < 0 or label_index >= len(LION_CLASS_NAMES):
        return None
    class_name = LION_CLASS_NAMES[label_index]
    return class_name if class_name in OBJECT_TYPE_TO_INDEX else None


def normalize_position_3d(position: tuple[float, float]) -> tuple[float, float]:
    return (float(position[0]) / POSITION_3D_NORM_X, float(position[1]) / POSITION_3D_NORM_Y)


def denormalize_position_3d(position: tuple[float, float]) -> tuple[float, float]:
    return (float(position[0]) * POSITION_3D_NORM_X, float(position[1]) * POSITION_3D_NORM_Y)


def normalize_image_ref(image_ref: tuple[float, float]) -> tuple[float, float]:
    return (float(image_ref[0]) / IMAGE_REF_NORM_X, float(image_ref[1]) / IMAGE_REF_NORM_Y)


def denormalize_image_ref(image_ref: tuple[float, float]) -> tuple[float, float]:
    return (float(image_ref[0]) * IMAGE_REF_NORM_X, float(image_ref[1]) * IMAGE_REF_NORM_Y)


def build_structured_supervision(sample: PreparedSample) -> GeoVLMSupervision | None:
    subtemplate = sample.subtemplate
    if subtemplate not in SUPPORTED_STRUCTURED_SUBTEMPLATES:
        return None
    targets = sample.structured_targets or {}
    if not isinstance(targets, dict):
        return None

    if subtemplate == "1_1_1_fine_type":
        camera_index, image_ref = _extract_first_image_ref(targets.get("object_image_refs"))
        position = targets.get("object_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("object_type"), OBJECT_TYPE_TO_INDEX, "object_type"),
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("object_type"),
            target_raw_tracking_id=targets.get("raw_tracking_id"),
        )

    if subtemplate == "1_1_2_side_exists":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            binary_answer=1.0 if bool(targets.get("exists")) else 0.0,
            count_value=float(targets.get("count", 0.0)),
        )

    if subtemplate == "1_1_3_side_count":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            object_type_index=_index_or_none(targets.get("object_type"), OBJECT_TYPE_TO_INDEX, "object_type"),
            count_value=float(targets["count"]) if targets.get("count") is not None else None,
        )

    if subtemplate == "1_1_4_relative_neighbor_type":
        camera_index, image_ref = _extract_first_image_ref(targets.get("target_object_image_refs"))
        position = targets.get("target_object_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("object_type"), OBJECT_TYPE_TO_INDEX, "object_type"),
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("object_type"),
            target_raw_tracking_id=targets.get("target_object_raw_tracking_id"),
        )

    if subtemplate == "2_1_2_ped_to_far_edge":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            side_index=_index_or_none(targets.get("crosswalk"), SIDE_TO_INDEX, "crosswalk"),
            distance_value=float(targets["distance_m"]) if targets.get("distance_m") is not None else None,
        )

    if subtemplate == "2_1_4_nearest_vehicle_to_ped":
        resolved_position, resolved_type = _resolve_tracking_target(sample, targets.get("vehicle_raw_tracking_id"))
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("vehicle_type"), OBJECT_TYPE_TO_INDEX, "vehicle_type"),
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            distance_value=float(targets["distance_m"]) if targets.get("distance_m") is not None else None,
            position_3d=resolved_position,
            target_object_type_name=targets.get("vehicle_type") or resolved_type,
            target_raw_tracking_id=targets.get("vehicle_raw_tracking_id"),
        )

    if subtemplate == "3_1_1_current_motion_state":
        camera_index, image_ref = _extract_first_image_ref(targets.get("object_image_refs"))
        position = targets.get("object_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("object_type"), OBJECT_TYPE_TO_INDEX, "object_type"),
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            motion_state_index=_index_or_none(targets.get("motion_state"), MOTION_STATE_TO_INDEX, "motion_state"),
            speed_value=float(targets["speed"]) if targets.get("speed") is not None else None,
            acceleration_value=float(targets["acceleration"]) if targets.get("acceleration") is not None else None,
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("object_type"),
            target_raw_tracking_id=targets.get("raw_tracking_id"),
        )

    if subtemplate == "3_4_2_nearest_conflict_participant":
        camera_index, image_ref = _extract_first_image_ref(targets.get("conflict_partner_image_refs"))
        position = targets.get("conflict_partner_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(
                targets.get("conflict_partner_type"), OBJECT_TYPE_TO_INDEX, "conflict_partner_type"
            ),
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("conflict_partner_type"),
            target_raw_tracking_id=targets.get("conflict_partner_raw_tracking_id"),
        )

    if subtemplate == "3_4_3_primary_risk_subject":
        camera_index, image_ref = _extract_first_image_ref(targets.get("subject_image_refs"))
        position = targets.get("subject_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("subject_type"), OBJECT_TYPE_TO_INDEX, "subject_type"),
            risk_reason_index=_index_or_none(targets.get("risk_reason"), RISK_REASON_TO_INDEX, "risk_reason"),
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("subject_type"),
            target_raw_tracking_id=targets.get("subject_raw_tracking_id"),
        )

    if subtemplate == "4_2_1_speeding_risk":
        numeric_targets = targets.get("numeric_targets") or {}
        resolved_position, resolved_type = _resolve_tracking_target(sample, targets.get("vehicle_raw_tracking_id"))
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            binary_answer=1.0 if bool(targets.get("has_speeding_risk")) else 0.0,
            object_type_index=_index_or_none(targets.get("vehicle_type"), OBJECT_TYPE_TO_INDEX, "vehicle_type"),
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            speed_value=float(numeric_targets["speed_mps"]) if numeric_targets.get("speed_mps") is not None else None,
            position_3d=resolved_position,
            target_object_type_name=targets.get("vehicle_type") or resolved_type,
            target_raw_tracking_id=targets.get("vehicle_raw_tracking_id"),
        )

    if subtemplate == "4_3_1_intersection_action":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            intersection_action_index=_index_or_none(
                targets.get("action_state"), INTERSECTION_ACTION_TO_INDEX, "intersection_action"
            ),
        )

    if subtemplate == "4_3_2_side_action":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            side_action_index=_index_or_none(targets.get("action_state"), SIDE_ACTION_TO_INDEX, "side_action"),
        )

    if subtemplate == "4_3_3_lane_action":
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            side_index=_index_or_none(targets.get("side"), SIDE_TO_INDEX, "side"),
            lane_function_index=_index_or_none(targets.get("lane_function"), LANE_FUNCTION_TO_INDEX, "lane_function"),
            lane_action_index=_index_or_none(targets.get("action_state"), LANE_ACTION_TO_INDEX, "lane_action"),
        )

    if subtemplate == "4_3_4_object_action":
        camera_index, image_ref = _extract_first_image_ref(targets.get("object_image_refs"))
        position = targets.get("object_position") or {}
        return GeoVLMSupervision(
            subtemplate=subtemplate,
            subtemplate_index=SUBTEMPLATE_TO_INDEX[subtemplate],
            object_type_index=_index_or_none(targets.get("object_type"), OBJECT_TYPE_TO_INDEX, "object_type"),
            object_action_index=_index_or_none(targets.get("action_state"), OBJECT_ACTION_TO_INDEX, "object_action"),
            position_3d=_extract_xy(position),
            camera_index=camera_index,
            image_ref=image_ref,
            target_object_type_name=targets.get("object_type"),
            target_raw_tracking_id=targets.get("raw_tracking_id"),
        )

    return None


def supervision_to_device_dict(supervision: GeoVLMSupervision, device: torch.device | str) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {
        "subtemplate_index": torch.tensor([supervision.subtemplate_index], dtype=torch.long, device=device),
    }
    if supervision.object_type_index is not None:
        payload["object_type_index"] = torch.tensor([supervision.object_type_index], dtype=torch.long, device=device)
    if supervision.side_index is not None:
        payload["side_index"] = torch.tensor([supervision.side_index], dtype=torch.long, device=device)
    if supervision.motion_state_index is not None:
        payload["motion_state_index"] = torch.tensor([supervision.motion_state_index], dtype=torch.long, device=device)
    if supervision.risk_reason_index is not None:
        payload["risk_reason_index"] = torch.tensor([supervision.risk_reason_index], dtype=torch.long, device=device)
    if supervision.intersection_action_index is not None:
        payload["intersection_action_index"] = torch.tensor(
            [supervision.intersection_action_index], dtype=torch.long, device=device
        )
    if supervision.side_action_index is not None:
        payload["side_action_index"] = torch.tensor([supervision.side_action_index], dtype=torch.long, device=device)
    if supervision.lane_action_index is not None:
        payload["lane_action_index"] = torch.tensor([supervision.lane_action_index], dtype=torch.long, device=device)
    if supervision.object_action_index is not None:
        payload["object_action_index"] = torch.tensor([supervision.object_action_index], dtype=torch.long, device=device)
    if supervision.object_selection_index is not None:
        payload["object_selection_index"] = torch.tensor([supervision.object_selection_index], dtype=torch.long, device=device)
    if supervision.lane_function_index is not None:
        payload["lane_function_index"] = torch.tensor([supervision.lane_function_index], dtype=torch.long, device=device)
    if supervision.binary_answer is not None:
        payload["binary_answer"] = torch.tensor([supervision.binary_answer], dtype=torch.float32, device=device)
    if supervision.count_value is not None:
        payload["count_value"] = torch.tensor([supervision.count_value], dtype=torch.float32, device=device)
    if supervision.distance_value is not None:
        payload["distance_value"] = torch.tensor([supervision.distance_value], dtype=torch.float32, device=device)
    if supervision.speed_value is not None:
        payload["speed_value"] = torch.tensor([supervision.speed_value], dtype=torch.float32, device=device)
    if supervision.acceleration_value is not None:
        payload["acceleration_value"] = torch.tensor([supervision.acceleration_value], dtype=torch.float32, device=device)
    if supervision.position_3d is not None:
        payload["position_3d"] = torch.tensor(
            [list(normalize_position_3d(supervision.position_3d))], dtype=torch.float32, device=device
        )
    if supervision.camera_index is not None:
        payload["camera_index"] = torch.tensor([supervision.camera_index], dtype=torch.long, device=device)
    if supervision.image_ref is not None:
        payload["image_ref"] = torch.tensor(
            [list(normalize_image_ref(supervision.image_ref))], dtype=torch.float32, device=device
        )
    return payload
