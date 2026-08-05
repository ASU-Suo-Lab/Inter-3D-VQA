#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate_intersection_vqa as base


NATURAL_OBJECT_PATTERN = re.compile(
    r"the\s+([a-z0-9_ ]+?)"
    r"(?:\s+in the\s+(left-turn lane|through lane|right-turn lane)\s+on the\s+(north|south|east|west)\s+approach"
    r"|\s+on the\s+(north|south|east|west)\s+approach"
    r"|\s+in the center of the intersection)?$"
)

NATURAL_OBJECT_SEARCH_PATTERN = re.compile(
    r"the\s+([a-z0-9_ ]+?)"
    r"(?:\s+in the\s+(left-turn lane|through lane|right-turn lane)\s+on the\s+(north|south|east|west)\s+approach"
    r"|\s+on the\s+(north|south|east|west)\s+approach"
    r"|\s+in the center of the intersection)"
)

V6_INTERSECTION_ACTION_BY_TEXT = {
    base.normalize_parse_text("Traffic should continue orderly progression while maintaining normal caution."): "FLOW_STABLE",
    base.normalize_parse_text("Traffic should proceed more calmly with moderated speed and extra spacing."): "FLOW_CALMING",
    base.normalize_parse_text("Traffic should proceed in a tightly managed and orderly way to prevent queue growth."): "QUEUE_MANAGEMENT",
    base.normalize_parse_text("Traffic should proceed conservatively and give extra priority to suppressing risky interactions, especially around vulnerable or conflicted movement."): "CONFLICT_SUPPRESSION",
}

V6_SIDE_ACTION_BY_TEXT = {
    base.normalize_parse_text("Traffic should keep safer spacing and protect local clearance around nearby participants."): "SIDE_CLEARANCE_PROTECTION",
    base.normalize_parse_text("Traffic should pay close attention to crossing movement and remain ready to yield if needed."): "SIDE_CROSSING_AWARENESS",
    base.normalize_parse_text("Traffic should moderate speed and maintain safer local spacing."): "SIDE_SPEED_MODERATION",
    base.normalize_parse_text("Traffic should stabilize queue movement and avoid unnecessary disruption."): "SIDE_QUEUE_STABILIZATION",
    base.normalize_parse_text("Traffic should proceed cautiously and remain orderly."): "SIDE_GENERAL_CAUTION",
}

V6_LANE_ACTION_BY_TEXT = {
    base.normalize_parse_text("Traffic should maintain local clearance and avoid tight conflicts in this lane."): "LANE_CLEARANCE_MAINTENANCE",
    base.normalize_parse_text("Traffic should prepare to stop and avoid pressing forward in this lane."): "LANE_PREPARE_TO_STOP",
    base.normalize_parse_text("Traffic should preserve queue order and avoid unnecessary lane reorganization."): "LANE_QUEUE_PRESERVATION",
    base.normalize_parse_text("Traffic should reduce speed and proceed more conservatively in this lane."): "LANE_SPEED_REDUCTION",
    base.normalize_parse_text("Traffic should maintain orderly progression, remain in lane, and avoid unnecessary lane reorganization."): "LANE_GENERAL_ORDER",
}

V6_INTERSECTION_ACTION_PHRASES = {
    "FLOW_STABLE": ("continue orderly progression", "maintaining normal caution", "maintain normal caution"),
    "FLOW_CALMING": ("proceed more calmly", "moderated speed", "extra spacing"),
    "QUEUE_MANAGEMENT": ("tightly managed", "prevent queue growth"),
    "CONFLICT_SUPPRESSION": ("proceed conservatively", "suppressing risky interactions", "vulnerable or conflicted movement"),
}

V6_SIDE_ACTION_PHRASES = {
    "SIDE_CLEARANCE_PROTECTION": ("keep safer spacing", "protect local clearance", "local clearance around nearby participants"),
    "SIDE_CROSSING_AWARENESS": ("crossing movement", "ready to yield", "remain ready to yield"),
    "SIDE_SPEED_MODERATION": ("moderate speed", "maintain safer local spacing", "safer local spacing"),
    "SIDE_QUEUE_STABILIZATION": ("stabilize queue movement", "avoid unnecessary disruption"),
    "SIDE_GENERAL_CAUTION": ("proceed cautiously", "remain orderly"),
}

V6_LANE_ACTION_PHRASES = {
    "LANE_CLEARANCE_MAINTENANCE": ("maintain local clearance", "avoid tight conflicts"),
    "LANE_PREPARE_TO_STOP": ("prepare to stop", "avoid pressing forward"),
    "LANE_QUEUE_PRESERVATION": ("preserve queue order",),
    "LANE_SPEED_REDUCTION": ("reduce speed", "proceed more conservatively"),
    "LANE_GENERAL_ORDER": ("maintain orderly progression", "remain in lane"),
}

V6_OBJECT_ACTION_SUFFIX_TO_STATE = {
    "should yield momentarily and allow the nearby interaction to clear before continuing.": "OBJECT_YIELD_NOW",
    "should prepare to stop and wait for nearby movement to clear.": "OBJECT_PREPARE_TO_STOP",
    "should slow down and create more local safety margin.": "OBJECT_SLOW_DOWN",
    "should proceed cautiously while monitoring nearby participants.": "OBJECT_PROCEED_CAUTIOUSLY",
}

V6_ABNORMAL_TEXT_TO_LABEL = {
    "proximity interaction": "abnormal_proximity",
    "crosswalk_blocking": "crosswalk_blocking",
    "lingering_pedestrian": "lingering_pedestrian",
    "speeding": "speeding",
    "stopline_overrun": "stopline_overrun",
    "wrong_way_two_wheeler": "wrong_way_two_wheeler",
    "queue_spillback": "queue_spillback",
}


V6_MANEUVER_TEXT_TO_LABEL = {
    "making a left turn": "left turn",
    "going straight": "straight",
    "making a right turn": "right turn",
    "making a lane change": "lane change",
    "executing a stop-and-wait": "stop-and-wait",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V6 Intersection VQA predictions against sidecar structured targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sidecar-jsonl",
        type=Path,
        default=Path("LlamaFactory/data/intersection_vqa_v6/intersection_vqa_v6_eval_sidecar.jsonl"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("LlamaFactory/eval/intersection_vqa_v6"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=base.default_device())
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for debugging.")
    return parser.parse_args()


def _parse_type_only_sentence(text: str, field_name: str) -> Dict[str, Any]:
    match = re.match(r"it is a[n]?\s+([a-z0-9_ ]+?)\.?$", base.normalize_parse_text(text))
    if not match:
        return {}
    return {field_name: base.normalize_object_type(match.group(1))}


def _parse_natural_object_phrase(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    match = NATURAL_OBJECT_PATTERN.match(norm)
    if not match:
        return {}
    lane = match.group(2)
    side = match.group(3) or match.group(4)
    if "center of the intersection" in norm:
        side = "center"
    out = {"object_type": base.normalize_object_type(match.group(1))}
    if side is not None:
        out["side"] = side
    if lane is not None:
        out["lane_function"] = lane
    return out


def _extract_natural_object_phrase(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text).rstrip(".")
    parsed = _parse_natural_object_phrase(norm)
    if parsed:
        return parsed
    match = NATURAL_OBJECT_SEARCH_PATTERN.search(norm)
    if not match:
        return {}
    lane = match.group(2)
    side = match.group(3) or match.group(4)
    snippet = match.group(0)
    if "center of the intersection" in snippet:
        side = "center"
    out = {"object_type": base.normalize_object_type(match.group(1))}
    if side is not None:
        out["side"] = side
    if lane is not None:
        out["lane_function"] = lane
    return out


def parse_1_1_1_v6(text: str) -> Dict[str, Any]:
    return _parse_type_only_sentence(text, "object_type")


def parse_1_1_2_v6(text: str) -> Dict[str, Any]:
    return _parse_type_only_sentence(text, "target_object_type")


def parse_1_1_3_v6(text: str) -> Dict[str, Any]:
    exists = base.parse_yes_no(text)
    if exists is None:
        return {}
    out = {"exists": exists}
    if exists:
        match = re.search(r"\b(?:there is|there are)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+vrus?", base.normalize_parse_text(text))
        out["count"] = base.parse_int_token(match.group(1)) if match else None
    return out


def parse_1_2_1_v6(text: str) -> Dict[str, Any]:
    match = re.match(r"the\s+([a-z0-9_ ]+?)\s+is classified as a[n]?\s+([a-z-]+)-size vehicle\.?$", base.normalize_parse_text(text))
    if not match:
        return {}
    return {
        "object_type": base.normalize_object_type(match.group(1)),
        "size_bucket": match.group(2),
    }


def parse_1_2_2_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    match = re.match(r"the\s+[a-z0-9_ ]+?\s+is\s+(.+?)\.?$", norm)
    if not match:
        return {}
    return {"visibility": match.group(1).strip()}


def parse_1_3_1_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    out: Dict[str, Any] = {}
    weather_match = re.search(r"\b(rainy|cloudy|sunny)\b", norm)
    if weather_match:
        out["weather"] = weather_match.group(1)
    time_match = re.search(r"\b(daytime|nighttime)\b", norm)
    if time_match:
        out["time_of_day"] = time_match.group(1)
    if re.search(r"\bno\s+(?:strong\s+)?sun glare\b", norm):
        out["light"] = {"sun_glare": False, "sun_glare_sides": []}
    elif "sun glare" in norm:
        sides: List[str] = []
        segment_match = re.search(r"sun glare(?:[^.]*?)in the\s+(.+?)\s+views?", norm)
        if segment_match:
            raw = segment_match.group(1)
            seen = set()
            for token in re.split(r",\s*|\s+and\s+", raw):
                token = token.strip()
                if token.endswith(" view"):
                    token = token[:-5]
                if token in base.APPROACH_ORDER and token not in seen:
                    sides.append(token)
                    seen.add(token)
            sides = [side for side in base.APPROACH_ORDER if side in seen]
        out["light"] = {"sun_glare": True, "sun_glare_sides": sides}
    return out


def parse_2_1_1_v6(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+([a-z0-9_ ]+?)\s+is\s+(-?\d+(?:\.\d+)?)\s+(?:m|meter|meters)\s+(?:away\s+)?from\s+the\s+stop\s+line\.?$",
        base.normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "object_type": base.normalize_object_type(match.group(1)),
        "distance_m": base.format_float(float(match.group(2))),
    }


def parse_2_1_2_v6(text: str) -> Dict[str, Any]:
    match = re.match(r"the pedestrian is\s+(-?\d+(?:\.\d+)?)\s+m\s+from the exit area\.?$", base.normalize_parse_text(text))
    if not match:
        return {}
    return {"distance_m": base.format_float(float(match.group(1)))}


def parse_2_1_3_v6(text: str) -> Dict[str, Any]:
    match = re.match(r"the\s+([a-z0-9_ ]+?)\s+is\s+(-?\d+(?:\.\d+)?)\s+m\s+from the\s+([a-z0-9_ ]+?)\.?$", base.normalize_parse_text(text))
    if not match:
        return {}
    return {
        "obj1_type": base.normalize_object_type(match.group(1)),
        "distance_m": base.format_float(float(match.group(2))),
        "obj2_type": base.normalize_object_type(match.group(3)),
    }


def parse_2_1_4_v6(text: str) -> Dict[str, Any]:
    match = re.match(
        r"(?:it is|the nearest one is|the nearest vehicle is)\s+a[n]?\s+([a-z0-9_ ]+?)"
        r"(?:,\s*|\s+at\s+)(-?\d+(?:\.\d+)?)\s+(?:m|meter|meters)(?:\s+away)?\.?$",
        base.normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "vehicle_type": base.normalize_object_type(match.group(1)),
        "distance_m": base.format_float(float(match.group(2))),
    }


def parse_2_2_1_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    if "within the crosswalk" in norm:
        return {"ped_zone": "within the crosswalk"}
    for label in ("entry zone", "exit zone", "waiting zone"):
        if label in norm:
            return {"ped_zone": label}
    return {}


def parse_2_2_5_v6(text: str) -> Dict[str, Any]:
    blocked = base.parse_yes_no(text)
    if blocked is None:
        return {}
    out = {"crosswalk_blocked": blocked}
    if not blocked:
        return out
    match = re.match(r"yes,\s+a[n]?\s+([a-z0-9_ ]+?)\s+is currently blocking the\s+(north|south|east|west)\s+crosswalk\.?$", base.normalize_parse_text(text))
    if match:
        out["object_type"] = base.normalize_object_type(match.group(1))
        out["side"] = match.group(2)
    return out


def parse_3_1_1_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    match = re.match(
        r"the\s+([a-z0-9_ ]+?)\s+is\s+([a-z-]+)\s+at(?:\s+a\s+speed\s+of)?\s+(-?\d+(?:\.\d+)?)\s+m/s"
        r"(?:\s*(?:and|,)\s*(accelerating|decelerating)\s+at(?:\s+an\s+acceleration\s+of)?\s+(-?\d+(?:\.\d+)?)\s+m/s\^2)?\.?$",
        norm,
    )
    if not match:
        return {}
    out = {
        "object_type": base.normalize_object_type(match.group(1)),
        "motion_state": match.group(2),
        "speed": base.format_float(float(match.group(3))),
    }
    if match.group(5) is not None:
        out["accel_state"] = match.group(4)
        out["acceleration"] = base.format_float(float(match.group(5)))
    return out


def parse_3_1_2_v6(text: str) -> Dict[str, Any]:
    match = re.match(r"the\s+([a-z0-9_ ]+?)\s+is\s+(.+?)(?:\.)?$", base.normalize_parse_text(text))
    if not match:
        return {}
    maneuver = V6_MANEUVER_TEXT_TO_LABEL.get(match.group(2).strip())
    return {
        "object_type": base.normalize_object_type(match.group(1)),
        "maneuver": maneuver,
    }


def parse_3_2_2_v6(text: str) -> Dict[str, Any]:
    match = re.match(r"the\s+([a-z0-9_ ]+?)\s+is likely to\s+(.+?)(?:\.)?$", base.normalize_parse_text(text))
    if not match:
        return {}
    return {
        "object_type": base.normalize_object_type(match.group(1)),
        "future_region": base.FUTURE_REGION_TEXT_TO_LABEL.get(match.group(2).strip()),
    }


def parse_3_4_2_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    candidates = [norm.rstrip(".")]
    for prefix in (
        "the most probable participant is ",
        "the most likely participant is ",
        "the likely conflict participant is ",
        "the conflict partner is ",
    ):
        if norm.startswith(prefix):
            candidates.insert(0, norm[len(prefix):].rstrip("."))
            break
    parsed: Dict[str, Any] = {}
    for candidate in candidates:
        parsed = _extract_natural_object_phrase(candidate)
        if parsed:
            break
    if not parsed:
        return {}
    out = {"conflict_partner_type": parsed["object_type"]}
    if parsed.get("side") is not None:
        out["conflict_partner_side"] = parsed["side"]
    if parsed.get("lane_function") is not None:
        out["conflict_partner_lane_function"] = parsed["lane_function"]
    return out


def parse_3_4_3_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    reason_match = re.search(r"\b(?:because of|due to)\b", norm)
    if not reason_match:
        return {}
    object_part = norm[: reason_match.start()].rstrip(" ,.")
    for prefix in (
        "the primary risk subject is ",
        "primary risk subject is ",
        "the risk subject is ",
    ):
        if object_part.startswith(prefix):
            object_part = object_part[len(prefix) :]
            break
    reason_part = norm[reason_match.end() :].strip(" .")
    parsed = _extract_natural_object_phrase(object_part)
    if not parsed:
        return {}
    out = {
        "subject_type": parsed["object_type"],
        "risk_reason": base.RISK_REASON_TEXT_TO_LABEL.get(reason_part.strip(), reason_part.strip().replace(" ", "_")),
    }
    if parsed.get("side") is not None:
        out["subject_side"] = parsed["side"]
    if parsed.get("lane_function") is not None:
        out["subject_lane_function"] = parsed["lane_function"]
    return out

def parse_3_4_4_v6(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the dominant conflict pattern is\s+([a-z0-9_-]+)\s+(?:on the\s+(north|south|east|west)\s+approach|in the center of the intersection)\.?$",
        base.normalize_parse_text(text),
    )
    if not match:
        return {}
    side = match.group(2) if match.group(2) else "center"
    return {"interaction_pattern": match.group(1), "side": side}


def parse_4_2_1_v6(text: str) -> Dict[str, Any]:
    risk = base.parse_yes_no(text)
    if risk is None:
        return {}
    out = {"has_speeding_risk": risk}
    if not risk:
        return out
    norm = base.normalize_parse_text(text)
    match = re.match(
        r"yes[.]?\s+a[n]?\s+([a-z0-9_ ]+?)\s+(?:on the\s+(north|south|east|west)\s+approach|in the center of the intersection)\s+"
        r"(?:appears to be\s+)?(?:traveling|moving)(?:\s+at\s+high\s+speed)?\s+at(?:\s+about)?\s+(-?\d+(?:\.\d+)?)\s+m/s\.?$",
        norm,
    )
    if not match:
        return out
    side = match.group(2) if match.group(2) else "center"
    out["vehicle_type"] = base.normalize_object_type(match.group(1))
    out["side"] = side
    out["numeric_targets"] = {"speed_mps": base.format_float(float(match.group(3)))}
    return out


def parse_4_2_2_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    label_text = None
    abnormal_phrases = (
        (("proximity interaction", "abnormal proximity"), "proximity interaction"),
        (("crosswalk blocking", "crosswalk_blocking"), "crosswalk_blocking"),
        (("lingering pedestrian", "lingering_pedestrian"), "lingering_pedestrian"),
        (("speeding",), "speeding"),
        (("stopline overrun", "stopline_overrun", "stop-line overrun"), "stopline_overrun"),
        (("wrong way two wheeler", "wrong_way_two_wheeler", "wrong-way two-wheeler"), "wrong_way_two_wheeler"),
        (("queue spillback", "queue_spillback"), "queue_spillback"),
    )
    for phrases, candidate in abnormal_phrases:
        if any(phrase in norm for phrase in phrases):
            label_text = candidate
            break
    if label_text is None:
        return {}
    region = base.extract_region_focus(norm)
    if region is None:
        return {}
    return {
        "notable_abnormal": V6_ABNORMAL_TEXT_TO_LABEL.get(label_text),
        "risk_region": region,
    }


def parse_4_3_1_v6(text: str) -> Dict[str, Any]:
    return base.parse_action_state_by_phrases(text, V6_INTERSECTION_ACTION_BY_TEXT, V6_INTERSECTION_ACTION_PHRASES)


def parse_4_3_2_v6(text: str) -> Dict[str, Any]:
    return base.parse_action_state_by_phrases(text, V6_SIDE_ACTION_BY_TEXT, V6_SIDE_ACTION_PHRASES)


def parse_4_3_3_v6(text: str) -> Dict[str, Any]:
    return base.parse_action_state_by_phrases(text, V6_LANE_ACTION_BY_TEXT, V6_LANE_ACTION_PHRASES)


def parse_4_3_4_v6(text: str) -> Dict[str, Any]:
    norm = base.normalize_parse_text(text)
    match = re.match(r"that\s+([a-z0-9_ ]+?)\s+(.+)$", norm)
    if not match:
        return {}
    action_state = None
    for suffix, state in V6_OBJECT_ACTION_SUFFIX_TO_STATE.items():
        if norm.endswith(suffix):
            action_state = state
            break
    return {"object_type": base.normalize_object_type(match.group(1)), "action_state": action_state}


def get_numeric_fields_exists_count(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    return base.numeric_bucket_map(count=["count"]) if gt.get("exists") else {}


def get_discrete_fields_crosswalk_blocking_v6(gt: Dict[str, Any]) -> List[str]:
    fields = ["crosswalk_blocked"]
    if gt.get("crosswalk_blocked"):
        fields.extend(["object_type", "side"])
    return fields


def get_discrete_fields_4_2_1_v6(gt: Dict[str, Any]) -> List[str]:
    fields = ["has_speeding_risk"]
    if gt.get("has_speeding_risk"):
        fields.extend(["vehicle_type", "side"])
    return fields


def get_numeric_fields_4_2_1_v6(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    return base.numeric_bucket_map(speed=["numeric_targets.speed_mps"]) if gt.get("has_speeding_risk") else {}


V6_TEMPLATE_SPECS: Dict[str, base.TemplateEvalSpec] = {
    "1_1_1_lane_first_object_type": base.TemplateEvalSpec(parse_1_1_1_v6, discrete_fields=lambda _gt: ["object_type"]),
    "1_1_2_front_neighbor_type": base.TemplateEvalSpec(parse_1_1_2_v6, discrete_fields=lambda _gt: ["target_object_type"]),
    "1_1_3_approach_vru_exists": base.TemplateEvalSpec(parse_1_1_3_v6, discrete_fields=lambda _gt: ["exists"], numeric_fields=get_numeric_fields_exists_count),
    "1_1_4_approach_type_count": base.TemplateEvalSpec(base.parse_1_1_3, discrete_fields=lambda _gt: ["side", "object_type"], numeric_fields=lambda _gt: base.numeric_bucket_map(count=["count"])),
    "1_2_1_size_bucket": base.TemplateEvalSpec(parse_1_2_1_v6, discrete_fields=lambda _gt: ["object_type", "size_bucket"]),
    "1_2_2_visibility": base.TemplateEvalSpec(parse_1_2_2_v6, discrete_fields=lambda _gt: ["visibility"]),
    "1_3_1_environment": base.TemplateEvalSpec(parse_1_3_1_v6, discrete_fields=base.get_discrete_fields_1_3_1),
    "1_3_2_vehicle_signal_state": base.TemplateEvalSpec(base.parse_1_3_2, discrete_fields=lambda _gt: ["signal_state"]),
    "2_1_1_stopline_distance": base.TemplateEvalSpec(parse_2_1_1_v6, discrete_fields=lambda _gt: ["object_type"], numeric_fields=lambda _gt: base.numeric_bucket_map(distance=["distance_m"])),
    "2_1_2_ped_to_far_edge": base.TemplateEvalSpec(parse_2_1_2_v6, numeric_fields=lambda _gt: base.numeric_bucket_map(distance=["distance_m"])),
    "2_1_3_participant_distance": base.TemplateEvalSpec(parse_2_1_3_v6, discrete_fields=lambda _gt: ["obj1_type", "obj2_type"], numeric_fields=lambda _gt: base.numeric_bucket_map(distance=["distance_m"])),
    "2_1_4_nearest_vehicle": base.TemplateEvalSpec(parse_2_1_4_v6, discrete_fields=lambda _gt: ["vehicle_type"], numeric_fields=lambda _gt: base.numeric_bucket_map(distance=["distance_m"])),
    "2_2_1_ped_zone": base.TemplateEvalSpec(parse_2_2_1_v6, discrete_fields=lambda _gt: ["ped_zone"]),
    "2_2_2_lane_queue_count": base.TemplateEvalSpec(base.parse_2_2_3, discrete_fields=lambda _gt: ["lane_function", "side"], numeric_fields=lambda _gt: base.numeric_bucket_map(count=["count"])),
    "2_2_3_stopline_back_5m_count": base.TemplateEvalSpec(base.parse_2_2_4, discrete_fields=lambda _gt: ["side"], numeric_fields=lambda _gt: base.numeric_bucket_map(count=["count"])),
    "2_2_4_longest_queue_lane": base.TemplateEvalSpec(base.parse_2_2_5, discrete_fields=lambda _gt: ["lane_function", "side"]),
    "2_2_5_crosswalk_blocking": base.TemplateEvalSpec(parse_2_2_5_v6, discrete_fields=get_discrete_fields_crosswalk_blocking_v6),
    "3_1_1_current_motion_state": base.TemplateEvalSpec(parse_3_1_1_v6, discrete_fields=lambda _gt: ["object_type", "motion_state", "accel_state"], numeric_fields=lambda _gt: base.numeric_bucket_map(speed=["speed"], acceleration=["acceleration"])),
    "3_1_2_vehicle_maneuver": base.TemplateEvalSpec(parse_3_1_2_v6, discrete_fields=lambda _gt: ["object_type", "maneuver"]),
    "3_2_1_waypoints": base.TemplateEvalSpec(base.parse_3_2_3, numeric_fields=lambda _gt: base.numeric_bucket_map(waypoint_xy=["trajectory.waypoints_xy[*].dx", "trajectory.waypoints_xy[*].dy"])),
    "3_2_2_future_region": base.TemplateEvalSpec(parse_3_2_2_v6, discrete_fields=lambda _gt: ["object_type", "future_region"]),
    "3_3_1_safe_following": base.TemplateEvalSpec(base.parse_3_3_1, discrete_fields=lambda _gt: ["is_safe"], numeric_fields=lambda _gt: base.numeric_bucket_map(distance=["distance_m"])),
    "3_3_2_likely_long_queue_lane": base.TemplateEvalSpec(base.parse_3_3_2, discrete_fields=lambda _gt: ["lane_function", "side"], numeric_fields=base.get_numeric_fields_3_3_2),
    "3_4_1_pair_conflict": base.TemplateEvalSpec(base.parse_3_4_1, discrete_fields=lambda _gt: ["has_conflict"]),
    "3_4_2_nearest_conflict_participant": base.TemplateEvalSpec(parse_3_4_2_v6, discrete_fields=lambda _gt: ["conflict_partner_type"]),
    "3_4_3_primary_risk_subject": base.TemplateEvalSpec(parse_3_4_3_v6, discrete_fields=lambda _gt: ["subject_type", "risk_reason"]),
    "3_4_4_risk_pattern": base.TemplateEvalSpec(parse_3_4_4_v6, discrete_fields=lambda _gt: ["interaction_pattern", "side"]),
    "4_1_1_overall_state": base.TemplateEvalSpec(base.parse_4_1_1, discrete_fields=lambda _gt: ["overall_state"], numeric_fields=lambda _gt: base.numeric_bucket_map(count=["moving_vehicles", "total_vehicles"])),
    "4_1_2_approach_motion_status": base.TemplateEvalSpec(base.parse_4_1_2, discrete_fields=lambda _gt: ["side", "motion_label"], numeric_fields=base.get_numeric_fields_4_1_2),
    "4_1_3_scene_summary": base.TemplateEvalSpec(base.parse_4_1_3, discrete_fields=base.get_discrete_fields_4_1_3),
    "4_1_4_heaviest_traffic_approach": base.TemplateEvalSpec(base.parse_4_1_4, discrete_fields=base.get_discrete_fields_4_1_4, numeric_fields=base.get_numeric_fields_4_1_4),
    "4_2_1_speeding_risk": base.TemplateEvalSpec(parse_4_2_1_v6, discrete_fields=get_discrete_fields_4_2_1_v6, numeric_fields=get_numeric_fields_4_2_1_v6),
    "4_2_2_notable_abnormal": base.TemplateEvalSpec(parse_4_2_2_v6, discrete_fields=lambda _gt: ["notable_abnormal", "risk_region"]),
    "4_3_1_intersection_action": base.TemplateEvalSpec(parse_4_3_1_v6, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_2_approach_action": base.TemplateEvalSpec(parse_4_3_2_v6, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_3_lane_action": base.TemplateEvalSpec(parse_4_3_3_v6, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_4_object_action": base.TemplateEvalSpec(parse_4_3_4_v6, discrete_fields=lambda _gt: ["object_type", "action_state"]),
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_rows = base.read_sidecar(args.sidecar_jsonl, args.split, args.limit)
    predictions = base.read_predictions(args.predictions, sidecar_rows)
    records = base.build_records(sidecar_rows, predictions)

    unknown_subtemplates = sorted({record.subtemplate for record in records if record.subtemplate not in V6_TEMPLATE_SPECS})
    base.require(not unknown_subtemplates, f"Unsupported V6 subtemplates in evaluation input: {unknown_subtemplates}")

    text_metrics = base.compute_text_metrics(records)
    if not args.skip_semantic_metrics:
        preds = [record.prediction for record in records]
        refs = [record.answer for record in records]
        bert_scores = base.BertScoreScorer(args.bertscore_model, args.device, args.batch_size).score_pairs(preds, refs)
        sim_scores = base.SentenceSimilarityScorer(args.sim_model, args.device, args.batch_size).score_pairs(preds, refs)
        for metrics, bert, sim in zip(text_metrics, bert_scores, sim_scores):
            metrics["bertscore"] = bert
            metrics["simcse"] = sim
    for metrics in text_metrics:
        metrics["weighted_text_score"] = base.weighted_average(metrics, base.TEXT_METRIC_WEIGHTS)

    per_sample_rows = []
    for record, sample_text_metrics in zip(records, text_metrics):
        spec = V6_TEMPLATE_SPECS[record.subtemplate]
        parsed_prediction = spec.parser(record.prediction)
        discrete_metrics = base.evaluate_discrete_fields(record.structured_targets, parsed_prediction, spec.discrete_fields(record.structured_targets))
        numeric_metrics = base.evaluate_numeric_fields_by_bucket(
            record.structured_targets,
            parsed_prediction,
            spec.numeric_fields(record.structured_targets),
        )
        per_sample_rows.append(
            {
                "question_id": record.question_id,
                "scene_id": record.scene_id,
                "chapter": record.chapter,
                "section": record.section,
                "subtemplate": record.subtemplate,
                "reference": record.answer,
                "prediction": record.prediction,
                "text_metrics": sample_text_metrics,
                "numeric_metrics": numeric_metrics,
                "discrete_metrics": discrete_metrics,
                "structured_targets": record.structured_targets,
                "parsed_prediction": parsed_prediction,
            }
        )

    per_subtemplate_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_chapter_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_section_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        per_subtemplate_rows[row["subtemplate"]].append(row)
        per_chapter_rows[row["chapter"]].append(row)
        per_section_rows[row["section"]].append(row)

    per_subtemplate = {name: base.summarize_rows(rows) for name, rows in sorted(per_subtemplate_rows.items())}
    per_chapter = {name: base.summarize_rows(rows) for name, rows in sorted(per_chapter_rows.items())}
    per_section = {name: base.summarize_rows(rows) for name, rows in sorted(per_section_rows.items())}

    metrics_payload = {
        "metadata": {
            "sidecar_jsonl": str(args.sidecar_jsonl.resolve()),
            "predictions": str(args.predictions.resolve()),
            "split": args.split,
            "sample_count": len(records),
            "semantic_metrics_enabled": not args.skip_semantic_metrics,
            "bertscore_model": None if args.skip_semantic_metrics else args.bertscore_model,
            "sim_model": None if args.skip_semantic_metrics else args.sim_model,
            "text_metric_weights": base.TEXT_METRIC_WEIGHTS,
            "template_version": "v6",
            "evaluation_schema": {
                "text": "normalized_exact_match, bleu_4, rouge_l, bertscore, simcse, weighted_text_score",
                "numeric_error": "V6 top-level numeric metrics retain valid, expected, coverage, total_bucket_error, and by_bucket; each by_bucket entry retains mae/rmse diagnostics plus bounded bucket_error in [0,1], where bucket_error = 1 - (1 - raw_bucket_error) * coverage",
                "discrete_semantics": "V6 retains precision, recall, f1, and coverage; f1 plus coverage are the recommended headline view for exact-label semantic correctness and parser extraction coverage over answer-required fields",
            },
            "numeric_total_error_thresholds": base.NUMERIC_TOTAL_ERROR_THRESHOLDS,
            "numeric_error_buckets": {
                "count": "count-like structured numeric fields",
                "distance": "distance values such as distance_m",
                "speed": "speed values such as speed and numeric_targets.speed_mps",
                "acceleration": "acceleration values",
                "global_3d_xy": "scene/global ground-plane Euclidean x/y distance",
                "image_2d_xy": "image-plane Euclidean x1/y1 pixel distance",
                "waypoint_xy": "future waypoint Euclidean dx/dy offset distance",
            },
        },
        "overall_text": base.aggregate_text_group(per_sample_rows),
        "overall_numeric_error": base.public_numeric_metrics(base.aggregate_numeric_group(per_sample_rows)),
        "overall_discrete_semantics": base.aggregate_discrete_group(per_sample_rows),
        "macro_by_subtemplate": base.aggregate_macro(per_subtemplate),
        "per_subtemplate": per_subtemplate,
        "per_chapter": per_chapter,
        "per_section": per_section,
    }

    per_sample_output_rows = []
    for row in per_sample_rows:
        row_out = dict(row)
        row_out["numeric_metrics"] = base.public_numeric_metrics(row["numeric_metrics"])
        per_sample_output_rows.append(row_out)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample_output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote metrics to {args.output_dir / 'metrics.json'}")
    print(f"Wrote per-sample results to {args.output_dir / 'per_sample_results.jsonl'}")


if __name__ == "__main__":
    main()
