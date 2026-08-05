from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from nuscenesqa_v5.utils.io import ensure

INTERSECTION_ACTION_TEXT_BY_STATE = {
    "FLOW_STABLE": "Traffic should continue in a coordinated and orderly way.",
    "FLOW_CALMING": "Traffic should move more calmly and with reduced speed.",
    "QUEUE_MANAGEMENT": "Traffic should proceed in a more orderly and tightly managed way.",
    "CONFLICT_SUPPRESSION": "Traffic should reduce aggressive movement and prioritize conflict avoidance.",
}

SIDE_ACTION_TEXT_BY_STATE = {
    "SIDE_CLEARANCE_PROTECTION": "Traffic should keep safer local spacing.",
    "SIDE_SPEED_MODERATION": "Traffic should moderate speed and maintain safer spacing.",
    "SIDE_GENERAL_CAUTION": "Traffic should proceed cautiously and remain orderly.",
    "SIDE_QUEUE_STABILIZATION": "Traffic should stabilize queue movement and avoid unnecessary disruption.",
    "SIDE_CROSSING_AWARENESS": "Traffic should yield clearly to crossing activity.",
}

LANE_ACTION_TEXT_BY_STATE = {
    "LANE_CLEARANCE_MAINTENANCE": "Traffic should maintain clearance and avoid tight local conflicts.",
    "LANE_PREPARE_TO_STOP": "Traffic should prepare to stop and avoid pressing forward.",
    "LANE_QUEUE_PRESERVATION": "Traffic should preserve queue order and avoid disruption.",
    "LANE_GENERAL_ORDER": "Traffic should proceed in an orderly manner.",
    "LANE_SPEED_REDUCTION": "Traffic should reduce speed and proceed more conservatively.",
}

OBJECT_ACTION_SUFFIX_BY_STATE = {
    "OBJECT_YIELD_NOW": "should yield now.",
    "OBJECT_SLOW_DOWN": "should slow down and keep safer local spacing.",
    "OBJECT_PREPARE_TO_STOP": "should prepare to stop.",
    "OBJECT_PROCEED_CAUTIOUSLY": "should proceed cautiously.",
}

MANEUVER_TEXT_BY_LABEL = {
    "straight": "going straight",
    "left turn": "making a left turn",
    "right turn": "making a right turn",
    "lane change": "changing lanes",
    "stop-and-wait": "stopping and waiting",
}

FUTURE_REGION_TEXT_BY_LABEL = {
    "before stop line": "remain before the stop line",
    "intersection center": "move into the intersection center",
    "left-turn exit": "move toward the left-turn exit",
    "through exit": "move toward the through exit",
    "right-turn exit": "move toward the right-turn exit",
}

ABNORMAL_TEXT_BY_LABEL = {
    "abnormal_proximity": "high-risk abnormal proximity interaction",
    "speeding": "high-risk speeding event",
    "crosswalk_blocking": "crosswalk-blocking event",
    "lingering_pedestrian": "lingering-pedestrian event",
    "stopline_overrun": "stop-line overrun event",
    "wrong_way_two_wheeler": "wrong-way two-wheeler event",
}

RISK_REASON_TEXT_BY_LABEL = {
    "path_crossing": "path crossing",
    "overspeed": "overspeed",
    "vru_conflict": "vru conflict",
    "lane_change_conflict": "lane change conflict",
    "proximity": "proximity",
}

V5_SUBTEMPLATE_PREFIXES = {
    "1_1_1_fine_type": "It is a ",
    "1_1_2_side_exists": "",
    "1_1_3_side_count": "The ",
    "1_1_4_relative_neighbor_type": "LOC_REF:",
    "1_2_1_size_bucket": "The ",
    "1_2_2_visibility": "",
    "1_3_1_weather": "The scene appears to be ",
    "1_3_2_vehicle_signal_state": "The signal is currently ",
    "2_1_1_stopline_distance": "STOPLINE_DIST:",
    "2_1_2_ped_to_far_edge": "PED_EXIT_DIST:",
    "2_1_3_participant_distance": "PAIR_DIST:",
    "2_1_4_nearest_vehicle_to_ped": "PED_NEAREST_VEH:",
    "2_2_1_lane_function": "The ",
    "2_2_2_ped_zone": "The pedestrian is currently ",
    "2_2_3_left_turn_queue_count": "There are ",
    "2_2_4_stopline_back_5m_count": "There are ",
    "2_2_5_longest_queue_lane": "The ",
    "2_2_6_crosswalk_blocking": "",
    "3_1_1_current_motion_state": "MOTION_STATE:",
    "3_1_2_vehicle_maneuver": "The ",
    "3_2_2_future_region": "The ",
    "3_2_3_waypoints": "Future trajectory:",
    "3_3_1_safe_following": "",
    "3_3_2_likely_long_queue_lane": "The ",
    "3_4_1_vehicle_ped_conflict": "",
    "3_4_2_nearest_conflict_participant": "CONFLICT_PARTNER:",
    "3_4_3_primary_risk_subject": "RISK_SUBJECT_REASON:",
    "3_4_4_risk_pattern": "The dominant conflict pattern is ",
    "4_1_1_overall_state": "The intersection is currently ",
    "4_1_2_side_motion_status": "The ",
    "4_1_3_scene_summary": "The most notable current risk is ",
    "4_1_4_flow_imbalance": "The ",
    "4_2_1_speeding_risk": "",
    "4_2_2_notable_abnormal": "A ",
    "4_3_1_intersection_action": "Traffic should ",
    "4_3_2_side_action": "Traffic should ",
    "4_3_3_lane_action": "Traffic should ",
    "4_3_4_object_action": "The ",
}

V6_SUBTEMPLATE_PREFIXES = {
    "1_1_1_lane_first_object_type": "It is a ",
    "1_1_2_front_neighbor_type": "It is a ",
    "1_1_3_approach_vru_exists": "",
    "1_1_4_approach_type_count": "The ",
    "1_2_1_size_bucket": "The ",
    "1_2_2_visibility": "The ",
    "1_3_1_environment": "The scene appears to be ",
    "1_3_2_vehicle_signal_state": "The signal is currently ",
    "2_1_1_stopline_distance": "The ",
    "2_1_2_ped_to_far_edge": "The pedestrian is ",
    "2_1_3_participant_distance": "The ",
    "2_1_4_nearest_vehicle": "It is a ",
    "2_2_1_ped_zone": "The pedestrian is currently ",
    "2_2_2_lane_queue_count": "There are ",
    "2_2_3_stopline_back_5m_count": "There are ",
    "2_2_4_longest_queue_lane": "The ",
    "2_2_5_crosswalk_blocking": "",
    "3_1_1_current_motion_state": "MOTION_STATE_V6:",
    "3_1_2_vehicle_maneuver": "The ",
    "3_2_1_waypoints": "Future trajectory:",
    "3_2_2_future_region": "The ",
    "3_3_1_safe_following": "",
    "3_3_2_likely_long_queue_lane": "The ",
    "3_4_1_pair_conflict": "",
    "3_4_2_nearest_conflict_participant": "The most probable participant is ",
    "3_4_3_primary_risk_subject": "The primary risk subject is ",
    "3_4_4_risk_pattern": "The dominant conflict pattern is ",
    "4_1_1_overall_state": "The intersection is currently ",
    "4_1_2_approach_motion_status": "The ",
    "4_1_3_scene_summary": "The most notable current risk is ",
    "4_1_4_heaviest_traffic_approach": "The ",
    "4_2_1_speeding_risk": "SPEEDING_RISK_V6:",
    "4_2_2_notable_abnormal": "A high-risk abnormal ",
    "4_3_1_intersection_action": "Traffic should ",
    "4_3_2_approach_action": "Traffic should ",
    "4_3_3_lane_action": "Traffic should ",
    "4_3_4_object_action": "That ",
}

PREFIXES_BY_VERSION = {
    "v5": V5_SUBTEMPLATE_PREFIXES,
    "v6": V6_SUBTEMPLATE_PREFIXES,
}

DEFAULT_PREFIX = ""
COMPACT_TERMINATOR = "#"
MOTION_STATE_ACCEL_PREFIX = "MOTION_STATE_ACC:"
RISK_SUBJECT_REASON_PREFIX = "RISK_SUBJECT_REASON:"
MOTION_STATE_V6_PREFIX = "MOTION_STATE_V6:"
MOTION_STATE_ACCEL_V6_PREFIX = "MOTION_STATE_ACC_V6:"
SPEEDING_RISK_V6_PREFIX = "SPEEDING_RISK_V6:"

ONE_SENTENCE_TEMPLATES = {
    "1_1_1_fine_type",
    "1_1_2_side_exists",
    "1_1_3_side_count",
    "1_1_4_relative_neighbor_type",
    "1_2_1_size_bucket",
    "1_2_2_visibility",
    "1_3_2_vehicle_signal_state",
    "2_1_1_stopline_distance",
    "2_1_2_ped_to_far_edge",
    "2_1_3_participant_distance",
    "2_1_4_nearest_vehicle_to_ped",
    "2_2_1_lane_function",
    "2_2_2_ped_zone",
    "2_2_3_left_turn_queue_count",
    "2_2_4_stopline_back_5m_count",
    "2_2_5_longest_queue_lane",
    "2_2_6_crosswalk_blocking",
    "3_1_1_current_motion_state",
    "3_1_2_vehicle_maneuver",
    "3_2_2_future_region",
    "3_3_1_safe_following",
    "3_3_2_likely_long_queue_lane",
    "3_4_1_vehicle_ped_conflict",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "3_4_4_risk_pattern",
    "4_1_1_overall_state",
    "4_1_2_side_motion_status",
    "4_1_4_flow_imbalance",
    "4_2_1_speeding_risk",
    "4_2_2_notable_abnormal",
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

V5_STRICT_LOCATION_TEMPLATES = {
    "1_1_4_relative_neighbor_type",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
}

V6_STRICT_LOCATION_TEMPLATES: set[str] = set()

STRICT_LOCATION_TEMPLATES_BY_VERSION = {
    "v5": V5_STRICT_LOCATION_TEMPLATES,
    "v6": V6_STRICT_LOCATION_TEMPLATES,
}

V5_YES_NO_TEMPLATES = {
    "1_1_2_side_exists",
    "2_2_6_crosswalk_blocking",
    "3_3_1_safe_following",
    "3_4_1_vehicle_ped_conflict",
    "4_2_1_speeding_risk",
}

V6_YES_NO_TEMPLATES = {
    "1_1_3_approach_vru_exists",
    "2_2_5_crosswalk_blocking",
    "3_3_1_safe_following",
    "3_4_1_pair_conflict",
}

YES_NO_TEMPLATES_BY_VERSION = {
    "v5": V5_YES_NO_TEMPLATES,
    "v6": V6_YES_NO_TEMPLATES,
}

V5_ACTION_TEMPLATES = {
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

V6_ACTION_TEMPLATES = {
    "4_3_1_intersection_action",
    "4_3_2_approach_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

ACTION_TEMPLATES_BY_VERSION = {
    "v5": V5_ACTION_TEMPLATES,
    "v6": V6_ACTION_TEMPLATES,
}

V5_COMPACT_SUPERVISION_TEMPLATES = {
    "1_1_4_relative_neighbor_type",
    "2_1_1_stopline_distance",
    "2_1_2_ped_to_far_edge",
    "2_1_3_participant_distance",
    "2_1_4_nearest_vehicle_to_ped",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
}

V6_COMPACT_SUPERVISION_TEMPLATES = {
    "3_1_1_current_motion_state",
    "4_2_1_speeding_risk",
}

COMPACT_SUPERVISION_TEMPLATES_BY_VERSION = {
    "v5": V5_COMPACT_SUPERVISION_TEMPLATES,
    "v6": V6_COMPACT_SUPERVISION_TEMPLATES,
}

V6_ONE_SENTENCE_TEMPLATES = {
    "1_1_1_lane_first_object_type",
    "1_1_2_front_neighbor_type",
    "1_1_3_approach_vru_exists",
    "1_1_4_approach_type_count",
    "1_2_1_size_bucket",
    "1_2_2_visibility",
    "1_3_1_environment",
    "1_3_2_vehicle_signal_state",
    "2_1_1_stopline_distance",
    "2_1_2_ped_to_far_edge",
    "2_1_3_participant_distance",
    "2_1_4_nearest_vehicle",
    "2_2_1_ped_zone",
    "2_2_2_lane_queue_count",
    "2_2_3_stopline_back_5m_count",
    "2_2_4_longest_queue_lane",
    "2_2_5_crosswalk_blocking",
    "3_1_1_current_motion_state",
    "3_1_2_vehicle_maneuver",
    "3_2_2_future_region",
    "3_3_1_safe_following",
    "3_3_2_likely_long_queue_lane",
    "3_4_1_pair_conflict",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "3_4_4_risk_pattern",
    "4_1_1_overall_state",
    "4_1_2_approach_motion_status",
    "4_1_3_scene_summary",
    "4_1_4_heaviest_traffic_approach",
    "4_2_1_speeding_risk",
    "4_2_2_notable_abnormal",
    "4_3_1_intersection_action",
    "4_3_2_approach_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

ONE_SENTENCE_TEMPLATES_BY_VERSION = {
    "v5": ONE_SENTENCE_TEMPLATES,
    "v6": V6_ONE_SENTENCE_TEMPLATES,
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_parse_text(text: str) -> str:
    return clean_text(text).lower()


def maybe_text(value: Any) -> str:
    if value is None:
        return ""
    text = clean_text(value)
    return "" if text.lower() == "none" else text


def normalize_dataset_version(dataset_version: str) -> str:
    version = clean_text(dataset_version).lower() or "v5"
    ensure(version in {"v5", "v6"}, f"Unsupported dataset version: {dataset_version}")
    return version


def subtemplate_prefix_map(dataset_version: str) -> Mapping[str, str]:
    return PREFIXES_BY_VERSION[normalize_dataset_version(dataset_version)]


def prefix_for_subtemplate(subtemplate: str, dataset_version: str) -> str:
    return subtemplate_prefix_map(dataset_version).get(subtemplate, DEFAULT_PREFIX)


def object_label_text(value: Any) -> str:
    return clean_text(str(value)).replace("_", " ")


def pluralize(label: str, count: int) -> str:
    plain = object_label_text(label)
    if count == 1:
        return plain
    if plain == "bus":
        return "buses"
    if plain.endswith("y") and not plain.endswith(("ay", "ey", "iy", "oy", "uy")):
        return plain[:-1] + "ies"
    if plain.endswith("s"):
        return plain + "es"
    return plain + "s"


def format_decimal(value: Any, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}"


def first_image_ref(image_refs: Iterable[Mapping[str, Any]] | None) -> Mapping[str, Any]:
    refs = list(image_refs or [])
    ensure(refs, "Expected at least one image reference.")
    ref = refs[0]
    ensure("image_name" in ref and "x1" in ref and "y1" in ref, f"Invalid image ref: {ref}")
    return ref


def side_phrase(side: str | None) -> str:
    value = clean_text(side or "")
    if value == "center":
        return "in the center area"
    ensure(value in {"north", "south", "east", "west"}, f"Unsupported side value: {side}")
    return f"on the {value} approach"


def v6_side_phrase(side: str | None) -> str:
    value = clean_text(side or "")
    if value == "center":
        return "in the center of the intersection"
    ensure(value in {"north", "south", "east", "west"}, f"Unsupported side value: {side}")
    return f"on the {value} approach"


def indefinite_article(label: str) -> str:
    plain = object_label_text(label)
    return "an" if plain[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def side_from_image_refs(image_refs: Iterable[Mapping[str, Any]] | None) -> str | None:
    refs = list(image_refs or [])
    if not refs:
        return None
    image_name = clean_text(str(refs[0].get("image_name", "")))
    if image_name.endswith("_image"):
        side = image_name[:-6]
        if side in {"north", "south", "east", "west"}:
            return side
    return None


def format_location_blob(position: Mapping[str, Any], image_refs: Iterable[Mapping[str, Any]]) -> str:
    ref = first_image_ref(image_refs)
    return (
        "<"
        f"{format_decimal(position['x'])},"
        f"{format_decimal(position['y'])},"
        f"{clean_text(ref['image_name'])},"
        f"{format_decimal(ref['x1'])},"
        f"{format_decimal(ref['y1'])}"
        ">"
    )


def format_reason_text(reason: Any) -> str:
    key = clean_text(str(reason))
    return RISK_REASON_TEXT_BY_LABEL.get(key, key.replace("_", " "))


def format_abnormal_text(label: Any) -> str:
    key = clean_text(str(label))
    return ABNORMAL_TEXT_BY_LABEL.get(key, key.replace("_", " "))


def dedupe_repeated_tail(text: str) -> str:
    normalized = clean_text(text)
    terminal = "." if normalized.endswith(".") else ""
    core = normalized[:-1] if terminal else normalized
    tokens = core.split()
    if not tokens:
        return ""
    changed = True
    while changed:
        changed = False
        max_seg = min(12, len(tokens) // 2)
        for seg_len in range(1, max_seg + 1):
            if tokens[-2 * seg_len : -seg_len] == tokens[-seg_len:]:
                tokens = tokens[:-seg_len]
                changed = True
                break
    return (" ".join(tokens) + terminal).strip()


def first_sentence(text: str) -> str:
    normalized = clean_text(text)
    match = re.search(r"\.(?=\s+[A-Z]|$)", normalized)
    if match:
        return normalized[: match.start() + 1].strip()
    return normalized


LOCATION_BLOB_REGEX = re.compile(
    r"<\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(north_image|south_image|east_image|west_image)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*>",
    flags=re.IGNORECASE,
)


def extract_first_location_blob(text: str) -> str | None:
    match = LOCATION_BLOB_REGEX.search(text)
    if not match:
        return None
    return (
        "<"
        f"{format_decimal(match.group(1))},"
        f"{format_decimal(match.group(2))},"
        f"{clean_text(match.group(3)).lower()},"
        f"{format_decimal(match.group(4))},"
        f"{format_decimal(match.group(5))}"
        ">"
    )


def extract_yes_no(text: str) -> str | None:
    norm = clean_text(text).lower()
    if norm.startswith("yes"):
        return "Yes."
    if norm.startswith("no"):
        return "No."
    return None


def normalize_side_token(value: Any) -> str:
    token = clean_text(str(value)).lower()
    ensure(token in {"north", "south", "east", "west", "center"}, f"Unsupported side token: {value}")
    return token


def is_numeric_token(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", clean_text(value)))


def is_image_token(value: str) -> bool:
    return clean_text(value).lower() in {"north_image", "south_image", "east_image", "west_image"}


def split_compact_fields(text: str, prefix: str) -> list[str]:
    body = strip_prefix(text, prefix)
    if COMPACT_TERMINATOR in body:
        body = body.split(COMPACT_TERMINATOR, 1)[0]
    return [clean_text(part) for part in body.split("|") if clean_text(part)]


def find_first_side(parts: list[str], start: int = 0) -> tuple[str | None, int | None]:
    for index in range(start, len(parts)):
        token = clean_text(parts[index]).lower()
        if token in {"north", "south", "east", "west", "center"}:
            return token, index
    return None, None


def find_first_numeric(parts: list[str], start: int = 0) -> tuple[str | None, int | None]:
    for index in range(start, len(parts)):
        token = clean_text(parts[index])
        if is_numeric_token(token):
            return token, index
    return None, None


def find_first_image(parts: list[str], start: int = 0) -> tuple[str | None, int | None]:
    for index in range(start, len(parts)):
        token = clean_text(parts[index]).lower()
        if is_image_token(token):
            return token, index
    return None, None


def is_known_reason_token(value: str) -> bool:
    normalized = clean_text(value).lower().replace("_", " ")
    return normalized in set(RISK_REASON_TEXT_BY_LABEL.values())


def parse_side_from_phrase(text: str) -> str:
    normalized = clean_text(text).lower()
    if "center area" in normalized:
        return "center"
    match = re.search(r"\bon the\s+(north|south|east|west)\s+approach\b", normalized)
    ensure(match is not None, f"Unable to parse side phrase from answer: {text}")
    return str(match.group(1))


def strip_prefix(text: str, prefix: str) -> str:
    normalized = clean_text(text)
    if prefix and normalized.startswith(prefix):
        return normalized[len(prefix) :].strip()
    return normalized


def compact_location_payload(object_type: str, position: Mapping[str, Any], image_refs: Iterable[Mapping[str, Any]]) -> str:
    ref = first_image_ref(image_refs)
    return "|".join(
        [
            object_label_text(object_type),
            format_decimal(position["x"]),
            format_decimal(position["y"]),
            clean_text(ref["image_name"]).lower(),
            format_decimal(ref["x1"]),
            format_decimal(ref["y1"]),
        ]
    )


def render_location_from_payload(prefix: str, payload: str) -> str | None:
    parts = split_compact_fields(payload, prefix)
    if len(parts) != 6:
        return None
    object_type, x, y, image_name, x1, y1 = parts
    if image_name not in {"north_image", "south_image", "east_image", "west_image"}:
        return None
    try:
        blob = (
            "<"
            f"{format_decimal(float(x))},"
            f"{format_decimal(float(y))},"
            f"{image_name},"
            f"{format_decimal(float(x1))},"
            f"{format_decimal(float(y1))}"
            ">"
        )
    except ValueError:
        return None
    return object_type, blob


def render_compact_prediction(subtemplate: str, prediction: str, prefix_override: str | None = None) -> str | None:
    prefix = prefix_override if prefix_override is not None else V5_SUBTEMPLATE_PREFIXES.get(subtemplate, DEFAULT_PREFIX)
    parts = split_compact_fields(prediction, prefix)

    try:
        if subtemplate == "2_1_1_stopline_distance" and len(parts) >= 3:
            object_type = object_label_text(parts[0])
            side, side_index = find_first_side(parts, 1)
            distance_m, _ = find_first_numeric(parts, 1)
            if side is None or distance_m is None:
                return None
            return f"The {object_label_text(object_type)} is {format_decimal(float(distance_m))} m from the stop line on the {normalize_side_token(side)} approach."
        if subtemplate == "2_1_2_ped_to_far_edge" and len(parts) >= 2:
            crosswalk, _ = find_first_side(parts, 0)
            distance_m, _ = find_first_numeric(parts, 0)
            if crosswalk is None or distance_m is None:
                return None
            crosswalk = normalize_side_token(crosswalk)
            return f"The pedestrian on the {crosswalk} crosswalk is {format_decimal(float(distance_m))} m from the exit area."
        if subtemplate == "2_1_3_participant_distance" and len(parts) >= 5:
            obj1_type = object_label_text(parts[0])
            obj1_side, side1_index = find_first_side(parts, 1)
            distance_m, dist_index = find_first_numeric(parts, 1)
            if obj1_side is None or distance_m is None or dist_index is None:
                return None
            obj2_type = None
            for index in range(dist_index + 1, len(parts)):
                token = clean_text(parts[index]).lower()
                if token not in {"north", "south", "east", "west", "center"} and not is_numeric_token(token) and not is_image_token(token):
                    obj2_type = object_label_text(parts[index])
                    obj2_index = index
                    break
            else:
                return None
            obj2_side, _ = find_first_side(parts, obj2_index + 1)
            if obj2_side is None:
                return None
            return (
                f"The {object_label_text(obj1_type)} {side_phrase(normalize_side_token(obj1_side))} "
                f"is {format_decimal(float(distance_m))} m from the {object_label_text(obj2_type)} {side_phrase(normalize_side_token(obj2_side))}."
            )
        if subtemplate == "2_1_4_nearest_vehicle_to_ped" and len(parts) >= 3:
            vehicle_type = object_label_text(parts[0])
            side, _ = find_first_side(parts, 1)
            distance_m, _ = find_first_numeric(parts, 1)
            if side is None or distance_m is None:
                return None
            return f"It is a {object_label_text(vehicle_type)} on the {normalize_side_token(side)} approach, {format_decimal(float(distance_m))} m away."
        if subtemplate == "3_1_1_current_motion_state" and len(parts) >= 4:
            object_type = object_label_text(parts[0])
            side, side_index = find_first_side(parts, 1)
            if side is None or side_index is None:
                return None
            motion_state = clean_text(parts[side_index + 1]) if side_index + 1 < len(parts) else ""
            speed = clean_text(parts[side_index + 2]) if side_index + 2 < len(parts) else ""
            if motion_state in {"north", "south", "east", "west", "center"} or is_numeric_token(motion_state) or not is_numeric_token(speed):
                return None
            answer = f"The {object_label_text(object_type)} {side_phrase(normalize_side_token(side))} is {clean_text(motion_state)} at {format_decimal(float(speed))} m/s"
            if side_index + 4 < len(parts):
                accel_state = clean_text(parts[side_index + 3])
                acceleration = clean_text(parts[side_index + 4])
                if accel_state in {"accelerating", "decelerating"} and is_numeric_token(acceleration):
                    answer += f" and {clean_text(accel_state)} at {format_decimal(float(acceleration))} m/s^2"
            return answer + "."
        if subtemplate == "1_1_4_relative_neighbor_type":
            if len(parts) < 6:
                return None
            object_type, x, y, image_name, x1, y1 = parts[:6]
            location = render_location_from_payload(prefix, prefix + "|".join([object_type, x, y, image_name, x1, y1]))
            if location is None:
                return None
            object_type, blob = location
            return f"It is a {object_label_text(object_type)} at {blob}."
        if subtemplate == "3_4_2_nearest_conflict_participant":
            if len(parts) < 6:
                return None
            object_type, x, y, image_name, x1, y1 = parts[:6]
            location = render_location_from_payload(prefix, prefix + "|".join([object_type, x, y, image_name, x1, y1]))
            if location is None:
                return None
            object_type, blob = location
            return f"The most probable participant is the {object_label_text(object_type)} at {blob}."
        if subtemplate == "3_4_3_primary_risk_subject":
            if prefix == RISK_SUBJECT_REASON_PREFIX and len(parts) >= 7:
                object_type = parts[0]
                reason = clean_text(parts[1]).replace("_", " ")
                x, y, image_name, x1, y1 = parts[2:7]
                rendered = render_location_from_payload(prefix, prefix + "|".join([object_type, x, y, image_name, x1, y1]))
                if rendered is None or not is_known_reason_token(reason):
                    return None
                _, blob = rendered
                return f"The primary risk subject is the {object_label_text(object_type)} at {blob} because of {reason}."
            if len(parts) >= 6:
                object_type, x, y, image_name, x1, y1 = parts[:6]
                rendered = render_location_from_payload(prefix, prefix + "|".join([object_type, x, y, image_name, x1, y1]))
                if rendered is None:
                    return None
                _, blob = rendered
                reason = None
                for token in parts[6:]:
                    if is_known_reason_token(token):
                        reason = clean_text(token).replace("_", " ")
                        break
                if reason is None:
                    return None
                return f"The primary risk subject is the {object_label_text(object_type)} at {blob} because of {reason}."
    except ValueError:
        return None
    return None


def render_compact_prediction_v6(subtemplate: str, prediction: str, prefix_override: str | None = None) -> str | None:
    prefix = prefix_override if prefix_override is not None else V6_SUBTEMPLATE_PREFIXES.get(subtemplate, DEFAULT_PREFIX)
    parts = split_compact_fields(prediction, prefix)
    try:
        if subtemplate == "3_1_1_current_motion_state":
            if len(parts) < 3:
                return None
            object_type = object_label_text(parts[0])
            motion_state = clean_text(parts[1])
            speed = clean_text(parts[2])
            if not motion_state or motion_state in {"north", "south", "east", "west", "center"} or is_numeric_token(motion_state):
                return None
            if not is_numeric_token(speed):
                return None
            answer = f"The {object_type} is {motion_state} at {format_decimal(float(speed))} m/s"
            expects_accel = prefix == MOTION_STATE_ACCEL_V6_PREFIX
            if expects_accel:
                if len(parts) < 5:
                    return None
                accel_state = clean_text(parts[3])
                acceleration = clean_text(parts[4])
                if accel_state not in {"accelerating", "decelerating"} or not is_numeric_token(acceleration):
                    return None
                answer += f" and {accel_state} at {format_decimal(float(acceleration))} m/s^2"
            elif len(parts) >= 5:
                accel_state = clean_text(parts[3])
                acceleration = clean_text(parts[4])
                if accel_state in {"accelerating", "decelerating"} and is_numeric_token(acceleration):
                    answer += f" and {accel_state} at {format_decimal(float(acceleration))} m/s^2"
            return answer + "."
        if subtemplate == "4_2_1_speeding_risk":
            if not parts:
                return None
            flag = clean_text(parts[0]).upper()
            if flag == "NO":
                return "No."
            if flag != "YES" or len(parts) < 4:
                return None
            object_type = object_label_text(parts[1])
            side = normalize_side_token(parts[2])
            speed = clean_text(parts[3])
            if not is_numeric_token(speed):
                return None
            article = indefinite_article(object_type).capitalize()
            return (
                f"Yes. {article} {object_type} {v6_side_phrase(side)} "
                f"appears to be traveling at high speed at about {format_decimal(float(speed))} m/s."
            )
    except ValueError:
        return None
    return None


def render_supervision_answer_v6(subtemplate: str, structured_targets: Mapping[str, Any], default_answer: str) -> str:
    del subtemplate, structured_targets
    answer = clean_text(default_answer)
    ensure(answer, "V6 supervision answer is empty.")
    return answer


def normalize_prediction_v6(
    subtemplate: str,
    prediction: str,
    decoder_prefix: str | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    raw = clean_text(prediction)
    normalized = dedupe_repeated_tail(raw)
    if normalized != raw:
        reasons.append("repeated_tail")

    if subtemplate in COMPACT_SUPERVISION_TEMPLATES_BY_VERSION["v6"]:
        compact_render = render_compact_prediction_v6(subtemplate, normalized, prefix_override=decoder_prefix)
        if compact_render is not None:
            if compact_render != normalized:
                reasons.append("compact_render")
            return compact_render, reasons

    if subtemplate == "4_2_1_speeding_risk":
        yn = extract_yes_no(normalized)
        if yn is not None:
            reasons.append("yes_no_only_without_speed")
            if yn != normalized:
                reasons.append("yes_no_normalized")
            return yn, reasons

    if subtemplate in YES_NO_TEMPLATES_BY_VERSION["v6"]:
        yn = extract_yes_no(normalized)
        if yn is not None:
            if yn != normalized:
                reasons.append("yes_no_normalized")
            return yn, reasons

    if subtemplate == "3_2_1_waypoints":
        coords = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", normalized)
        if coords:
            body = ",".join(f"({format_decimal(x, 2)},{format_decimal(y, 2)})" for x, y in coords)
            rendered = f"Future trajectory:{body}"
            if rendered != normalized:
                reasons.append("waypoint_normalized")
            return rendered, reasons

    if subtemplate in ONE_SENTENCE_TEMPLATES_BY_VERSION["v6"]:
        sentence = first_sentence(normalized)
        if sentence != normalized:
            reasons.append("first_sentence")
        normalized = sentence

    return normalized, reasons


def diagnose_prediction_v6(subtemplate: str, raw_prediction: str, normalized_prediction: str) -> list[str]:
    reasons: list[str] = []
    raw = clean_text(raw_prediction).lower()
    normalized = clean_text(normalized_prediction).lower()
    if subtemplate != "3_3_2_likely_long_queue_lane" and "most likely to form a long queue" in raw:
        reasons.append("wrong_template_collapse")
    if subtemplate == "3_2_1_waypoints" and "(" not in normalized_prediction:
        reasons.append("missing_waypoints")
    if subtemplate in ACTION_TEMPLATES_BY_VERSION["v6"] and "should " not in normalized:
        reasons.append("missing_action_phrase")
    if subtemplate == "3_1_1_current_motion_state":
        if "m/s" not in normalized:
            reasons.append("missing_speed_payload")
        if raw.startswith(MOTION_STATE_V6_PREFIX.lower()) or raw.startswith(MOTION_STATE_ACCEL_V6_PREFIX.lower()):
            if "m/s" not in normalized:
                reasons.append("invalid_motion_schema")
            if raw.startswith(MOTION_STATE_ACCEL_V6_PREFIX.lower()) and "m/s^2" not in normalized:
                reasons.append("missing_acceleration_payload")
    if subtemplate == "4_2_1_speeding_risk":
        if normalized in {"yes.", "no."} and "m/s" not in normalized:
            reasons.append("yes_no_only_without_speed")
        if raw.startswith(SPEEDING_RISK_V6_PREFIX.lower()) and "m/s" not in normalized and normalized != "no.":
            reasons.append("invalid_speeding_schema")
        if raw.startswith("yes") and "m/s" not in raw:
            reasons.append("missing_side_or_speed")
    return reasons


def render_supervision_answer(subtemplate: str, structured_targets: Mapping[str, Any], default_answer: str) -> str:
    st = structured_targets or {}
    answer = clean_text(default_answer)

    if subtemplate == "1_1_1_fine_type":
        return f"It is a {object_label_text(st['object_type'])} located {side_phrase(st['side'])} of the intersection."
    if subtemplate == "1_1_2_side_exists":
        return "Yes." if bool(st.get("exists")) else "No."
    if subtemplate == "1_1_3_side_count":
        count = int(st["count"])
        return f"The {clean_text(st['side'])} approach currently has {count} {pluralize(st['object_type'], count)}."
    if subtemplate == "1_1_4_relative_neighbor_type":
        return f"It is a {object_label_text(st['object_type'])} at {format_location_blob(st['target_object_position'], st['target_object_image_refs'])}."
    if subtemplate == "1_2_1_size_bucket":
        return (
            f"The {object_label_text(st['object_type'])} {side_phrase(st['side'])} of the intersection "
            f"is classified as a {clean_text(st['size_bucket'])}-size vehicle."
        )
    if subtemplate == "1_2_2_visibility":
        visibility = st.get("visibility")
        return "None" if visibility in {None, "", "None"} else clean_text(visibility)
    if subtemplate == "1_3_1_weather":
        base = f"The scene appears to be {clean_text(st['weather'])}, {clean_text(st['time_of_day'])}."
        light = st.get("light") or {}
        if not light.get("sun_glare"):
            return base
        sides = [clean_text(side) for side in light.get("sun_glare_sides") or [] if clean_text(side)]
        if sides:
            if len(sides) == 1:
                views = sides[0]
            else:
                views = ", ".join(sides[:-1]) + f" and {sides[-1]}"
            return f"The scene appears to be {clean_text(st['weather'])}, {clean_text(st['time_of_day'])}, with sun glare in the {views} views."
        return f"The scene appears to be {clean_text(st['weather'])}, {clean_text(st['time_of_day'])}, with sun glare."
    if subtemplate == "1_3_2_vehicle_signal_state":
        return f"The signal is currently {clean_text(st['signal_state'])}."
    if subtemplate == "2_1_1_stopline_distance":
        return f"The {object_label_text(st['object_type'])} is {format_decimal(st['distance_m'])} m from the stop line on the {clean_text(st['stopline_side'])} approach."
    if subtemplate == "2_1_2_ped_to_far_edge":
        return f"The pedestrian on the {clean_text(st['crosswalk'])} crosswalk is {format_decimal(st['distance_m'])} m from the exit area."
    if subtemplate == "2_1_3_participant_distance":
        return answer
    if subtemplate == "2_1_4_nearest_vehicle_to_ped":
        return f"It is a {object_label_text(st['vehicle_type'])} on the {clean_text(st['side'])} approach, {format_decimal(st['distance_m'])} m away."
    if subtemplate == "2_2_1_lane_function":
        return f"The {object_label_text(st['object_type'])} {side_phrase(st['side'])} is currently in the {clean_text(st['lane_function'])}."
    if subtemplate == "2_2_2_ped_zone":
        zone = clean_text(st["ped_zone"])
        crosswalk = clean_text(st["crosswalk"])
        if zone == "within the crosswalk":
            return f"The pedestrian is currently within the {crosswalk} crosswalk."
        if zone == "waiting zone":
            return f"The pedestrian is currently in the waiting zone near the {crosswalk} crosswalk."
        return f"The pedestrian is currently in the {crosswalk} crosswalk {zone}."
    if subtemplate == "2_2_3_left_turn_queue_count":
        count = int(st["count"])
        return f"There are {count} queued vehicles in the {clean_text(st['lane_function'])} on the {clean_text(st['side'])} approach."
    if subtemplate == "2_2_4_stopline_back_5m_count":
        count = int(st["count"])
        return f"There are {count} vehicles within 5 m behind the {clean_text(st['side'])} stop line."
    if subtemplate == "2_2_5_longest_queue_lane":
        return f"The {clean_text(st['lane_function'])} on the {clean_text(st['side'])} approach currently has the longest queue."
    if subtemplate == "2_2_6_crosswalk_blocking":
        if bool(st.get("crosswalk_blocked")):
            return f"Yes, a {object_label_text(st['object_type'])} is currently blocking the {clean_text(st['side'])} crosswalk."
        return f"No, no vehicle is currently blocking the {clean_text(st['side'])} crosswalk."
    if subtemplate == "3_1_1_current_motion_state":
        side = st.get("side") or side_from_image_refs(st.get("object_image_refs"))
        location = side_phrase(side)
        base = (
            f"The {object_label_text(st['object_type'])} {location} is {clean_text(st['motion_state'])} "
            f"at {format_decimal(st['speed'])} m/s"
        )
        accel = st.get("acceleration")
        accel_state = st.get("accel_state")
        if accel is not None:
            if accel_state is None and float(accel) != 0:
                accel_state = "accelerating" if float(accel) > 0 else "decelerating"
            if accel_state:
                base += f" and {clean_text(accel_state)} at {format_decimal(accel)} m/s^2"
        return base + "."
    if subtemplate == "3_1_2_vehicle_maneuver":
        side = st.get("side") or side_from_image_refs(st.get("object_image_refs"))
        maneuver = MANEUVER_TEXT_BY_LABEL[clean_text(st["maneuver"])]
        return f"The {object_label_text(st['object_type'])} {side_phrase(side)} is {maneuver}."
    if subtemplate == "3_2_2_future_region":
        return answer
    if subtemplate == "3_2_3_waypoints":
        points = st.get("trajectory", {}).get("waypoints_xy") or []
        coords = ",".join(f"({format_decimal(item['dx'], 2)},{format_decimal(item['dy'], 2)})" for item in points)
        return f"Future trajectory:{coords}"
    if subtemplate == "3_3_1_safe_following":
        prefix = "Yes" if bool(st.get("is_safe")) else "No"
        safety = "safe" if bool(st.get("is_safe")) else "unsafe"
        return f"{prefix}, the following distance is currently {safety}, about {format_decimal(st['distance_m'])} m."
    if subtemplate == "3_3_2_likely_long_queue_lane":
        evidence = ((st.get("queue_evidence") or {}).get("queue_evidence") or {})
        parts = [
            f"{int(evidence.get('stopped_vehicles', 0))} stopped vehicles",
            f"{int(evidence.get('slow_vehicles', 0))} creeping vehicles",
            f"{int(evidence.get('moving_vehicles', 0))} moving vehicles",
        ]
        return (
            f"The {clean_text(st['lane_function'])} on the {clean_text(st['side'])} approach "
            f"is most likely to form a long queue because it already contains {', '.join(parts)}."
        )
    if subtemplate == "3_4_1_vehicle_ped_conflict":
        return "Yes." if bool(st.get("has_conflict")) else "No."
    if subtemplate == "3_4_2_nearest_conflict_participant":
        return (
            f"The most probable participant is the {object_label_text(st['conflict_partner_type'])} "
            f"at {format_location_blob(st['conflict_partner_position'], st['conflict_partner_image_refs'])}."
        )
    if subtemplate == "3_4_3_primary_risk_subject":
        return (
            f"The primary risk subject is the {object_label_text(st['subject_type'])} "
            f"at {format_location_blob(st['subject_position'], st['subject_image_refs'])} "
            f"because of {format_reason_text(st['risk_reason'])}."
        )
    if subtemplate == "3_4_4_risk_pattern":
        side = clean_text(st.get("side") or "center")
        suffix = "in the center area" if side == "center" else f"on the {side} approach"
        return f"The dominant conflict pattern is {clean_text(st['interaction_pattern'])} {suffix}."
    if subtemplate == "4_1_1_overall_state":
        return (
            f"The intersection is currently {clean_text(st['overall_state'])}, "
            f"with {int(st['moving_vehicles'])} of {int(st['total_vehicles'])} vehicles moving."
        )
    if subtemplate == "4_1_2_side_motion_status":
        counts = st.get("counts") or {}
        if "moving_vehicles" in counts and "stopped_vehicles" in counts:
            return (
                f"The {clean_text(st['side'])} approach is {clean_text(st['motion_label'])}, "
                f"with {int(counts['moving_vehicles'])} vehicles moving and {int(counts['stopped_vehicles'])} stopped."
            )
        return f"The {clean_text(st['side'])} approach is {clean_text(st['motion_label'])}."
    if subtemplate == "4_1_3_scene_summary":
        abnormal = clean_text(st.get("abnormal_or_none") or st.get("notable_abnormal") or "")
        if abnormal in {"none", ""}:
            return "No clear abnormal behavior is currently dominant."
        focus = clean_text(st.get("primary_focus") or "center")
        suffix = "in the center area" if focus == "center" else f"on the {focus} approach"
        return f"The most notable current risk is {format_abnormal_text(abnormal)} {suffix}."
    if subtemplate == "4_1_4_flow_imbalance":
        count = int(st["participant_count"])
        return f"The {clean_text(st['dominant_side'])} approach is the busiest, with {count} traffic participants."
    if subtemplate == "4_2_1_speeding_risk":
        if not bool(st.get("has_speeding_risk")):
            return "No."
        numeric = st.get("numeric_targets") or {}
        speed = numeric.get("speed_mps")
        side = st.get("side") or "center"
        location = side_phrase(side)
        return f"Yes, a {object_label_text(st['vehicle_type'])} {location} is still moving at about {format_decimal(speed)} m/s."
    if subtemplate == "4_2_2_notable_abnormal":
        region = clean_text(st.get("risk_region") or "center")
        suffix = "in the center area" if region == "center" else f"on the {region} approach"
        return f"A {format_abnormal_text(st['notable_abnormal'])} exists {suffix}."
    if subtemplate == "4_3_1_intersection_action":
        return INTERSECTION_ACTION_TEXT_BY_STATE[clean_text(st["action_state"])]
    if subtemplate == "4_3_2_side_action":
        return SIDE_ACTION_TEXT_BY_STATE[clean_text(st["action_state"])]
    if subtemplate == "4_3_3_lane_action":
        return LANE_ACTION_TEXT_BY_STATE[clean_text(st["action_state"])]
    if subtemplate == "4_3_4_object_action":
        side = st.get("side") or side_from_image_refs(st.get("object_image_refs"))
        return (
            f"The {object_label_text(st['object_type'])} {side_phrase(side)} "
            f"{OBJECT_ACTION_SUFFIX_BY_STATE[clean_text(st['action_state'])]}"
        )

    return answer


def render_compact_supervision_answer(
    subtemplate: str,
    structured_targets: Mapping[str, Any],
    default_answer: str,
    dataset_version: str = "v5",
) -> str | None:
    version = normalize_dataset_version(dataset_version)
    st = structured_targets or {}
    if version == "v6":
        if subtemplate == "3_1_1_current_motion_state":
            parts = [
                object_label_text(st["object_type"]),
                clean_text(st["motion_state"]),
                format_decimal(st["speed"]),
            ]
            if st.get("acceleration") is not None:
                accel_state = clean_text(st.get("accel_state") or ("accelerating" if float(st["acceleration"]) > 0 else "decelerating"))
                parts.extend([accel_state, format_decimal(st["acceleration"])])
            return "|".join(parts)
        if subtemplate == "4_2_1_speeding_risk":
            if not bool(st.get("has_speeding_risk")):
                return "NO"
            numeric = st.get("numeric_targets") or {}
            speed = numeric.get("speed_mps")
            ensure(speed is not None, "Expected numeric_targets.speed_mps for v6 speeding risk supervision.")
            side = normalize_side_token(st.get("side") or st.get("risk_region") or "center")
            return "|".join(["YES", object_label_text(st["vehicle_type"]), side, format_decimal(speed)])
        return None

    if version != "v5":
        return None
    if subtemplate == "2_1_1_stopline_distance":
        return "|".join([object_label_text(st["object_type"]), format_decimal(st["distance_m"]), normalize_side_token(st["stopline_side"])])
    if subtemplate == "2_1_2_ped_to_far_edge":
        return "|".join([normalize_side_token(st["crosswalk"]), format_decimal(st["distance_m"])])
    if subtemplate == "2_1_3_participant_distance":
        norm = normalize_parse_text(default_answer)
        match = re.match(
            r"the\s+([a-z0-9_ ]+?)\s+(in the center area|on the north approach|on the south approach|on the east approach|on the west approach)\s+is\s+(-?\d+(?:\.\d+)?)\s+m\s+from\s+the\s+([a-z0-9_ ]+?)\s+(in the center area|on the north approach|on the south approach|on the east approach|on the west approach)\.",
            norm,
        )
        if match:
            return "|".join(
                [
                    object_label_text(match.group(1)),
                    parse_side_from_phrase(match.group(2)),
                    format_decimal(float(match.group(3))),
                    object_label_text(match.group(4)),
                    parse_side_from_phrase(match.group(5)),
                ]
            )
        return None
    if subtemplate == "2_1_4_nearest_vehicle_to_ped":
        return "|".join([object_label_text(st["vehicle_type"]), normalize_side_token(st["side"]), format_decimal(st["distance_m"])])
    if subtemplate == "3_1_1_current_motion_state":
        parts = [
            object_label_text(st["object_type"]),
            normalize_side_token(st.get("side") or side_from_image_refs(st.get("object_image_refs"))),
            clean_text(st["motion_state"]),
            format_decimal(st["speed"]),
        ]
        if st.get("acceleration") is not None:
            accel_state = clean_text(st.get("accel_state") or ("accelerating" if float(st["acceleration"]) > 0 else "decelerating"))
            parts.extend([accel_state, format_decimal(st["acceleration"])])
        return "|".join(parts)
    if subtemplate == "1_1_4_relative_neighbor_type":
        return compact_location_payload(st["object_type"], st["target_object_position"], st["target_object_image_refs"])
    if subtemplate == "3_4_2_nearest_conflict_participant":
        return compact_location_payload(st["conflict_partner_type"], st["conflict_partner_position"], st["conflict_partner_image_refs"])
    if subtemplate == "3_4_3_primary_risk_subject":
        payload = compact_location_payload(st["subject_type"], st["subject_position"], st["subject_image_refs"])
        object_type, x, y, image_name, x1, y1 = payload.split("|")
        return "|".join([object_type, format_reason_text(st["risk_reason"]), x, y, image_name, x1, y1])
    return None


def supervision_fields(
    subtemplate: str,
    structured_targets: Mapping[str, Any],
    default_answer: str,
    dataset_version: str = "v5",
) -> dict[str, str]:
    version = normalize_dataset_version(dataset_version)
    prefix = prefix_for_subtemplate(subtemplate, version)
    if version == "v5" and subtemplate == "3_1_1_current_motion_state" and (structured_targets or {}).get("acceleration") is not None:
        prefix = MOTION_STATE_ACCEL_PREFIX
    if version == "v5" and subtemplate == "3_4_3_primary_risk_subject":
        prefix = RISK_SUBJECT_REASON_PREFIX
    if version == "v6" and subtemplate == "3_1_1_current_motion_state" and (structured_targets or {}).get("acceleration") is not None:
        prefix = MOTION_STATE_ACCEL_V6_PREFIX
    compact = render_compact_supervision_answer(subtemplate, structured_targets, default_answer, dataset_version=version)
    if compact is not None:
        answer = prefix + compact + COMPACT_TERMINATOR
    elif version == "v6":
        answer = render_supervision_answer_v6(subtemplate, structured_targets, default_answer)
    else:
        answer = render_supervision_answer(subtemplate, structured_targets, default_answer)
    ensure(answer, f"Supervision answer is empty for subtemplate={subtemplate}")
    return {"decoder_prefix": prefix, "supervision_answer": clean_text(answer)}


def track_type_candidates(structured_targets: Mapping[str, Any], dataset_version: str = "v5") -> list[tuple[str, str]]:
    del dataset_version
    st = structured_targets or {}
    pairs: list[tuple[str, str]] = []
    pair_type = maybe_text(st.get("pair_type"))
    pair_parts = [object_label_text(part) for part in pair_type.split("-")] if pair_type else []

    def add(track_key: str, type_value: Any) -> None:
        tracking_id = maybe_text(st.get(track_key))
        if not tracking_id or type_value in {None, ""}:
            return
        pairs.append((tracking_id, object_label_text(type_value)))

    add("raw_tracking_id", st.get("object_type"))
    add("obj1_raw_tracking_id", st.get("obj1_type"))
    add("obj2_raw_tracking_id", st.get("obj2_type"))
    add("vehicle_raw_tracking_id", st.get("vehicle_type"))
    add("subject_raw_tracking_id", st.get("subject_type"))
    add("conflict_partner_raw_tracking_id", st.get("conflict_partner_type"))
    add("target_object_raw_tracking_id", st.get("target_object_type") or st.get("object_type"))
    add("focus_raw_tracking_id", st.get("focus_type"))
    add("blocking_vehicle_raw_tracking_id", st.get("object_type"))
    pedestrian_tracking_id = maybe_text(st.get("pedestrian_raw_tracking_id"))
    if pedestrian_tracking_id:
        pairs.append((pedestrian_tracking_id, "pedestrian"))
    vru_tracking_id = maybe_text(st.get("vru_raw_tracking_id"))
    if vru_tracking_id and len(pair_parts) == 2:
        pairs.append((vru_tracking_id, pair_parts[1]))
    leader_tracking_id = maybe_text(st.get("leader_raw_tracking_id"))
    if leader_tracking_id and len(pair_parts) == 2:
        pairs.append((leader_tracking_id, pair_parts[0]))
    follower_tracking_id = maybe_text(st.get("follower_raw_tracking_id"))
    if follower_tracking_id and len(pair_parts) == 2:
        pairs.append((follower_tracking_id, pair_parts[1]))
    return pairs


def build_answer_type_lookup(records: Iterable[Mapping[str, Any]], dataset_version: str = "v5") -> dict[tuple[str, str], dict[str, str]]:
    per_frame: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in records:
        key = (clean_text(row["scene_id"]), clean_text(row["frame_token"]))
        for tracking_id, label in track_type_candidates(row.get("structured_targets") or {}, dataset_version=dataset_version):
            per_frame[key][tracking_id][label] += 1
    result: dict[tuple[str, str], dict[str, str]] = {}
    for frame_key, mapping in per_frame.items():
        result[frame_key] = {track_id: counter.most_common(1)[0][0] for track_id, counter in mapping.items() if counter}
    return result


def build_answer_label_vocab(records: Iterable[Mapping[str, Any]], dataset_version: str = "v5") -> list[str]:
    labels = Counter()
    for row in records:
        for _, label in track_type_candidates(row.get("structured_targets") or {}, dataset_version=dataset_version):
            labels[label] += 1
    ordered = [label for label, _ in labels.most_common()]
    if "unknown object" not in ordered:
        ordered.append("unknown object")
    return ordered


def normalize_prediction(
    subtemplate: str,
    prediction: str,
    decoder_prefix: str | None = None,
    dataset_version: str = "v5",
) -> tuple[str, list[str]]:
    version = normalize_dataset_version(dataset_version)
    if version == "v6":
        return normalize_prediction_v6(subtemplate, prediction, decoder_prefix=decoder_prefix)
    raw = clean_text(prediction)
    reasons: list[str] = []
    normalized = dedupe_repeated_tail(raw)
    if normalized != raw:
        reasons.append("repeated_tail")

    if subtemplate in COMPACT_SUPERVISION_TEMPLATES_BY_VERSION["v5"]:
        compact_render = render_compact_prediction(subtemplate, normalized, prefix_override=decoder_prefix)
        if compact_render is not None:
            if compact_render != normalized:
                reasons.append("compact_render")
            return compact_render, reasons

    if subtemplate in YES_NO_TEMPLATES_BY_VERSION["v5"]:
        yn = extract_yes_no(normalized)
        if yn is not None:
            if yn != normalized:
                reasons.append("yes_no_normalized")
            return yn, reasons

    if subtemplate in STRICT_LOCATION_TEMPLATES_BY_VERSION["v5"]:
        blob = extract_first_location_blob(normalized)
        lowered = normalized.lower()
        if subtemplate == "1_1_4_relative_neighbor_type":
            match = re.search(r"it is a\s+([a-z0-9_ ]+?)\s+at", lowered)
            if match and blob:
                return f"It is a {object_label_text(match.group(1))} at {blob}.", reasons
        if subtemplate == "3_4_2_nearest_conflict_participant":
            match = re.search(r"the most probable participant is the\s+([a-z0-9_ ]+?)\s+at", lowered)
            if match and blob:
                return f"The most probable participant is the {object_label_text(match.group(1))} at {blob}.", reasons
        if subtemplate == "3_4_3_primary_risk_subject":
            match = re.search(r"the primary risk subject is the\s+([a-z0-9_ ]+?)\s+at", lowered)
            reason_match = re.search(r"because of\s+([a-z_ ]+)", lowered)
            if match and blob and reason_match:
                return (
                    f"The primary risk subject is the {object_label_text(match.group(1))} "
                    f"at {blob} because of {clean_text(reason_match.group(1)).replace('_', ' ')}."
                ), reasons

    if subtemplate == "3_2_3_waypoints":
        coords = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", normalized)
        if coords:
            body = ",".join(f"({format_decimal(x, 2)},{format_decimal(y, 2)})" for x, y in coords)
            return f"Future trajectory:{body}", reasons

    if subtemplate in ONE_SENTENCE_TEMPLATES_BY_VERSION["v5"]:
        sentence = first_sentence(normalized)
        if sentence != normalized:
            reasons.append("first_sentence")
        normalized = sentence

    return normalized, reasons


def diagnose_prediction(
    subtemplate: str,
    raw_prediction: str,
    normalized_prediction: str,
    dataset_version: str = "v5",
) -> list[str]:
    version = normalize_dataset_version(dataset_version)
    if version == "v6":
        return diagnose_prediction_v6(subtemplate, raw_prediction, normalized_prediction)
    reasons: list[str] = []
    raw = clean_text(raw_prediction).lower()
    normalized = clean_text(normalized_prediction).lower()
    if subtemplate != "3_3_2_likely_long_queue_lane" and "most likely to form a long queue" in raw:
        reasons.append("wrong_template_collapse")
    if subtemplate in STRICT_LOCATION_TEMPLATES_BY_VERSION["v5"] and LOCATION_BLOB_REGEX.search(normalized_prediction) is None:
        reasons.append("malformed_location_blob")
    if subtemplate == "3_1_1_current_motion_state" and "m/s" not in normalized:
        reasons.append("missing_numeric_field")
    if subtemplate in ACTION_TEMPLATES_BY_VERSION["v5"] and "traffic should" not in normalized and "should " not in normalized:
        reasons.append("missing_action_phrase")
    if subtemplate == "3_2_3_waypoints" and "(" not in normalized_prediction:
        reasons.append("missing_waypoints")
    return reasons
