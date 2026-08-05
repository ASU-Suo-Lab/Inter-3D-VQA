#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

TEXT_METRIC_WEIGHTS = {
    "bertscore": 0.35,
    "simcse": 0.25,
    "rouge_l": 0.20,
    "normalized_exact_match": 0.15,
    "bleu_4": 0.05,
}

NUMERIC_ERROR_BUCKETS = (
    "count",
    "distance",
    "speed",
    "acceleration",
    "global_3d_xy",
    "image_2d_xy",
    "waypoint_xy",
)

INTEGER_NUMERIC_BUCKETS = {"count"}
EUCLIDEAN_NUMERIC_BUCKETS = {"global_3d_xy", "image_2d_xy", "waypoint_xy"}

NUMERIC_TOTAL_ERROR_THRESHOLDS = {
    "count": 2.0,
    "distance": 10.0,
    "speed": 5.0,
    "acceleration": 1.0,
    "global_3d_xy": 8.0,
    "image_2d_xy": 150.0,
    "waypoint_xy": 5.0,
}

INTERSECTION_ACTION_BY_TEXT = {
    "traffic should continue in a coordinated and orderly way.": "FLOW_STABLE",
    "traffic should move more calmly and with reduced speed.": "FLOW_CALMING",
    "traffic should proceed in a more orderly and tightly managed way.": "QUEUE_MANAGEMENT",
    "traffic should reduce aggressive movement and prioritize conflict avoidance.": "CONFLICT_SUPPRESSION",
}

INTERSECTION_ACTION_PHRASES = {
    "FLOW_STABLE": (
        "coordinated and orderly way",
        "coordinated and orderly manner",
    ),
    "FLOW_CALMING": (
        "more calmly and with reduced speed",
        "reduced speed",
    ),
    "QUEUE_MANAGEMENT": (
        "more orderly and tightly managed way",
        "more orderly and tightly managed manner",
        "tightly managed",
    ),
    "CONFLICT_SUPPRESSION": (
        "reduce aggressive movement and prioritize conflict avoidance",
        "prioritize conflict avoidance",
    ),
}

SIDE_ACTION_BY_TEXT = {
    "traffic should keep safer local spacing.": "SIDE_CLEARANCE_PROTECTION",
    "traffic should moderate speed and maintain safer spacing.": "SIDE_SPEED_MODERATION",
    "traffic should proceed cautiously and remain orderly.": "SIDE_GENERAL_CAUTION",
    "traffic should stabilize queue movement and avoid unnecessary disruption.": "SIDE_QUEUE_STABILIZATION",
    "traffic should yield clearly to crossing activity.": "SIDE_CROSSING_AWARENESS",
}

SIDE_ACTION_PHRASES = {
    "SIDE_CLEARANCE_PROTECTION": (
        "keep safer local spacing",
        "keep safer distance",
    ),
    "SIDE_SPEED_MODERATION": (
        "moderate speed and maintain safer spacing",
    ),
    "SIDE_GENERAL_CAUTION": (
        "proceed cautiously and remain orderly",
        "proceed cautiously",
    ),
    "SIDE_QUEUE_STABILIZATION": (
        "stabilize queue movement and avoid unnecessary disruption",
    ),
    "SIDE_CROSSING_AWARENESS": (
        "yield clearly to crossing activity",
        "crossing activity",
    ),
}

LANE_ACTION_BY_TEXT = {
    "traffic should maintain clearance and avoid tight local conflicts.": "LANE_CLEARANCE_MAINTENANCE",
    "traffic should prepare to stop and avoid pressing forward.": "LANE_PREPARE_TO_STOP",
    "traffic should preserve queue order and avoid disruption.": "LANE_QUEUE_PRESERVATION",
    "traffic should proceed in an orderly manner.": "LANE_GENERAL_ORDER",
    "traffic should reduce speed and proceed more conservatively.": "LANE_SPEED_REDUCTION",
}

LANE_ACTION_PHRASES = {
    "LANE_CLEARANCE_MAINTENANCE": (
        "maintain clearance and avoid tight local conflicts",
        "maintain clearance and avoid tight conflicts",
    ),
    "LANE_PREPARE_TO_STOP": (
        "prepare to stop and avoid pressing forward",
    ),
    "LANE_QUEUE_PRESERVATION": (
        "preserve queue order and avoid disruption",
    ),
    "LANE_GENERAL_ORDER": (
        "proceed in an orderly manner",
    ),
    "LANE_SPEED_REDUCTION": (
        "reduce speed and proceed more conservatively",
    ),
}

OBJECT_ACTION_SUFFIX_TO_STATE = {
    "should yield now.": "OBJECT_YIELD_NOW",
    "should slow down and keep safer local spacing.": "OBJECT_SLOW_DOWN",
    "should prepare to stop.": "OBJECT_PREPARE_TO_STOP",
    "should proceed cautiously.": "OBJECT_PROCEED_CAUTIOUSLY",
}

MANEUVER_TEXT_TO_LABEL = {
    "going straight": "straight",
    "making a left turn": "left turn",
    "making a right turn": "right turn",
    "changing lanes": "lane change",
    "stopping and waiting": "stop-and-wait",
}

FUTURE_REGION_TEXT_TO_LABEL = {
    "remain before the stop line": "before stop line",
    "move into the intersection center": "intersection center",
    "move toward the left-turn exit": "left-turn exit",
    "move toward the through exit": "through exit",
    "move toward the right-turn exit": "right-turn exit",
}

ABNORMAL_TEXT_TO_LABEL = {
    "high-risk abnormal proximity interaction": "abnormal_proximity",
    "high-risk speeding event": "speeding",
    "crosswalk-blocking event": "crosswalk_blocking",
    "lingering-pedestrian event": "lingering_pedestrian",
    "stop-line overrun event": "stopline_overrun",
    "wrong-way two-wheeler event": "wrong_way_two_wheeler",
}

RISK_REASON_TEXT_TO_LABEL = {
    "path crossing": "path_crossing",
    "overspeed": "overspeed",
    "vru conflict": "vru_conflict",
    "lane change conflict": "lane_change_conflict",
    "proximity": "proximity",
}


@dataclass
class SampleRecord:
    question_id: str
    scene_id: str
    chapter: str
    section: str
    subtemplate: str
    answer: str
    prediction: str
    structured_targets: Dict[str, Any]


@dataclass
class TemplateEvalSpec:
    parser: Callable[[str], Dict[str, Any]]
    discrete_fields: Callable[[Dict[str, Any]], List[str]] = field(default_factory=lambda: (lambda _gt: []))
    numeric_fields: Callable[[Dict[str, Any]], Dict[str, List[str]]] = field(default_factory=lambda: (lambda _gt: {}))


def numeric_bucket_map(**bucket_to_patterns: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for bucket in NUMERIC_ERROR_BUCKETS:
        patterns = bucket_to_patterns.get(bucket) or []
        if patterns:
            out[bucket] = patterns
    return out


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V5 Intersection VQA predictions against sidecar structured targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sidecar-jsonl",
        type=Path,
        default=Path("LlamaFactory/data/intersection_vqa/intersection_vqa_eval_sidecar.jsonl"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("LlamaFactory/eval/intersection_vqa"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for debugging.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    require(path.is_file(), f"File not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    require(path.is_file(), f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"\b(\d+)\.0\b", r"\1", text)
    return text


def normalize_parse_text(text: str) -> str:
    return normalize_text(text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def text_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+(?:\.[0-9]+)?", normalize_text(text))


def semantic_value_tokens(field_name: str, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return ["yes"] if value else ["no"]
    if isinstance(value, (int, float)):
        text = str(int(value)) if float(value).is_integer() else f"{float(value):.6f}".rstrip("0").rstrip(".")
        return [text]
    text = str(value).strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    if field_name.endswith("image_name"):
        text = re.sub(r"\s+", "", text)
        return [text] if text else []
    text = text.replace("_", " ").replace("-", " ")
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)


def parse_int_token(token: str) -> Optional[int]:
    token = token.strip().lower()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return COUNT_WORDS.get(token)


def parse_yes_no(text: str) -> Optional[bool]:
    norm = normalize_parse_text(text)
    if norm.startswith("yes"):
        return True
    if norm.startswith("no"):
        return False
    return None


def parse_first_float(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return None if match is None else float(match.group(0))


def format_float(value: float) -> float:
    return round(float(value), 6)


def singularize(word: str) -> str:
    if word.endswith("us"):
        return word
    if word == "buses":
        return "bus"
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ses") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_object_type(text: str) -> str:
    text = normalize_whitespace(text.lower())
    parts = text.split(" ")
    if not parts:
        return text
    parts[-1] = singularize(parts[-1])
    return " ".join(parts)


APPROACH_PATTERN = r"(?:north|south|east|west)"
APPROACH_WORD_PATTERN = r"(?:approach|side)"
APPROACH_ORDER = ("east", "north", "west", "south")
APPROACH_LOCATION_PATTERN = (
    rf"(?:in the center area(?: of the intersection)?|"
    rf"in the center of the intersection|"
    rf"on the {APPROACH_PATTERN} {APPROACH_WORD_PATTERN}(?: of the intersection)?)"
)
APPROACH_LOCATION_WITH_INTERSECTION_PATTERN = APPROACH_LOCATION_PATTERN


def extract_side_phrase(text: str) -> Optional[str]:
    norm = normalize_parse_text(text)
    if "center area" in norm or "center of the intersection" in norm:
        return "center"
    match = re.search(r"\b(?:on|of)\s+the\s+(north|south|east|west)\s+(?:side|approach)\b", norm)
    if match:
        return match.group(1)
    return None


def extract_crosswalk(text: str) -> Optional[str]:
    match = re.search(r"\b(north|south|east|west)\s+crosswalk\b", normalize_parse_text(text))
    return match.group(1) if match else None


def parse_location_reference(reference_text: str) -> Dict[str, Any]:
    blob = reference_text.strip()
    if blob.startswith("<") and blob.endswith(">"):
        blob = blob[1:-1]
    parts = [part.strip() for part in blob.split(",") if part.strip()]
    if len(parts) < 2:
        return {}
    try:
        x = format_float(float(parts[0]))
        y = format_float(float(parts[1]))
    except ValueError:
        return {}
    image_refs = []
    remaining = parts[2:]
    for idx in range(0, len(remaining), 3):
        chunk = remaining[idx : idx + 3]
        if len(chunk) != 3:
            break
        image_name = chunk[0]
        try:
            x1 = format_float(float(chunk[1]))
            y1 = format_float(float(chunk[2]))
        except ValueError:
            continue
        image_refs.append({"image_name": image_name, "x1": x1, "y1": y1})
    return {"position": {"x": x, "y": y}, "image_refs": image_refs}


def parse_object_type_and_side(text: str) -> Tuple[Optional[str], Optional[str]]:
    norm = normalize_parse_text(text)
    match = re.search(
        rf"\b(?:it is a|it is an|the|a|an)\s+(?P<object_type>[a-z0-9_ ]+?)\s+"
        rf"(?P<location>(?:located\s+)?{APPROACH_LOCATION_WITH_INTERSECTION_PATTERN})",
        norm,
    )
    if not match:
        return None, None
    object_type = normalize_object_type(match.group("object_type"))
    location = re.sub(r"^located\s+", "", match.group("location"))
    side = extract_side_phrase(location)
    return object_type, side


def normalize_signal_state_label(
    raw_label: str,
    *,
    movement: Optional[str] = None,
    side: Optional[str] = None,
) -> Optional[str]:
    label = normalize_parse_text(raw_label).strip().strip(".")
    label = label.removesuffix(" signal").strip()
    if label in {"red", "yellow", "green"}:
        if movement == "left-turn" and side in {"east", "west"}:
            return f"{label} arrow"
        return f"{label} light"
    if label in {"red light", "yellow light", "green light", "red arrow", "yellow arrow", "green arrow"}:
        return label
    return None


def parse_action_state_exact(text: str, mapping: Dict[str, str]) -> Dict[str, Any]:
    action_state = mapping.get(normalize_parse_text(text))
    return {"action_state": action_state}


def extract_region_focus(text: str) -> Optional[str]:
    norm = normalize_parse_text(text)
    if any(
        phrase in norm
        for phrase in (
            "around the intersection",
            "in the center",
            "in the center area",
            "in the center of the intersection",
            "center area of the intersection",
        )
    ):
        return "center"
    return extract_side_phrase(norm)


def parse_action_state_by_phrases(
    text: str,
    mapping: Dict[str, str],
    phrase_mapping: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    action_state = mapping.get(norm)
    if action_state is not None:
        return {"action_state": action_state}
    best_state: Optional[str] = None
    best_score = 0
    tied = False
    for state, phrases in phrase_mapping.items():
        score = sum(1 for phrase in phrases if phrase in norm)
        if score > best_score:
            best_state = state
            best_score = score
            tied = False
        elif score > 0 and score == best_score:
            tied = True
    if best_score == 0 or tied:
        return {"action_state": None}
    return {"action_state": best_state}


def parse_1_1_1(text: str) -> Dict[str, Any]:
    object_type, side = parse_object_type_and_side(text)
    return {"object_type": object_type, "side": side}


def parse_1_1_2(text: str) -> Dict[str, Any]:
    exists = parse_yes_no(text)
    out = {"exists": exists}
    if exists:
        match = re.search(r"\b(?:there is|there are)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+vru", normalize_parse_text(text))
        out["count"] = parse_int_token(match.group(1)) if match else None
    return out


def parse_1_1_3(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(
        r"the\s+(north|south|east|west)\s+approach\s+(?:currently\s+)?has\s+"
        r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z0-9_ ]+?)\.?$",
        norm,
    )
    if not match:
        match = re.match(
            r"there\s+(?:is|are)\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"([a-z0-9_ ]+?)\s+on\s+the\s+(north|south|east|west)\s+(?:approach|side)\.?$",
            norm,
        )
        if not match:
            return {}
        return {
            "side": match.group(3),
            "count": parse_int_token(match.group(1)),
            "object_type": normalize_object_type(match.group(2)),
        }
    return {
        "side": match.group(1),
        "count": parse_int_token(match.group(2)),
        "object_type": normalize_object_type(match.group(3)),
    }


def parse_1_1_4(text: str) -> Dict[str, Any]:
    match = re.match(r"it is a\s+([a-z0-9_ ]+?)\s+at\s+(<.+>)\.", normalize_parse_text(text))
    if not match:
        return {}
    ref = parse_location_reference(match.group(2))
    return {
        "object_type": normalize_object_type(match.group(1)),
        "target_object_position": ref.get("position"),
        "target_object_image_refs": ref.get("image_refs", []),
    }


def parse_1_2_1(text: str) -> Dict[str, Any]:
    object_type, side = parse_object_type_and_side(text)
    match = re.search(r"classified as a[n]?\s+([a-z-]+)-size vehicle", normalize_parse_text(text))
    size_bucket = match.group(1) if match else None
    return {"object_type": object_type, "side": side, "size_bucket": size_bucket}


def parse_1_2_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    value = None if norm in {"", "none", "n/a", "na"} else norm
    return {"visibility": value}


def parse_1_3_1(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    out: Dict[str, Any] = {}
    weather_match = re.search(r"\b(rainy|cloudy|sunny)\b", norm)
    if weather_match:
        out["weather"] = weather_match.group(1)
    time_match = re.search(r"\b(daytime|nighttime)\b", norm)
    if time_match:
        out["time_of_day"] = time_match.group(1)
    if "sun glare" in norm:
        sides: List[str] = []
        segment_match = re.search(r"sun glare(?:[^.]*?)in the\s+(.+?)\s+views?", norm)
        if segment_match:
            raw = segment_match.group(1)
            seen = set()
            for token in re.split(r",\s*|\s+and\s+", raw):
                token = token.strip()
                if token.endswith(" view"):
                    token = token[:-5]
                if token in APPROACH_ORDER and token not in seen:
                    sides.append(token)
                    seen.add(token)
            sides = [side for side in APPROACH_ORDER if side in seen]
        out["light"] = {"sun_glare": True, "sun_glare_sides": sides}
    return out


def parse_1_3_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    side = extract_side_phrase(norm)
    movement: Optional[str] = None
    shorthand = re.search(r"\b(north|south|east|west)-(left|through)\s+signal\b", norm)
    if shorthand:
        side = shorthand.group(1)
        movement = "left-turn" if shorthand.group(2) == "left" else "through"
    elif "left-turn signal" in norm:
        movement = "left-turn"
    elif "through signal" in norm:
        movement = "through"
    match = re.search(r"\bis\s+(?:currently\s+)?([a-z]+(?:\s+(?:light|arrow))?)\.?$", norm)
    if not match:
        return {}
    signal_state = normalize_signal_state_label(match.group(1), movement=movement, side=side)
    return {"signal_state": signal_state} if signal_state else {}


def parse_2_1_1(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+([a-z0-9_ ]+?)\s+is\s+(-?\d+(?:\.\d+)?)\s+(?:m|meter|meters)\s+"
        r"(?:away\s+)?from\s+(?:"
        r"the\s+stop line on the\s+(north|south|east|west)\s+(?:approach|side)|"
        r"the\s+(north|south|east|west)\s+stop line)\.?$",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    stopline_side = match.group(3) or match.group(4)
    return {
        "object_type": normalize_object_type(match.group(1)),
        "distance_m": format_float(float(match.group(2))),
        "stopline_side": stopline_side,
    }


def parse_2_1_2(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+pedestrian\s+on\s+the\s+(north|south|east|west)\s+crosswalk\s+is\s+(-?\d+(?:\.\d+)?)\s+m\s+from\s+the\s+exit area\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {"crosswalk": match.group(1), "distance_m": format_float(float(match.group(2)))}


def parse_2_1_3(text: str) -> Dict[str, Any]:
    match = re.match(
        rf"the\s+([a-z0-9_ ]+?)\s+{APPROACH_LOCATION_PATTERN}\s+is\s+(-?\d+(?:\.\d+)?)\s+m\s+from\s+the\s+([a-z0-9_ ]+?)\s+{APPROACH_LOCATION_PATTERN}\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "obj1_type": normalize_object_type(match.group(1)),
        "distance_m": format_float(float(match.group(2))),
        "obj2_type": normalize_object_type(match.group(3)),
    }


def parse_2_1_4(text: str) -> Dict[str, Any]:
    match = re.match(
        r"it is a\s+([a-z0-9_ ]+?)\s+on the\s+(north|south|east|west|center)\s+approach,\s+(-?\d+(?:\.\d+)?)\s+m away\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "vehicle_type": normalize_object_type(match.group(1)),
        "side": match.group(2),
        "distance_m": format_float(float(match.group(3))),
    }


def parse_2_2_1(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+([a-z0-9_ ]+?)\s+on the\s+(north|south|east|west)\s+(?:approach|side)\s+is currently in the\s+(left-turn lane|through lane|right-turn lane)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "object_type": normalize_object_type(match.group(1)),
        "side": match.group(2),
        "lane_function": match.group(3),
    }


def parse_2_2_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    within = re.match(r"the pedestrian is currently within the\s+(north|south|east|west)\s+crosswalk\.", norm)
    if within:
        return {"crosswalk": within.group(1), "ped_zone": "within the crosswalk"}
    entry = re.match(
        r"the pedestrian is currently in the\s+(north|south|east|west)\s+crosswalk\s+(entry zone|exit zone)\.",
        norm,
    )
    if entry:
        return {"crosswalk": entry.group(1), "ped_zone": entry.group(2)}
    waiting = re.match(
        r"the pedestrian is currently in the\s+waiting zone near the\s+(north|south|east|west)\s+crosswalk\.",
        norm,
    )
    if waiting:
        return {"crosswalk": waiting.group(1), "ped_zone": "waiting zone"}
    return {}


def parse_2_2_3(text: str) -> Dict[str, Any]:
    match = re.match(
        r"there are\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+queued vehicles?\s+in the\s+(left-turn lane|through lane|right-turn lane)\s+on the\s+(north|south|east|west)\s+approach\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {"count": parse_int_token(match.group(1)), "lane_function": match.group(2), "side": match.group(3)}


def parse_2_2_4(text: str) -> Dict[str, Any]:
    match = re.match(
        r"there are\s+(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+vehicles within 5 m behind the\s+"
        r"(?:(north|south|east|west)\s+stop line|stop line on the\s+(north|south|east|west)\s+approach)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    side = match.group(2) or match.group(3)
    return {"count": parse_int_token(match.group(1)), "side": side}


def parse_2_2_5(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+(left-turn lane|through lane|right-turn lane)\s+on the\s+(north|south|east|west)\s+approach\s+currently has the longest queue\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {"lane_function": match.group(1), "side": match.group(2)}


def parse_2_2_6(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    blocked = parse_yes_no(norm)
    out = {"crosswalk_blocked": blocked}
    if blocked is None:
        return {}
    if blocked:
        match = re.match(
            r"yes,\s+a\s+([a-z0-9_ ]+?)\s+is currently blocking the\s+(north|south|east|west)\s+crosswalk\.",
            norm,
        )
        if match:
            out["object_type"] = normalize_object_type(match.group(1))
            out["side"] = match.group(2)
    else:
        match = re.match(r"no,\s+no vehicle is currently blocking the\s+(north|south|east|west)\s+crosswalk\.", norm)
        if match:
            out["side"] = match.group(1)
    return out


def parse_3_1_1(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    base_match = re.match(
        rf"the\s+([a-z0-9_ ]+?)\s+({APPROACH_LOCATION_PATTERN})\s+is\s+([a-z-]+)\s+(.+?)\.?$",
        norm,
    )
    if base_match:
        tail = base_match.group(4)
        speed_match = re.search(
            r"(?:at|at\s+a\s+speed\s+of|moving\s+at|traveling\s+at|with\s+speed(?:\s+of)?)\s+"
            r"(-?\d+(?:\.\d+)?)\s+m/s\b",
            tail,
        )
        if speed_match:
            out = {
                "object_type": normalize_object_type(base_match.group(1)),
                "side": extract_side_phrase(base_match.group(2)),
                "motion_state": base_match.group(3),
                "speed": format_float(float(speed_match.group(1))),
            }
            accel_match = re.search(
                r"(?:and|,)?\s*(?:(accelerating|decelerating)\s+at|(?:with\s+)?(?:an\s+)?acceleration\s+of)\s+"
                r"(-?\d+(?:\.\d+)?)\s+m/s\^2",
                tail,
            )
            if accel_match:
                acceleration = format_float(float(accel_match.group(2)))
                accel_state = accel_match.group(1)
                if accel_state is None:
                    if out["motion_state"] == "starting":
                        accel_state = "accelerating"
                    elif out["motion_state"] == "braking":
                        accel_state = "decelerating"
                    else:
                        accel_state = "accelerating" if acceleration >= 0 else "decelerating"
                out["accel_state"] = accel_state
                out["acceleration"] = acceleration
            return out
    accel_only = re.match(
        rf"the\s+([a-z0-9_ ]+?)\s+({APPROACH_LOCATION_PATTERN})\s+is\s+([a-z-]+)\s+"
        rf"(?:at(?:\s+an\s+acceleration\s+of)?|with(?:\s+an)?\s+acceleration\s+of)\s+(-?\d+(?:\.\d+)?)\s+m/s\^2\.?$",
        norm,
    )
    if not accel_only:
        return {}
    acceleration = format_float(float(accel_only.group(4)))
    motion_state = accel_only.group(3)
    if motion_state == "starting":
        accel_state = "accelerating"
    elif motion_state == "braking":
        accel_state = "decelerating"
    else:
        accel_state = "accelerating" if acceleration >= 0 else "decelerating"
    return {
        "object_type": normalize_object_type(accel_only.group(1)),
        "side": extract_side_phrase(accel_only.group(2)),
        "motion_state": motion_state,
        "accel_state": accel_state,
        "acceleration": acceleration,
    }


def parse_3_1_2(text: str) -> Dict[str, Any]:
    match = re.match(
        rf"the\s+([a-z0-9_ ]+?)\s+({APPROACH_LOCATION_PATTERN})\s+is\s+(.+)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    maneuver = MANEUVER_TEXT_TO_LABEL.get(match.group(3).strip())
    return {
        "object_type": normalize_object_type(match.group(1)),
        "side": extract_side_phrase(match.group(2)),
        "maneuver": maneuver,
    }


def parse_3_1_3(text: str) -> Dict[str, Any]:
    return parse_3_1_2(text)


def parse_3_2_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(
        rf"the\s+([a-z0-9_ ]+?)\s+{APPROACH_LOCATION_PATTERN}\s+is likely to\s+(.+)\.",
        norm,
    )
    if not match:
        return {}
    return {
        "object_type": normalize_object_type(match.group(1)),
        "future_region": FUTURE_REGION_TEXT_TO_LABEL.get(match.group(2).strip()),
    }


def parse_3_2_3(text: str) -> Dict[str, Any]:
    coords = re.findall(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)", normalize_parse_text(text))
    if not coords:
        return {}
    return {
        "trajectory": {
            "waypoints_xy": [{"dx": format_float(float(x)), "dy": format_float(float(y))} for x, y in coords]
        }
    }


def parse_3_3_1(text: str) -> Dict[str, Any]:
    is_safe = parse_yes_no(text)
    distance = parse_first_float(text)
    if is_safe is None or distance is None:
        return {}
    return {"is_safe": is_safe, "distance_m": format_float(distance)}


def parse_3_3_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(
        r"the\s+(left-turn lane|through lane|right-turn lane)\s+on the\s+(north|south|east|west)\s+approach\s+is most likely to form a long queue because it\s+(?:already\s+)?contains\s+(.+)\.",
        norm,
    )
    if not match:
        return {}
    detail = match.group(3)
    stopped = 0
    slow = 0
    moving = 0
    for count_token, label in re.findall(
        r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(stopped|creeping|moving)\s+vehicles?",
        detail,
    ):
        value = parse_int_token(count_token) or 0
        if label == "stopped":
            stopped = value
        elif label == "creeping":
            slow = value
        elif label == "moving":
            moving = value
    return {
        "lane_function": match.group(1),
        "side": match.group(2),
        "queue_evidence": {
            "queue_evidence": {
                "stopped_vehicles": stopped,
                "slow_vehicles": slow,
                "moving_vehicles": moving,
            }
        },
    }


def parse_3_4_1(text: str) -> Dict[str, Any]:
    has_conflict = parse_yes_no(text)
    return {} if has_conflict is None else {"has_conflict": has_conflict}


def parse_3_4_2(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the most probable participant is the\s+([a-z0-9_ ]+?)\s+at\s+(<.+>)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    ref = parse_location_reference(match.group(2))
    return {
        "conflict_partner_type": normalize_object_type(match.group(1)),
        "conflict_partner_position": ref.get("position"),
        "conflict_partner_image_refs": ref.get("image_refs", []),
    }


def parse_3_4_3(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the primary risk subject is the\s+([a-z0-9_ ]+?)\s+at\s+(<.+>)\s+because of\s+([a-z_ ]+)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    ref = parse_location_reference(match.group(2))
    return {
        "subject_type": normalize_object_type(match.group(1)),
        "subject_position": ref.get("position"),
        "subject_image_refs": ref.get("image_refs", []),
        "risk_reason": RISK_REASON_TEXT_TO_LABEL.get(match.group(3).strip(), match.group(3).strip().replace(" ", "_")),
    }


def parse_3_4_4(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(
        r"the dominant conflict pattern is\s+([a-z0-9_-]+)\s+(?:on the\s+(north|south|east|west)\s+approach|in the center area)\.",
        norm,
    )
    if not match:
        return {}
    side = match.group(2) if match.group(2) else "center"
    return {"interaction_pattern": match.group(1), "side": side}


def parse_4_1_1(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the intersection is currently\s+([a-z- ]+),\s+with\s+(\d+)\s+of\s+(\d+)\s+vehicles moving\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {
        "overall_state": match.group(1).strip(),
        "moving_vehicles": int(match.group(2)),
        "total_vehicles": int(match.group(3)),
    }


def parse_4_1_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    base = re.match(
        r"the\s+(north|south|east|west)\s+approach is\s+([a-z ]+?)(?:,\s+with\s+(\d+)\s+vehicles moving and\s+(\d+)\s+stopped)?\.",
        norm,
    )
    if not base:
        return {}
    out = {"side": base.group(1), "motion_label": base.group(2).strip()}
    if base.group(3) is not None and base.group(4) is not None:
        out["counts"] = {"moving_vehicles": int(base.group(3)), "stopped_vehicles": int(base.group(4))}
    return out


def parse_4_1_3(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    if any(
        phrase in norm
        for phrase in (
            "no clear abnormal behavior",
            "no clear abnormal behavior is currently dominant",
            "no clear abnormal behavior is dominant",
            "no dominant abnormal behavior",
            "no notable abnormal behavior",
            "light traffic with no clear abnormal behavior",
        )
    ):
        return {"abnormal_or_none": "none"}
    abnormal = None
    for phrases, label in (
        (("abnormal proximity",), "abnormal proximity"),
        (("speeding",), "speeding"),
    ):
        if any(phrase in norm for phrase in phrases):
            abnormal = label
            break
    if abnormal is None:
        return {}
    out = {"abnormal_or_none": abnormal}
    focus = extract_region_focus(norm)
    if focus is not None:
        out["primary_focus"] = focus
    return out


def parse_4_1_4(text: str) -> Dict[str, Any]:
    match = re.match(
        r"the\s+(north|south|east|west)\s+(?:approach|side)\s+(?:is the busiest|has the heaviest traffic),\s+with\s+"
        r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:traffic participants|participants)\.",
        normalize_parse_text(text),
    )
    if not match:
        return {}
    return {"dominant_side": match.group(1), "participant_count": parse_int_token(match.group(2))}


def parse_4_2_1(text: str) -> Dict[str, Any]:
    risk = parse_yes_no(text)
    if risk is None:
        return {}
    out = {"has_speeding_risk": risk}
    if not risk:
        return out
    match = re.match(
        rf"yes[,.]?\s+a\s+([a-z0-9_ ]+?)\s+({APPROACH_LOCATION_WITH_INTERSECTION_PATTERN})\s+"
        rf"(?:is\s+(?:still\s+)?(?:moving|traveling)|appears\s+to\s+be\s+(?:moving|traveling)(?:\s+at\s+high\s+speed)?)\s+"
        rf"at(?:\s+about)?\s+(-?\d+(?:\.\d+)?)\s+m/s\.?$",
        normalize_parse_text(text),
    )
    if not match:
        return out
    out["vehicle_type"] = normalize_object_type(match.group(1))
    out["side"] = extract_side_phrase(match.group(2))
    out["numeric_targets"] = {"speed_mps": format_float(float(match.group(3)))}
    return out


def parse_4_2_2(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(
        r"a\s+([a-z- ]+?)\s+exists\s+(?:on the\s+(north|south|east|west)\s+approach|in the center area)\.",
        norm,
    )
    if not match:
        return {}
    abnormal = ABNORMAL_TEXT_TO_LABEL.get(match.group(1).strip())
    region = match.group(2) if match.group(2) else "center"
    return {"notable_abnormal": abnormal, "risk_region": region}


def parse_4_3_1(text: str) -> Dict[str, Any]:
    parsed = parse_action_state_exact(text, INTERSECTION_ACTION_BY_TEXT)
    if parsed.get("action_state") is not None:
        return parsed
    return parse_action_state_by_phrases(text, INTERSECTION_ACTION_BY_TEXT, INTERSECTION_ACTION_PHRASES)


def parse_4_3_2(text: str) -> Dict[str, Any]:
    parsed = parse_action_state_exact(text, SIDE_ACTION_BY_TEXT)
    if parsed.get("action_state") is not None:
        return parsed
    return parse_action_state_by_phrases(text, SIDE_ACTION_BY_TEXT, SIDE_ACTION_PHRASES)


def parse_4_3_3(text: str) -> Dict[str, Any]:
    parsed = parse_action_state_exact(text, LANE_ACTION_BY_TEXT)
    if parsed.get("action_state") is not None:
        return parsed
    return parse_action_state_by_phrases(text, LANE_ACTION_BY_TEXT, LANE_ACTION_PHRASES)


def parse_4_3_4(text: str) -> Dict[str, Any]:
    norm = normalize_parse_text(text)
    match = re.match(rf"the\s+([a-z0-9_ ]+?)\s+{APPROACH_LOCATION_PATTERN}\s+(.+)", norm)
    if not match:
        return {}
    action_state = None
    for suffix, state in OBJECT_ACTION_SUFFIX_TO_STATE.items():
        if norm.endswith(suffix):
            action_state = state
            break
    return {"object_type": normalize_object_type(match.group(1)), "action_state": action_state}


def flatten_structure(value: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_structure(item, next_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            flat.update(flatten_structure(item, next_prefix))
        if not value and prefix:
            flat[prefix] = []
    else:
        flat[prefix] = value
    return flat


def path_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern).replace(r"\[\*\]", r"\[\d+\]")
    return re.compile(rf"^{escaped}$")


def resolve_field_patterns(flat_gt: Dict[str, Any], patterns: Iterable[str]) -> List[str]:
    matched: List[str] = []
    for pattern in patterns:
        regex = path_to_regex(pattern)
        keys = [key for key in flat_gt if regex.match(key)]
        if keys:
            matched.extend(sorted(keys))
        elif pattern in flat_gt:
            matched.append(pattern)
    unique = []
    seen = set()
    for key in matched:
        if key not in seen and flat_gt.get(key) is not None:
            unique.append(key)
            seen.add(key)
    return unique


def get_numeric_fields_1_1_2(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    return numeric_bucket_map(count=["count"]) if gt.get("exists") else {}


def get_discrete_fields_1_3_1(gt: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    if gt.get("weather") is not None:
        fields.append("weather")
    if gt.get("time_of_day") is not None:
        fields.append("time_of_day")
    light = gt.get("light") or {}
    if light.get("sun_glare"):
        fields.append("light.sun_glare")
        if light.get("sun_glare_sides"):
            fields.append("light.sun_glare_sides[*]")
    return fields


def get_numeric_fields_4_1_2(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    counts = gt.get("counts") or {}
    if (counts.get("moving_vehicles") or 0) == 0 and (counts.get("stopped_vehicles") or 0) == 0:
        return {}
    return numeric_bucket_map(count=["counts.moving_vehicles", "counts.stopped_vehicles"])


def get_discrete_fields_4_1_3(gt: Dict[str, Any]) -> List[str]:
    fields = ["abnormal_or_none"]
    if gt.get("abnormal_or_none") != "none":
        fields.append("primary_focus")
    return fields


def get_discrete_fields_4_1_4(gt: Dict[str, Any]) -> List[str]:
    return ["dominant_side"] if gt.get("dominant_side") is not None else []


def get_numeric_fields_4_1_4(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    return numeric_bucket_map(count=["participant_count"]) if gt.get("participant_count") is not None else {}


def get_discrete_fields_2_2_6(gt: Dict[str, Any]) -> List[str]:
    fields = ["crosswalk_blocked", "side"]
    if gt.get("object_type") is not None:
        fields.append("object_type")
    return fields


def get_discrete_fields_4_2_1(gt: Dict[str, Any]) -> List[str]:
    fields = ["has_speeding_risk"]
    if gt.get("has_speeding_risk"):
        fields.extend(["vehicle_type", "side"])
    return fields


def get_numeric_fields_4_2_1(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    return numeric_bucket_map(speed=["numeric_targets.speed_mps"]) if gt.get("has_speeding_risk") else {}


def get_numeric_fields_3_3_2(gt: Dict[str, Any]) -> Dict[str, List[str]]:
    nested = ((gt.get("queue_evidence") or {}).get("queue_evidence") or {})
    fields = []
    for key in ("stopped_vehicles", "slow_vehicles", "moving_vehicles"):
        if (nested.get(key) or 0) > 0:
            fields.append(f"queue_evidence.queue_evidence.{key}")
    return numeric_bucket_map(count=fields) if fields else {}


V5_TEMPLATE_SPECS: Dict[str, TemplateEvalSpec] = {
    "1_1_1_fine_type": TemplateEvalSpec(parse_1_1_1, discrete_fields=lambda _gt: ["object_type", "side"]),
    "1_1_2_side_exists": TemplateEvalSpec(parse_1_1_2, discrete_fields=lambda _gt: ["exists"], numeric_fields=get_numeric_fields_1_1_2),
    "1_1_3_side_count": TemplateEvalSpec(
        parse_1_1_3,
        discrete_fields=lambda _gt: ["side", "object_type"],
        numeric_fields=lambda _gt: numeric_bucket_map(count=["count"]),
    ),
    "1_1_4_relative_neighbor_type": TemplateEvalSpec(
        parse_1_1_4,
        discrete_fields=lambda _gt: ["object_type", "target_object_image_refs[*].image_name"],
        numeric_fields=lambda _gt: numeric_bucket_map(
            global_3d_xy=["target_object_position.x", "target_object_position.y"],
            image_2d_xy=["target_object_image_refs[*].x1", "target_object_image_refs[*].y1"],
        ),
    ),
    "1_2_1_size_bucket": TemplateEvalSpec(parse_1_2_1, discrete_fields=lambda _gt: ["object_type", "side", "size_bucket"]),
    "1_2_2_visibility": TemplateEvalSpec(parse_1_2_2, discrete_fields=lambda _gt: ["visibility"]),
    "1_3_1_weather": TemplateEvalSpec(parse_1_3_1, discrete_fields=get_discrete_fields_1_3_1),
    "1_3_2_vehicle_signal_state": TemplateEvalSpec(parse_1_3_2, discrete_fields=lambda _gt: ["signal_state"]),
    "2_1_1_stopline_distance": TemplateEvalSpec(
        parse_2_1_1,
        discrete_fields=lambda _gt: ["object_type", "stopline_side"],
        numeric_fields=lambda _gt: numeric_bucket_map(distance=["distance_m"]),
    ),
    "2_1_2_ped_to_far_edge": TemplateEvalSpec(
        parse_2_1_2,
        discrete_fields=lambda _gt: ["crosswalk"],
        numeric_fields=lambda _gt: numeric_bucket_map(distance=["distance_m"]),
    ),
    "2_1_3_participant_distance": TemplateEvalSpec(
        parse_2_1_3,
        discrete_fields=lambda _gt: ["obj1_type", "obj2_type"],
        numeric_fields=lambda _gt: numeric_bucket_map(distance=["distance_m"]),
    ),
    "2_1_4_nearest_vehicle_to_ped": TemplateEvalSpec(
        parse_2_1_4,
        discrete_fields=lambda _gt: ["vehicle_type", "side"],
        numeric_fields=lambda _gt: numeric_bucket_map(distance=["distance_m"]),
    ),
    "2_2_1_lane_function": TemplateEvalSpec(parse_2_2_1, discrete_fields=lambda _gt: ["object_type", "side", "lane_function"]),
    "2_2_2_ped_zone": TemplateEvalSpec(parse_2_2_2, discrete_fields=lambda _gt: ["crosswalk", "ped_zone"]),
    "2_2_3_left_turn_queue_count": TemplateEvalSpec(
        parse_2_2_3,
        discrete_fields=lambda _gt: ["side", "lane_function"],
        numeric_fields=lambda _gt: numeric_bucket_map(count=["count"]),
    ),
    "2_2_4_stopline_back_5m_count": TemplateEvalSpec(
        parse_2_2_4,
        discrete_fields=lambda _gt: ["side"],
        numeric_fields=lambda _gt: numeric_bucket_map(count=["count"]),
    ),
    "2_2_5_longest_queue_lane": TemplateEvalSpec(parse_2_2_5, discrete_fields=lambda _gt: ["lane_function", "side"]),
    "2_2_6_crosswalk_blocking": TemplateEvalSpec(parse_2_2_6, discrete_fields=get_discrete_fields_2_2_6),
    "3_1_1_current_motion_state": TemplateEvalSpec(
        parse_3_1_1,
        discrete_fields=lambda _gt: ["object_type", "side", "motion_state", "accel_state"],
        numeric_fields=lambda _gt: numeric_bucket_map(speed=["speed"], acceleration=["acceleration"]),
    ),
    "3_1_2_vehicle_maneuver": TemplateEvalSpec(parse_3_1_2, discrete_fields=lambda _gt: ["object_type", "side", "maneuver"]),
    "3_2_2_future_region": TemplateEvalSpec(parse_3_2_2, discrete_fields=lambda _gt: ["object_type", "future_region"]),
    "3_2_3_waypoints": TemplateEvalSpec(
        parse_3_2_3,
        numeric_fields=lambda _gt: numeric_bucket_map(waypoint_xy=["trajectory.waypoints_xy[*].dx", "trajectory.waypoints_xy[*].dy"]),
    ),
    "3_3_1_safe_following": TemplateEvalSpec(
        parse_3_3_1,
        discrete_fields=lambda _gt: ["is_safe"],
        numeric_fields=lambda _gt: numeric_bucket_map(distance=["distance_m"]),
    ),
    "3_3_2_likely_long_queue_lane": TemplateEvalSpec(
        parse_3_3_2,
        discrete_fields=lambda _gt: ["lane_function", "side"],
        numeric_fields=get_numeric_fields_3_3_2,
    ),
    "3_4_1_vehicle_ped_conflict": TemplateEvalSpec(parse_3_4_1, discrete_fields=lambda _gt: ["has_conflict"]),
    "3_4_2_nearest_conflict_participant": TemplateEvalSpec(
        parse_3_4_2,
        discrete_fields=lambda _gt: ["conflict_partner_type", "conflict_partner_image_refs[*].image_name"],
        numeric_fields=lambda _gt: numeric_bucket_map(
            global_3d_xy=["conflict_partner_position.x", "conflict_partner_position.y"],
            image_2d_xy=["conflict_partner_image_refs[*].x1", "conflict_partner_image_refs[*].y1"],
        ),
    ),
    "3_4_3_primary_risk_subject": TemplateEvalSpec(
        parse_3_4_3,
        discrete_fields=lambda _gt: ["subject_type", "risk_reason", "subject_image_refs[*].image_name"],
        numeric_fields=lambda _gt: numeric_bucket_map(
            global_3d_xy=["subject_position.x", "subject_position.y"],
            image_2d_xy=["subject_image_refs[*].x1", "subject_image_refs[*].y1"],
        ),
    ),
    "3_4_4_risk_pattern": TemplateEvalSpec(parse_3_4_4, discrete_fields=lambda _gt: ["interaction_pattern", "side"]),
    "4_1_1_overall_state": TemplateEvalSpec(
        parse_4_1_1,
        discrete_fields=lambda _gt: ["overall_state"],
        numeric_fields=lambda _gt: numeric_bucket_map(count=["moving_vehicles", "total_vehicles"]),
    ),
    "4_1_2_side_motion_status": TemplateEvalSpec(
        parse_4_1_2,
        discrete_fields=lambda _gt: ["side", "motion_label"],
        numeric_fields=get_numeric_fields_4_1_2,
    ),
    "4_1_3_scene_summary": TemplateEvalSpec(parse_4_1_3, discrete_fields=get_discrete_fields_4_1_3),
    "4_1_4_flow_imbalance": TemplateEvalSpec(parse_4_1_4, discrete_fields=get_discrete_fields_4_1_4, numeric_fields=get_numeric_fields_4_1_4),
    "4_2_1_speeding_risk": TemplateEvalSpec(
        parse_4_2_1,
        discrete_fields=get_discrete_fields_4_2_1,
        numeric_fields=get_numeric_fields_4_2_1,
    ),
    "4_2_2_notable_abnormal": TemplateEvalSpec(parse_4_2_2, discrete_fields=lambda _gt: ["notable_abnormal", "risk_region"]),
    "4_3_1_intersection_action": TemplateEvalSpec(parse_4_3_1, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_2_side_action": TemplateEvalSpec(parse_4_3_2, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_3_lane_action": TemplateEvalSpec(parse_4_3_3, discrete_fields=lambda _gt: ["action_state"]),
    "4_3_4_object_action": TemplateEvalSpec(parse_4_3_4, discrete_fields=lambda _gt: ["object_type", "action_state"]),
}


def raw_exact_match(pred: str, ref: str) -> float:
    return float(pred.strip() == ref.strip())


def normalized_exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            cur = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = text_tokens(pred)
    ref_tokens = text_tokens(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(pred: str, ref: str, n: int) -> float:
    pred_tokens = text_tokens(pred)
    ref_tokens = text_tokens(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        pred_counts = ngram_counts(pred_tokens, k)
        ref_counts = ngram_counts(ref_tokens, k)
        total = sum(pred_counts.values())
        if total == 0:
            return 0.0
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        precisions.append((overlap + 1e-9) / (total + 1e-9))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    bp = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1))
    return bp * geo_mean


class SentenceSimilarityScorer:
    def __init__(self, model_name: str, device: str, batch_size: int):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.F = F
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _encode(self, texts: List[str]):
        embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = self.F.normalize(pooled, p=2, dim=-1)
                embeddings.append(pooled.cpu())
        return self.torch.cat(embeddings, dim=0)

    def score_pairs(self, preds: List[str], refs: List[str]) -> List[float]:
        pred_emb = self._encode(preds)
        ref_emb = self._encode(refs)
        return [(pred_emb[i] * ref_emb[i]).sum().item() for i in range(len(preds))]


class BertScoreScorer:
    def __init__(self, model_name: str, device: str, batch_size: int):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.F = F
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _embed(self, texts: List[str]):
        all_hidden, all_masks = [], []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs).last_hidden_state
                hidden = self.F.normalize(outputs, p=2, dim=-1).cpu()
                masks = inputs["attention_mask"].cpu().bool()
                special = self.torch.zeros_like(masks)
                if self.tokenizer.cls_token_id is not None:
                    special |= inputs["input_ids"].cpu() == self.tokenizer.cls_token_id
                if self.tokenizer.sep_token_id is not None:
                    special |= inputs["input_ids"].cpu() == self.tokenizer.sep_token_id
                if self.tokenizer.bos_token_id is not None:
                    special |= inputs["input_ids"].cpu() == self.tokenizer.bos_token_id
                if self.tokenizer.eos_token_id is not None:
                    special |= inputs["input_ids"].cpu() == self.tokenizer.eos_token_id
                masks &= ~special
                all_hidden.extend(hidden)
                all_masks.extend(masks)
        return all_hidden, all_masks

    def score_pairs(self, preds: List[str], refs: List[str]) -> List[float]:
        pred_hidden, pred_masks = self._embed(preds)
        ref_hidden, ref_masks = self._embed(refs)
        scores = []
        for ph, pm, rh, rm in zip(pred_hidden, pred_masks, ref_hidden, ref_masks):
            p = ph[pm]
            r = rh[rm]
            if p.numel() == 0 or r.numel() == 0:
                scores.append(0.0)
                continue
            sim = p @ r.T
            precision = sim.max(dim=1).values.mean().item()
            recall = sim.max(dim=0).values.mean().item()
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
            scores.append(f1)
        return scores


def read_sidecar(path: Path, split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    require(rows, f"No sidecar samples found for split={split}")
    return rows


def read_predictions(path: Path, sidecar_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    prediction_keys = ("prediction", "predict", "answer", "response", "output")

    def resolve_prediction_text(row: Dict[str, Any], qid: str) -> str:
        for key in prediction_keys:
            if key in row:
                pred = row[key]
                require(isinstance(pred, str), f"Missing prediction text for question_id={qid}")
                return pred
        raise ValueError(f"Missing prediction text for question_id={qid}")

    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
        if rows and "question_id" in rows[0]:
            out = {}
            for row in rows:
                qid = str(row["question_id"])
                out[qid] = resolve_prediction_text(row, qid)
            return out
        if rows and "predict" in rows[0]:
            require(len(rows) == len(sidecar_rows), "Prediction row count does not match sidecar rows for order-based alignment")
            return {str(row["question_id"]): pred_row["predict"] for row, pred_row in zip(sidecar_rows, rows)}
        raise ValueError(f"Unsupported JSONL prediction format: {path}")

    payload = load_json(path)
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    if isinstance(payload, list):
        out = {}
        for row in payload:
            require(isinstance(row, dict) and "question_id" in row, f"Unsupported JSON list prediction row: {row}")
            qid = str(row["question_id"])
            out[qid] = resolve_prediction_text(row, qid)
        return out
    raise ValueError(f"Unsupported prediction payload: {path}")


def read_prediction_metadata(path: Path, sidecar_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if path.suffix != ".jsonl":
        return {}

    rows = load_jsonl(path)
    if not rows:
        return {}

    if "question_id" in rows[0]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            qid = str(row["question_id"])
            out[qid] = {key: value for key, value in row.items() if key not in {"question_id", "prediction", "predict", "answer", "response", "output"}}
        return out

    if "predict" in rows[0]:
        require(len(rows) == len(sidecar_rows), "Prediction row count does not match sidecar rows for order-based alignment")
        out = {}
        for row, pred_row in zip(sidecar_rows, rows):
            out[str(row["question_id"])] = {
                key: value for key, value in pred_row.items() if key not in {"prediction", "predict", "answer", "response", "output"}
            }
        return out

    return {}


def build_records(sidecar_rows: List[Dict[str, Any]], predictions: Dict[str, str]) -> List[SampleRecord]:
    records = []
    for row in sidecar_rows:
        qid = str(row["question_id"])
        require(qid in predictions, f"Missing prediction for question_id={qid}")
        records.append(
            SampleRecord(
                question_id=qid,
                scene_id=row["scene_id"],
                chapter=row["chapter"],
                section=row["section"],
                subtemplate=row["subtemplate"],
                answer=row["answer"],
                prediction=predictions[qid],
                structured_targets=row["structured_targets"],
            )
        )
    return records


def weighted_average(metrics: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    total_weight = 0.0
    weighted_sum = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if value is None:
            continue
        weighted_sum += value * weight
        total_weight += weight
    return None if total_weight == 0 else weighted_sum / total_weight


def mean(values: List[float]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


def bounded_bucket_error(raw_bucket_error: Optional[float], coverage: Optional[float], expected: int) -> Optional[float]:
    if expected <= 0:
        return None
    raw_value = 0.0 if raw_bucket_error is None else min(max(float(raw_bucket_error), 0.0), 1.0)
    coverage_value = 0.0 if coverage is None else min(max(float(coverage), 0.0), 1.0)
    return 1.0 - (1.0 - raw_value) * coverage_value


def discrete_field_mode(_field_name: str) -> str:
    return "exact"


def has_parsed_semantic_value(field_name: str, value: Any) -> bool:
    return bool(semantic_value_tokens(field_name, value))


def exact_label_metrics(field_name: str, gt_value: Any, pred_value: Any) -> Dict[str, Any]:
    gt_tokens = semantic_value_tokens(field_name, gt_value)
    pred_tokens = semantic_value_tokens(field_name, pred_value)
    parsed = bool(pred_tokens)
    matched = bool(gt_tokens) and pred_tokens == gt_tokens
    score = 1.0 if matched else 0.0
    return {
        "mode": "exact",
        "parsed": parsed,
        "precision": score,
        "recall": score,
        "f1": score,
    }


def token_precision_recall_f1(gt_tokens: List[str], pred_tokens: List[str]) -> Dict[str, float]:
    gt_counts = Counter(gt_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(count, pred_counts[token]) for token, count in gt_counts.items())
    pred_total = sum(pred_counts.values())
    gt_total = sum(gt_counts.values())
    precision = 0.0 if pred_total == 0 else overlap / pred_total
    recall = 0.0 if gt_total == 0 else overlap / gt_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_discrete_fields(gt: Dict[str, Any], pred: Dict[str, Any], patterns: Sequence[str]) -> Dict[str, Any]:
    gt_flat = flatten_structure(gt)
    pred_flat = flatten_structure(pred)
    field_names = resolve_field_patterns(gt_flat, patterns)
    field_scores = {}
    valid = 0
    for field_name in field_names:
        gt_value = gt_flat.get(field_name)
        pred_value = pred_flat.get(field_name)
        mode = discrete_field_mode(field_name)
        if mode == "exact":
            field_metric = exact_label_metrics(field_name, gt_value, pred_value)
        else:
            gt_tokens = semantic_value_tokens(field_name, gt_value)
            pred_tokens = semantic_value_tokens(field_name, pred_value)
            field_metric = {
                "mode": mode,
                "parsed": has_parsed_semantic_value(field_name, pred_value),
                **token_precision_recall_f1(gt_tokens, pred_tokens),
            }
        if field_metric["parsed"]:
            valid += 1
        field_scores[field_name] = field_metric
    precisions = [item["precision"] for item in field_scores.values()]
    recalls = [item["recall"] for item in field_scores.values()]
    f1s = [item["f1"] for item in field_scores.values()]
    expected = len(field_names)
    return {
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "valid": valid,
        "expected": expected,
        "coverage": None if expected == 0 else valid / expected,
        "field_scores": field_scores,
    }


def evaluate_numeric_like_fields(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    patterns: Sequence[str],
    *,
    integer_only: bool,
    threshold: float,
) -> Dict[str, Any]:
    gt_flat = flatten_structure(gt)
    pred_flat = flatten_structure(pred)
    field_names = resolve_field_patterns(gt_flat, patterns)
    field_errors = {}
    squared_errors = {}
    clipped_errors = []
    valid = 0
    expected = 0
    safe_threshold = max(float(threshold), 1e-12)
    for field_name in field_names:
        gt_value = gt_flat.get(field_name)
        if not isinstance(gt_value, (int, float)) or isinstance(gt_value, bool):
            continue
        expected += 1
        pred_value = pred_flat.get(field_name)
        if not isinstance(pred_value, (int, float)) or isinstance(pred_value, bool):
            continue
        if integer_only:
            gt_num = int(gt_value)
            pred_num = int(round(float(pred_value)))
        else:
            gt_num = float(gt_value)
            pred_num = float(pred_value)
        error = abs(pred_num - gt_num)
        field_errors[field_name] = error
        squared_errors[field_name] = error * error
        clipped_errors.append(min(error / safe_threshold, 1.0))
        valid += 1
    errors = list(field_errors.values())
    mae = mean(errors)
    rmse = None if not errors else math.sqrt(sum(squared_errors.values()) / len(squared_errors))
    raw_bucket_error = mean(clipped_errors)
    coverage = None if expected == 0 else valid / expected
    bucket_error = bounded_bucket_error(raw_bucket_error, coverage, expected)
    return {
        "mae": mae,
        "rmse": rmse,
        "raw_bucket_error": raw_bucket_error,
        "bucket_error": bucket_error,
        "valid": valid,
        "expected": expected,
        "coverage": coverage,
        "field_errors": field_errors,
    }


def coordinate_pair_for_field(field_name: str) -> Optional[Tuple[str, str, str]]:
    for x_suffix, y_suffix in ((".x", ".y"), (".x1", ".y1"), (".dx", ".dy")):
        if field_name.endswith(x_suffix):
            return field_name[: -len(x_suffix)], x_suffix, y_suffix
    return None


def evaluate_euclidean_numeric_fields(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    patterns: Sequence[str],
    *,
    threshold: float,
) -> Dict[str, Any]:
    gt_flat = flatten_structure(gt)
    pred_flat = flatten_structure(pred)
    field_names = resolve_field_patterns(gt_flat, patterns)
    pair_specs = []
    seen_pairs = set()
    for field_name in field_names:
        pair = coordinate_pair_for_field(field_name)
        if pair is None:
            continue
        prefix, x_suffix, y_suffix = pair
        if prefix in seen_pairs:
            continue
        seen_pairs.add(prefix)
        pair_specs.append((prefix, f"{prefix}{x_suffix}", f"{prefix}{y_suffix}"))

    field_errors = {}
    squared_errors = {}
    clipped_errors = []
    valid = 0
    expected = 0
    safe_threshold = max(float(threshold), 1e-12)
    for pair_name, x_field, y_field in pair_specs:
        gt_x = gt_flat.get(x_field)
        gt_y = gt_flat.get(y_field)
        if (
            not isinstance(gt_x, (int, float))
            or isinstance(gt_x, bool)
            or not isinstance(gt_y, (int, float))
            or isinstance(gt_y, bool)
        ):
            continue
        expected += 1
        pred_x = pred_flat.get(x_field)
        pred_y = pred_flat.get(y_field)
        if (
            not isinstance(pred_x, (int, float))
            or isinstance(pred_x, bool)
            or not isinstance(pred_y, (int, float))
            or isinstance(pred_y, bool)
        ):
            continue
        error = math.hypot(float(pred_x) - float(gt_x), float(pred_y) - float(gt_y))
        field_errors[pair_name] = error
        squared_errors[pair_name] = error * error
        clipped_errors.append(min(error / safe_threshold, 1.0))
        valid += 1

    errors = list(field_errors.values())
    mae = mean(errors)
    rmse = None if not errors else math.sqrt(sum(squared_errors.values()) / len(squared_errors))
    raw_bucket_error = mean(clipped_errors)
    coverage = None if expected == 0 else valid / expected
    bucket_error = bounded_bucket_error(raw_bucket_error, coverage, expected)
    return {
        "mae": mae,
        "rmse": rmse,
        "raw_bucket_error": raw_bucket_error,
        "bucket_error": bucket_error,
        "valid": valid,
        "expected": expected,
        "coverage": coverage,
        "field_errors": field_errors,
    }


def empty_numeric_metrics(*, include_field_errors: bool) -> Dict[str, Any]:
    metrics = {
        "mae": None,
        "rmse": None,
        "raw_bucket_error": None,
        "bucket_error": None,
        "valid": 0,
        "expected": 0,
        "coverage": None,
    }
    if include_field_errors:
        metrics["field_errors"] = {}
    return metrics


def aggregate_numeric_metric_entries(entries: Sequence[Dict[str, Any]], *, include_field_errors: bool) -> Dict[str, Any]:
    field_errors: Dict[str, float] = {}
    expected = 0
    valid = 0
    raw_error_sum = 0.0
    raw_squared_error_sum = 0.0
    raw_bucket_error_sum = 0.0
    for entry in entries:
        entry_expected = int(entry.get("expected") or 0)
        entry_valid = int(entry.get("valid") or 0)
        expected += entry_expected
        valid += entry_valid
        if entry_valid:
            entry_mae = entry.get("mae")
            entry_rmse = entry.get("rmse")
            entry_raw_bucket_error = entry.get("raw_bucket_error")
            if entry_raw_bucket_error is None:
                entry_raw_bucket_error = entry.get("bucket_error")
            if entry_mae is not None:
                raw_error_sum += float(entry_mae) * entry_valid
            if entry_rmse is not None:
                raw_squared_error_sum += (float(entry_rmse) ** 2) * entry_valid
            if entry_raw_bucket_error is not None:
                raw_bucket_error_sum += float(entry_raw_bucket_error) * entry_valid
        if include_field_errors:
            field_errors.update(entry.get("field_errors", {}))
    coverage = None if expected == 0 else valid / expected
    raw_bucket_error = None if valid == 0 else raw_bucket_error_sum / valid
    bucket_error = bounded_bucket_error(raw_bucket_error, coverage, expected)
    metrics = {
        "mae": None if valid == 0 else raw_error_sum / valid,
        "rmse": None if valid == 0 else math.sqrt(raw_squared_error_sum / valid),
        "raw_bucket_error": raw_bucket_error,
        "bucket_error": bucket_error,
        "valid": valid,
        "expected": expected,
        "coverage": coverage,
    }
    if include_field_errors:
        metrics["field_errors"] = field_errors
    return metrics


def compute_total_bucket_error(by_bucket: Dict[str, Dict[str, Any]]) -> Optional[float]:
    bucket_errors = []
    for bucket in NUMERIC_ERROR_BUCKETS:
        metrics = by_bucket.get(bucket)
        if not metrics or (metrics.get("expected") or 0) == 0:
            continue
        bucket_error = metrics.get("bucket_error")
        if bucket_error is not None:
            bucket_errors.append(float(bucket_error))
    return mean(bucket_errors)


def evaluate_numeric_fields_by_bucket(
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    bucket_patterns: Dict[str, List[str]],
) -> Dict[str, Any]:
    gt_flat = flatten_structure(gt)
    resolved_keys: Dict[str, List[str]] = {}
    key_to_bucket: Dict[str, str] = {}
    for bucket in NUMERIC_ERROR_BUCKETS:
        patterns = bucket_patterns.get(bucket) or []
        if not patterns:
            continue
        resolved = resolve_field_patterns(gt_flat, patterns)
        resolved_keys[bucket] = resolved
        for key in resolved:
            prev_bucket = key_to_bucket.get(key)
            require(
                prev_bucket is None or prev_bucket == bucket,
                f"Numeric field '{key}' matched multiple buckets: {prev_bucket}, {bucket}",
            )
            key_to_bucket[key] = bucket

    by_bucket: Dict[str, Dict[str, Any]] = {}
    bucket_entries = []
    for bucket in NUMERIC_ERROR_BUCKETS:
        patterns = resolved_keys.get(bucket, [])
        if not patterns:
            metrics = empty_numeric_metrics(include_field_errors=True)
        elif bucket in EUCLIDEAN_NUMERIC_BUCKETS:
            metrics = evaluate_euclidean_numeric_fields(
                gt,
                pred,
                patterns,
                threshold=NUMERIC_TOTAL_ERROR_THRESHOLDS.get(bucket, 1.0),
            )
        else:
            metrics = evaluate_numeric_like_fields(
                gt,
                pred,
                patterns,
                integer_only=bucket in INTEGER_NUMERIC_BUCKETS,
                threshold=NUMERIC_TOTAL_ERROR_THRESHOLDS.get(bucket, 1.0),
            )
        by_bucket[bucket] = metrics
        bucket_entries.append(metrics)

    overall = aggregate_numeric_metric_entries(bucket_entries, include_field_errors=True)
    overall["by_bucket"] = by_bucket
    overall["total_bucket_error"] = compute_total_bucket_error(by_bucket)
    return overall


def aggregate_text_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ["normalized_exact_match", "bleu_4", "rouge_l", "bertscore", "simcse", "weighted_text_score"]
    summary = {"count": len(rows)}
    for metric_name in metric_names:
        values = [row["text_metrics"][metric_name] for row in rows if row["text_metrics"].get(metric_name) is not None]
        summary[metric_name] = mean(values)
    return summary


def aggregate_numeric_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = [row["numeric_metrics"] for row in rows]
    summary = aggregate_numeric_metric_entries(entries, include_field_errors=False)
    summary["by_bucket"] = {}
    for bucket in NUMERIC_ERROR_BUCKETS:
        bucket_entries = [row["numeric_metrics"]["by_bucket"][bucket] for row in rows]
        summary["by_bucket"][bucket] = aggregate_numeric_metric_entries(bucket_entries, include_field_errors=False)
    summary["total_bucket_error"] = compute_total_bucket_error(summary["by_bucket"])
    return summary


def public_bucket_numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "bucket_error": metrics.get("bucket_error"),
        "valid": int(metrics.get("valid") or 0),
        "expected": int(metrics.get("expected") or 0),
        "coverage": metrics.get("coverage"),
    }


def public_numeric_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "valid": int(summary.get("valid") or 0),
        "expected": int(summary.get("expected") or 0),
        "coverage": summary.get("coverage"),
        "total_bucket_error": summary.get("total_bucket_error"),
        "by_bucket": {
            bucket: public_bucket_numeric_metrics(summary["by_bucket"][bucket])
            for bucket in NUMERIC_ERROR_BUCKETS
        },
    }


def aggregate_discrete_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    precisions = []
    recalls = []
    f1s = []
    valid = 0
    expected = 0
    for row in rows:
        group = row["discrete_metrics"]
        for field_metric in group["field_scores"].values():
            precisions.append(field_metric["precision"])
            recalls.append(field_metric["recall"])
            f1s.append(field_metric["f1"])
            valid += 1 if field_metric.get("parsed") else 0
            expected += 1
    return {
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "valid": valid,
        "expected": expected,
        "coverage": None if expected == 0 else valid / expected,
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "text": aggregate_text_group(rows),
        "numeric_error": public_numeric_metrics(aggregate_numeric_group(rows)),
        "discrete_semantics": aggregate_discrete_group(rows),
    }


def build_parser_diagnostics(
    parsed_prediction: Dict[str, Any],
    discrete_metrics: Dict[str, Any],
    numeric_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    field_scores = discrete_metrics.get("field_scores", {})
    parsed_fields = [field_name for field_name, metrics in field_scores.items() if metrics.get("parsed")]
    missing_fields = [field_name for field_name, metrics in field_scores.items() if not metrics.get("parsed")]

    expected_buckets = []
    fully_covered_buckets = []
    partially_covered_buckets = []
    missing_buckets = []
    for bucket_name, bucket_metrics in numeric_metrics.items():
        if not isinstance(bucket_metrics, dict):
            continue
        expected = int(bucket_metrics.get("expected") or 0)
        valid = int(bucket_metrics.get("valid") or 0)
        if expected <= 0:
            continue
        expected_buckets.append(bucket_name)
        if valid == expected:
            fully_covered_buckets.append(bucket_name)
        elif valid > 0:
            partially_covered_buckets.append(bucket_name)
        else:
            missing_buckets.append(bucket_name)

    return {
        "parse_nonempty": bool(parsed_prediction),
        "discrete_expected_fields": sorted(field_scores.keys()),
        "discrete_parsed_fields": parsed_fields,
        "discrete_missing_fields": missing_fields,
        "all_discrete_fields_parsed": bool(discrete_metrics.get("expected") == discrete_metrics.get("valid")),
        "zero_discrete_fields_parsed": bool(discrete_metrics.get("expected", 0) > 0 and discrete_metrics.get("valid", 0) == 0),
        "numeric_expected_buckets": expected_buckets,
        "numeric_fully_covered_buckets": fully_covered_buckets,
        "numeric_partially_covered_buckets": partially_covered_buckets,
        "numeric_missing_buckets": missing_buckets,
    }


def summarize_parser_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    parse_nonempty_count = 0
    all_discrete_parsed_count = 0
    zero_discrete_parsed_count = 0
    numeric_missing_bucket_counts = {bucket: 0 for bucket in NUMERIC_ERROR_BUCKETS}
    discrete_missing_field_counts: Counter[str] = Counter()

    for row in rows:
        diagnostics = row["parser_diagnostics"]
        if diagnostics["parse_nonempty"]:
            parse_nonempty_count += 1
        if diagnostics["all_discrete_fields_parsed"]:
            all_discrete_parsed_count += 1
        if diagnostics["zero_discrete_fields_parsed"]:
            zero_discrete_parsed_count += 1
        discrete_missing_field_counts.update(diagnostics["discrete_missing_fields"])
        for bucket_name in diagnostics["numeric_missing_buckets"]:
            numeric_missing_bucket_counts[bucket_name] += 1

    sample_count = len(rows)
    return {
        "parse_nonempty_count": parse_nonempty_count,
        "parse_nonempty_rate": None if sample_count == 0 else parse_nonempty_count / sample_count,
        "all_discrete_fields_parsed_count": all_discrete_parsed_count,
        "all_discrete_fields_parsed_rate": None if sample_count == 0 else all_discrete_parsed_count / sample_count,
        "zero_discrete_fields_parsed_count": zero_discrete_parsed_count,
        "zero_discrete_fields_parsed_rate": None if sample_count == 0 else zero_discrete_parsed_count / sample_count,
        "top_discrete_missing_fields": discrete_missing_field_counts.most_common(20),
        "numeric_missing_bucket_counts": numeric_missing_bucket_counts,
    }


def summarize_lidar_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    valid_count = 0
    invalid_all_zero_count = 0
    missing_object_count = 0
    missing_scene_count = 0
    prediction_step_missing_object_count = 0
    prediction_step_missing_scene_count = 0
    prediction_step_missing_object_key_count = 0
    prediction_step_missing_scene_key_count = 0

    for row in rows:
        status = str(row.get("lidar_diagnostics_status") or "missing")
        status_counts[status] += 1
        if status == "ok":
            valid_count += 1
        if status == "invalid_all_zero":
            invalid_all_zero_count += 1
        if not bool(row.get("has_lidar_object_features", False)):
            missing_object_count += 1
        if not bool(row.get("has_lidar_scene_features", False)):
            missing_scene_count += 1
        if not bool(row.get("prediction_step_has_lidar_object_features", False)):
            prediction_step_missing_object_count += 1
        if not bool(row.get("prediction_step_has_lidar_scene_features", False)):
            prediction_step_missing_scene_count += 1
        if not bool(row.get("prediction_step_has_lidar_object_key", False)):
            prediction_step_missing_object_key_count += 1
        if not bool(row.get("prediction_step_has_lidar_scene_key", False)):
            prediction_step_missing_scene_key_count += 1

    sample_count = len(rows)
    return {
        "lidar_diagnostics_valid_count": valid_count,
        "lidar_diagnostics_valid_rate": None if sample_count == 0 else valid_count / sample_count,
        "lidar_diagnostics_invalid_all_zero_count": invalid_all_zero_count,
        "prediction_step_missing_object_feature_count": prediction_step_missing_object_count,
        "prediction_step_missing_scene_feature_count": prediction_step_missing_scene_count,
        "prediction_step_missing_object_key_count": prediction_step_missing_object_key_count,
        "prediction_step_missing_scene_key_count": prediction_step_missing_scene_key_count,
        "lidar_batches_missing_object_features_count": missing_object_count,
        "lidar_batches_missing_scene_features_count": missing_scene_count,
        "lidar_diagnostics_status_counts": dict(sorted(status_counts.items())),
    }


def aggregate_macro(per_group: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sub_summaries = list(per_group.values())
    if not sub_summaries:
        return {}
    macro = {
        "text": {},
        "numeric_error": {},
        "discrete_semantics": {},
    }
    for group_name, metric_names in {
        "text": ["normalized_exact_match", "bleu_4", "rouge_l", "bertscore", "simcse", "weighted_text_score"],
        "numeric_error": ["total_bucket_error", "coverage"],
        "discrete_semantics": ["precision", "recall", "f1", "coverage"],
    }.items():
        for metric_name in metric_names:
            values = [summary[group_name][metric_name] for summary in sub_summaries if summary[group_name].get(metric_name) is not None]
            macro[group_name][metric_name] = mean(values)
    macro["numeric_error"]["valid"] = sum(int(summary["numeric_error"].get("valid") or 0) for summary in sub_summaries)
    macro["numeric_error"]["expected"] = sum(int(summary["numeric_error"].get("expected") or 0) for summary in sub_summaries)
    macro["numeric_error"]["by_bucket"] = {}
    for bucket in NUMERIC_ERROR_BUCKETS:
        macro["numeric_error"]["by_bucket"][bucket] = {}
        macro["numeric_error"]["by_bucket"][bucket]["valid"] = sum(
            int(summary["numeric_error"]["by_bucket"][bucket].get("valid") or 0) for summary in sub_summaries
        )
        macro["numeric_error"]["by_bucket"][bucket]["expected"] = sum(
            int(summary["numeric_error"]["by_bucket"][bucket].get("expected") or 0) for summary in sub_summaries
        )
        for metric_name in ("mae", "rmse", "bucket_error", "coverage"):
            values = [
                summary["numeric_error"]["by_bucket"][bucket][metric_name]
                for summary in sub_summaries
                if summary["numeric_error"]["by_bucket"][bucket].get(metric_name) is not None
            ]
            macro["numeric_error"]["by_bucket"][bucket][metric_name] = mean(values)
    macro["subtemplate_count"] = len(sub_summaries)
    return macro


def compute_text_metrics(records: List[SampleRecord]) -> List[Dict[str, Optional[float]]]:
    sample_metrics = []
    for record in records:
        metrics = {
            "normalized_exact_match": normalized_exact_match(record.prediction, record.answer),
            "bleu_4": bleu_n(record.prediction, record.answer, 4),
            "rouge_l": rouge_l(record.prediction, record.answer),
            "bertscore": None,
            "simcse": None,
        }
        sample_metrics.append(metrics)
    return sample_metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_rows = read_sidecar(args.sidecar_jsonl, args.split, args.limit)
    predictions = read_predictions(args.predictions, sidecar_rows)
    prediction_metadata = read_prediction_metadata(args.predictions, sidecar_rows)
    records = build_records(sidecar_rows, predictions)

    unknown_subtemplates = sorted({record.subtemplate for record in records if record.subtemplate not in V5_TEMPLATE_SPECS})
    require(not unknown_subtemplates, f"Unsupported V5 subtemplates in evaluation input: {unknown_subtemplates}")

    text_metrics = compute_text_metrics(records)
    if not args.skip_semantic_metrics:
        preds = [record.prediction for record in records]
        refs = [record.answer for record in records]
        bert_scores = BertScoreScorer(args.bertscore_model, args.device, args.batch_size).score_pairs(preds, refs)
        sim_scores = SentenceSimilarityScorer(args.sim_model, args.device, args.batch_size).score_pairs(preds, refs)
        for metrics, bert, sim in zip(text_metrics, bert_scores, sim_scores):
            metrics["bertscore"] = bert
            metrics["simcse"] = sim
    for metrics in text_metrics:
        metrics["weighted_text_score"] = weighted_average(metrics, TEXT_METRIC_WEIGHTS)

    per_sample_rows = []
    for record, sample_text_metrics in zip(records, text_metrics):
        spec = V5_TEMPLATE_SPECS[record.subtemplate]
        parsed_prediction = spec.parser(record.prediction)
        discrete_metrics = evaluate_discrete_fields(record.structured_targets, parsed_prediction, spec.discrete_fields(record.structured_targets))
        numeric_metrics = evaluate_numeric_fields_by_bucket(
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
                "parser_diagnostics": build_parser_diagnostics(parsed_prediction, discrete_metrics, numeric_metrics),
                "lidar_diagnostics_status": prediction_metadata.get(record.question_id, {}).get(
                    "lidar_diagnostics_status", "missing"
                ),
                "lidar_diagnostics_reason": prediction_metadata.get(record.question_id, {}).get(
                    "lidar_diagnostics_reason"
                ),
                "prediction_step_has_lidar_object_key": bool(
                    prediction_metadata.get(record.question_id, {}).get("prediction_step_has_lidar_object_key", False)
                ),
                "prediction_step_has_lidar_scene_key": bool(
                    prediction_metadata.get(record.question_id, {}).get("prediction_step_has_lidar_scene_key", False)
                ),
                "prediction_step_has_lidar_object_features": bool(
                    prediction_metadata.get(record.question_id, {}).get(
                        "prediction_step_has_lidar_object_features", False
                    )
                ),
                "prediction_step_has_lidar_scene_features": bool(
                    prediction_metadata.get(record.question_id, {}).get(
                        "prediction_step_has_lidar_scene_features", False
                    )
                ),
                "has_lidar_object_features": bool(
                    prediction_metadata.get(record.question_id, {}).get("has_lidar_object_features", False)
                ),
                "has_lidar_scene_features": bool(
                    prediction_metadata.get(record.question_id, {}).get("has_lidar_scene_features", False)
                ),
                "prepared_has_lidar_object_features": bool(
                    prediction_metadata.get(record.question_id, {}).get("prepared_has_lidar_object_features", False)
                ),
                "prepared_has_lidar_scene_features": bool(
                    prediction_metadata.get(record.question_id, {}).get("prepared_has_lidar_scene_features", False)
                ),
                "lidar_diagnostics": prediction_metadata.get(record.question_id, {}).get("lidar_diagnostics", {}),
            }
        )

    per_subtemplate_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_chapter_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_section_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        per_subtemplate_rows[row["subtemplate"]].append(row)
        per_chapter_rows[row["chapter"]].append(row)
        per_section_rows[row["section"]].append(row)

    per_subtemplate = {name: summarize_rows(rows) for name, rows in sorted(per_subtemplate_rows.items())}
    per_chapter = {name: summarize_rows(rows) for name, rows in sorted(per_chapter_rows.items())}
    per_section = {name: summarize_rows(rows) for name, rows in sorted(per_section_rows.items())}

    metrics_payload = {
        "metadata": {
            "sidecar_jsonl": str(args.sidecar_jsonl.resolve()),
            "predictions": str(args.predictions.resolve()),
            "split": args.split,
            "sample_count": len(records),
            "semantic_metrics_enabled": not args.skip_semantic_metrics,
            "bertscore_model": None if args.skip_semantic_metrics else args.bertscore_model,
            "sim_model": None if args.skip_semantic_metrics else args.sim_model,
            "text_metric_weights": TEXT_METRIC_WEIGHTS,
            "evaluation_schema": {
                "text": "normalized_exact_match, bleu_4, rouge_l, bertscore, simcse, weighted_text_score",
                "numeric_error": "top-level numeric metrics retain valid, expected, coverage, total_bucket_error, and by_bucket; each by_bucket entry retains mae/rmse diagnostics plus bounded bucket_error in [0,1], where bucket_error = 1 - (1 - raw_bucket_error) * coverage",
                "discrete_semantics": "field-level exact-label precision/recall/f1 plus parse coverage over answer-required semantic fields",
            },
            "numeric_total_error_thresholds": NUMERIC_TOTAL_ERROR_THRESHOLDS,
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
        "overall_text": aggregate_text_group(per_sample_rows),
        "overall_numeric_error": public_numeric_metrics(aggregate_numeric_group(per_sample_rows)),
        "overall_discrete_semantics": aggregate_discrete_group(per_sample_rows),
        "overall_parser_diagnostics": summarize_parser_diagnostics(per_sample_rows),
        "overall_lidar_diagnostics": summarize_lidar_diagnostics(per_sample_rows),
        "macro_by_subtemplate": aggregate_macro(per_subtemplate),
        "per_subtemplate": per_subtemplate,
        "per_chapter": per_chapter,
        "per_section": per_section,
    }

    per_sample_output_rows = []
    for row in per_sample_rows:
        row_out = dict(row)
        row_out["numeric_metrics"] = public_numeric_metrics(row["numeric_metrics"])
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
