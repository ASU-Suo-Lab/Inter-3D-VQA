from __future__ import annotations

import json
import pathlib
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from traffixqwen_v5.config.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_IMAGE_TOKEN_COST,
    DEFAULT_INFO_PKL,
    DEFAULT_MM_PROJECTOR,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREPARED_DIR,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_QA_JSON,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMPORAL_WINDOW,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SCENES,
    DEFAULT_VIEWS,
    DEFAULT_VISION_TOWER,
    LLM_ROOT,
    REPO_ROOT,
    VLM_ROOT,
)

INFO_CAM_TO_VIEW = {
    "CAM_SOUTH": "north",
    "CAM_NORTH": "south",
    "CAM_WEST": "east",
    "CAM_EAST": "west",
}
VIEW_TO_INFO_CAM = {view: cam for cam, view in INFO_CAM_TO_VIEW.items()}

V5_SUBTEMPLATE_TASK_INSTRUCTIONS = {
    "1_1_1_fine_type": (
        "Answer briefly with the referenced object's fine-grained type plus its current "
        "approach/location phrase.\n"
        "Do not expand to motion or interaction details."
    ),
    "1_1_2_side_exists": (
        "For this task, treat vulnerable road users as pedestrians, bicycles, motorcycles, "
        "and golf carts only.\n"
        "If yes, mention the count concisely; if not, answer briefly without extra explanation."
    ),
    "1_1_3_side_count": "Report only that count and avoid expanding into scene-level summary.",
    "1_1_4_relative_neighbor_type": (
        "Interpret the relative direction in the referenced object's own heading frame, not in "
        "the global north-up frame.\n"
        "Answer with the target object's type plus its location-and-image reference only, "
        "without extra scene description."
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
        "Answer only with the available environmental attributes for this task: weather, time "
        "of day, and any visible strong sun glare."
    ),
    "1_3_2_vehicle_signal_state": (
        "Use only directly visible signal evidence, and do not infer hidden lights from traffic "
        "behavior or traffic flow.\n"
        "Answer only with the signal-state label allowed for this task."
    ),
    "2_1_1_stopline_distance": (
        "Focus only on that stop-line distance, retain the numeric value with units, and do not "
        "expand into motion or queue interpretation."
    ),
    "2_1_2_ped_to_far_edge": (
        "Use the pedestrian's current crossing direction and crosswalk context, and answer only "
        "with the relevant exit-approach context plus the distance value."
    ),
    "2_1_3_participant_distance": "Answer only with the distance value and keep the unit.",
    "2_1_4_nearest_vehicle_to_ped": "Answer only with the vehicle type, its approach, and the distance.",
    "2_2_1_lane_function": (
        "Judge the current lane assignment rather than the object's future maneuver, and answer "
        "only with the lane-function label."
    ),
    "2_2_2_ped_zone": "Answer only with the normalized zone label for this task, and do not add extra explanation.",
    "2_2_3_left_turn_queue_count": (
        "Focus only on that approach-lane pair, return only the count, and do not compare with other lanes."
    ),
    "2_2_4_stopline_back_5m_count": (
        "Return only the count and do not turn the answer into a congestion summary."
    ),
    "2_2_5_longest_queue_lane": (
        "State the winning lane with its approach and concise local queue evidence, without "
        "turning the answer into control advice."
    ),
    "2_2_6_crosswalk_blocking": (
        "If yes, mention the blocking vehicle type and the relevant approach briefly; otherwise "
        "keep the answer brief."
    ),
    "3_1_1_current_motion_state": (
        "Report the current state together with the current speed in m/s.\n"
        "Use standing/walking/running for pedestrians and stopped/starting/moving/braking/"
        "creeping for other objects.\n"
        "For non-pedestrians in starting or braking states, also report accelerating/"
        "decelerating together with the current acceleration in m/s^2.\n"
        "Focus only on the current state of the referenced object, without expanding into future "
        "intent or maneuver explanation."
    ),
    "3_1_2_vehicle_maneuver": (
        "Focus on the dominant current maneuver only, and do not expand into a full future path "
        "or trajectory explanation."
    ),
    "3_2_2_future_region": "Return only the most likely region and avoid expanding into a multi-step path description.",
    "3_2_3_waypoints": (
        "Return the short-horizon future trajectory only as trajectory coordinates in temporal order.\n"
        "Each point must be an (dx,dy) XY offset in meters relative to the referenced object's "
        "current position, not an absolute global coordinate.\n"
        "Follow the required `Future trajectory:(x1,y1),(x2,y2),(x3,y3),(x4,y4)` format exactly "
        "and avoid any explanatory text."
    ),
    "3_3_1_safe_following": (
        "Answer with the safety judgment, the current distance, and the time headway only, "
        "without expanding into broader collision analysis."
    ),
    "3_3_2_likely_long_queue_lane": (
        "State the winning lane with its approach and concise local queue evidence, without "
        "turning the answer into control advice."
    ),
    "3_4_1_vehicle_ped_conflict": "Focus only on this pair, and keep negative answers brief.",
    "3_4_2_nearest_conflict_participant": (
        "Answer only with the participant type plus its location-and-image reference, using X,Y "
        "and the available image_name/x/y entries without object ID or speed."
    ),
    "3_4_3_primary_risk_subject": (
        "Answer only with the participant type plus its location-and-image reference, followed by "
        "the dominant risk reason, without object ID or speed."
    ),
    "3_4_4_risk_pattern": (
        "Return only the dominant pattern together with the relevant approach, without expanding "
        "into a long event description."
    ),
    "4_1_1_overall_state": (
        "Answer concisely and include the key supporting quantity or ratio that best explains the state."
    ),
    "4_1_2_side_motion_status": (
        "Use concise natural wording and include the most important motion-count evidence, without "
        "listing individual participants."
    ),
    "4_1_3_scene_summary": (
        "Provide a brief scene-level summary of the current intersection.\n"
        "Keep it short, highlight the main traffic condition, and mention the most notable "
        "current phenomenon."
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
        "Mention the abnormal type and the most important location or event evidence without "
        "listing multiple abnormalities."
    ),
    "4_3_1_intersection_action": (
        "Focus on the recommended action itself and do not explain the full reasoning chain."
    ),
    "4_3_2_side_action": "Keep the guidance local to that approach and avoid broad scene explanation.",
    "4_3_3_lane_action": "Keep the response focused on that lane and do not list alternatives.",
    "4_3_4_object_action": (
        "Focus on what that object should do now, and avoid discussing other participants or "
        "alternative actions."
    ),
}

V6_SUBTEMPLATE_TASK_INSTRUCTIONS = {
    "1_1_1_lane_first_object_type": (
        "Answer briefly with the first relevant object's fine-grained type and its lane or approach phrase only."
    ),
    "1_1_2_front_neighbor_type": (
        "Interpret the front-neighbor relation in the referenced object's own heading frame and answer only with the neighbor type."
    ),
    "1_1_3_approach_vru_exists": (
        "Treat vulnerable road users as pedestrians, bicycles, motorcycles, and golf carts only. "
        "Answer briefly with yes/no and the count when present."
    ),
    "1_1_4_approach_type_count": "Report only the requested participant type and count for that approach.",
    "1_2_1_size_bucket": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["1_2_1_size_bucket"],
    "1_2_2_visibility": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["1_2_2_visibility"],
    "1_3_1_environment": (
        "Use whole-scene visual evidence and answer only with the requested environment attributes."
    ),
    "1_3_2_vehicle_signal_state": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["1_3_2_vehicle_signal_state"],
    "2_1_1_stopline_distance": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_1_1_stopline_distance"],
    "2_1_2_ped_to_far_edge": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_1_2_ped_to_far_edge"],
    "2_1_3_participant_distance": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_1_3_participant_distance"],
    "2_1_4_nearest_vehicle": "Answer only with the nearest vehicle type, its approach, and the distance.",
    "2_2_1_ped_zone": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_2_2_ped_zone"],
    "2_2_2_lane_queue_count": "Return only the queue count for the specified approach lane.",
    "2_2_3_stopline_back_5m_count": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_2_4_stopline_back_5m_count"],
    "2_2_4_longest_queue_lane": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_2_5_longest_queue_lane"],
    "2_2_5_crosswalk_blocking": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["2_2_6_crosswalk_blocking"],
    "3_1_1_current_motion_state": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_1_1_current_motion_state"],
    "3_1_2_vehicle_maneuver": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_1_2_vehicle_maneuver"],
    "3_2_1_waypoints": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_2_3_waypoints"],
    "3_2_2_future_region": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_2_2_future_region"],
    "3_3_1_safe_following": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_3_1_safe_following"],
    "3_3_2_likely_long_queue_lane": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_3_2_likely_long_queue_lane"],
    "3_4_1_pair_conflict": "Focus only on this pair and keep negative answers brief.",
    "3_4_2_nearest_conflict_participant": (
        "Answer only with the participant type plus the requested lane/approach or image-reference evidence, without extra explanation."
    ),
    "3_4_3_primary_risk_subject": (
        "Answer only with the primary risk subject and the dominant risk reason, without extra scene description."
    ),
    "3_4_4_risk_pattern": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["3_4_4_risk_pattern"],
    "4_1_1_overall_state": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_1_1_overall_state"],
    "4_1_2_approach_motion_status": (
        "Use concise natural wording and include the most important motion-count evidence for that approach only."
    ),
    "4_1_3_scene_summary": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_1_3_scene_summary"],
    "4_1_4_heaviest_traffic_approach": (
        "Answer only with the heaviest-traffic approach and its traffic-participant count."
    ),
    "4_2_1_speeding_risk": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_2_1_speeding_risk"],
    "4_2_2_notable_abnormal": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_2_2_notable_abnormal"],
    "4_3_1_intersection_action": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_3_1_intersection_action"],
    "4_3_2_approach_action": "Keep the guidance local to that approach and avoid broad scene explanation.",
    "4_3_3_lane_action": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_3_3_lane_action"],
    "4_3_4_object_action": V5_SUBTEMPLATE_TASK_INSTRUCTIONS["4_3_4_object_action"],
}

VERSIONED_SUBTEMPLATE_TASK_INSTRUCTIONS = {
    "v5": V5_SUBTEMPLATE_TASK_INSTRUCTIONS,
    "v6": V6_SUBTEMPLATE_TASK_INSTRUCTIONS,
}


class CrossPlatformPathUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PureWindowsPath
        return super().find_class(module, name)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return CrossPlatformPathUnpickler(file).load()


def normalize_data_path(raw_path: str) -> str:
    ensure(bool(raw_path), "Encountered empty path while normalizing data path.")
    normalized = str(raw_path).replace("\\", "/")
    if normalized.startswith("/data/"):
        return str((LLM_ROOT / normalized.lstrip("/")).resolve())
    if normalized.startswith("data/"):
        return str((LLM_ROOT / normalized).resolve())
    return str(Path(normalized).resolve()) if Path(normalized).is_absolute() else str((LLM_ROOT / normalized).resolve())


def parse_frame_token(token: str) -> Tuple[int, int]:
    left, right = token.split("-", 1)
    return int(left), int(right)


def view_key(view_name: str) -> str:
    return f"{view_name}_image_path"


def validate_views(views: Sequence[str]) -> Tuple[str, ...]:
    ensure(tuple(views) == DEFAULT_VIEWS, f"TraffiX-Qwen V5 only supports views {DEFAULT_VIEWS}, got {tuple(views)}")
    return tuple(views)


def build_scene_split(
    scene_counts: Mapping[str, int],
    val_scenes: int,
    target_ratio: float,
) -> Tuple[List[str], List[str], int]:
    ranked = sorted(scene_counts.items(), key=lambda item: (-item[1], item[0]))
    ensure(0 < val_scenes < len(ranked), f"val_scenes must be in [1, {len(ranked) - 1}], got {val_scenes}")
    target_qas = round(sum(scene_counts.values()) * target_ratio)
    rank_index = {scene_id: index for index, (scene_id, _) in enumerate(ranked)}
    remaining = list(ranked)
    val_scene_ids: List[str] = []
    current_qas = 0

    for pick_index in range(val_scenes):
        slots_left = val_scenes - pick_index
        if len(remaining) == slots_left:
            for scene_id, count in remaining:
                val_scene_ids.append(scene_id)
                current_qas += count
            remaining.clear()
            break

        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                abs((current_qas + remaining[idx][1]) - target_qas),
                abs(((target_qas - current_qas) / slots_left) - remaining[idx][1]),
                rank_index[remaining[idx][0]],
            ),
        )
        scene_id, count = remaining.pop(best_idx)
        val_scene_ids.append(scene_id)
        current_qas += count

    val_scene_set = set(val_scene_ids)
    train_scene_ids = [scene_id for scene_id, _ in ranked if scene_id not in val_scene_set]
    return train_scene_ids, val_scene_ids, target_qas


def validate_subtemplate_registry(subtemplates: Iterable[str], dataset_version: str) -> None:
    registry = VERSIONED_SUBTEMPLATE_TASK_INSTRUCTIONS[dataset_version]
    missing = sorted({str(subtemplate) for subtemplate in subtemplates if str(subtemplate) not in registry})
    ensure(not missing, f"Missing {dataset_version} prompt instructions for subtemplates: {missing}")


def get_subtemplate_instruction(subtemplate: str, dataset_version: str) -> str:
    registry = VERSIONED_SUBTEMPLATE_TASK_INSTRUCTIONS[dataset_version]
    ensure(subtemplate in registry, f"Missing {dataset_version} prompt instruction for subtemplate={subtemplate}")
    return registry[subtemplate]


def build_system_prompt(views: Sequence[str], temporal_window: int) -> str:
    midpoint = temporal_window // 2
    context_labels = []
    for offset in range(-midpoint, midpoint + 1):
        if offset < 0:
            context_labels.append(f"t{offset}")
        elif offset > 0:
            context_labels.append(f"t+{offset}")
        else:
            context_labels.append("t")
    return (
        "You are an AI assistant specialized in traffic-scene analysis for a four-way intersection.\n\n"
        f"You are given {len(views) * temporal_window} camera images covering {temporal_window} consecutive timesteps. "
        f"For each timestep, the views are ordered as {', '.join(views)}. The timestep labels are "
        f"{', '.join(context_labels)}, and the middle timestep t is the current reference moment.\n\n"
        "Available inputs and grounding rules:\n"
        "- Use only the provided images as evidence.\n"
        "- Use cross-view and cross-time visual evidence conservatively and do not hallucinate unsupported objects, states, or interactions.\n"
        "- Do not infer hidden signal states or invisible objects from behavior alone.\n"
        "- The answer must stay grounded in the visible scene and the referenced object metadata.\n\n"
        "Object reference rules:\n"
        "- Object references use the format <ID,f,X,Y,image_name_1,X1_1,Y1_1,...,image_name_n,X1_n,Y1_n>.\n"
        "- ID is scene-stable within the same scene.\n"
        "- f is the relative time offset in seconds from the first frame of the same scene.\n"
        "- In image-only mode, do not rely on global X,Y coordinates as evidence.\n"
        "- Each repeated triplet image_name_k,X1_k,Y1_k describes one image in which the same object is visible.\n"
        "- image_name_k is written directly as north_image, south_image, east_image, or west_image.\n"
        "- The reference may contain one or multiple image triplets depending on visible views.\n\n"
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
        "- If the task is yes/no, keep negative answers brief."
    )


def build_user_prompt(
    question: str,
    chapter: str,
    section: str,
    subtemplate: str,
    dataset_version: str,
    views: Sequence[str],
    temporal_window: int,
) -> str:
    num_images = len(views) * temporal_window
    image_tokens = "".join(["<image>"] * num_images)
    midpoint = temporal_window // 2
    ordered_pairs = []
    numbered_images = []
    image_index = 1
    for frame_idx in range(temporal_window):
        offset = frame_idx - midpoint
        if offset < 0:
            time_label = f"t{offset}"
        elif offset > 0:
            time_label = f"t+{offset}"
        else:
            time_label = "t"
        for view in views:
            ordered_pairs.append(f"{time_label}:{view}")
            numbered_images.append(f"- image {image_index} = {time_label} {view} view")
            image_index += 1
    return (
        f"{image_tokens}\n"
        "Image-to-direction and time mapping:\n"
        + "\n".join(numbered_images)
        + "\n\n"
        "Task metadata:\n"
        f"- chapter: {chapter}\n"
        f"- section: {section}\n"
        f"- subtemplate: {subtemplate}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Task instructions:\n"
        f"{get_subtemplate_instruction(subtemplate, dataset_version)}\n\n"
        "Use all relevant views and timesteps. The current reference timestep is t, and images are ordered by time then by view: "
        f"{', '.join(ordered_pairs)}."
    )


def load_source_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    qa_pairs = payload.get("qa_pairs") if isinstance(payload, dict) else payload
    ensure(isinstance(qa_pairs, list) and qa_pairs, f"{path} does not contain a non-empty qa_pairs list.")
    return qa_pairs


def normalize_info_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: normalize_info_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_info_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(normalize_info_payload(value) for value in payload)
    if isinstance(payload, pathlib.PurePath):
        return normalize_data_path(str(payload))
    if isinstance(payload, str):
        return normalize_data_path(payload) if payload.startswith(("data/", "/data/")) else payload.replace("\\", "/")
    return payload


def load_infos_by_scene(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw_infos = load_pickle(path)
    infos = raw_infos["infos"] if isinstance(raw_infos, dict) and "infos" in raw_infos else raw_infos
    ensure(isinstance(infos, list) and infos, f"Unexpected info payload in {path}")
    scene_to_infos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for info in infos:
        normalized = normalize_info_payload(info)
        scene_to_infos[str(normalized["scene_id"])].append(normalized)
    for scene_id in scene_to_infos:
        scene_to_infos[scene_id].sort(key=lambda item: (float(item["timestamp"]), str(item["token"])))
    return scene_to_infos


def build_info_lookup(scene_to_infos: Mapping[str, Sequence[Dict[str, Any]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scene_id, infos in scene_to_infos.items():
        for info in infos:
            lookup[(scene_id, str(info["token"]))] = info
    return lookup


def resolve_view_image_path(info: Mapping[str, Any], view: str) -> str:
    cam_name = VIEW_TO_INFO_CAM[view]
    cams = info.get("cams") or {}
    ensure(cam_name in cams, f"Missing {cam_name} for frame_token={info.get('token')}")
    image_path = normalize_data_path(str(cams[cam_name]["image_paths"]))
    ensure(Path(image_path).is_file(), f"Image file not found for frame_token={info.get('token')} view={view}: {image_path}")
    return image_path


def validate_anchor_images(qa_pair: Mapping[str, Any], views: Sequence[str]) -> Dict[str, str]:
    images: Dict[str, str] = {}
    for view in views:
        raw_path = qa_pair.get(view_key(view))
        ensure(raw_path is not None, f"Missing {view_key(view)} for question_id={qa_pair.get('question_id')}")
        image_path = normalize_data_path(str(raw_path))
        ensure(Path(image_path).is_file(), f"Image file not found for question_id={qa_pair.get('question_id')}: {image_path}")
        images[view] = image_path
    return images


def build_context_frame_tokens(
    scene_infos: Sequence[Mapping[str, Any]],
    frame_token: str,
    temporal_window: int,
) -> List[str]:
    ensure(temporal_window % 2 == 1 and temporal_window >= 1, "temporal_window must be a positive odd integer.")
    ordered_tokens = [str(info["token"]) for info in scene_infos]
    ensure(frame_token in ordered_tokens, f"frame_token {frame_token} is missing from the scene info sequence.")
    anchor_idx = ordered_tokens.index(frame_token)
    midpoint = temporal_window // 2
    selected: List[str] = []
    for relative_idx in range(-midpoint, midpoint + 1):
        source_idx = max(0, min(anchor_idx + relative_idx, len(ordered_tokens) - 1))
        selected.append(ordered_tokens[source_idx])
    return selected


def build_image_sequence(
    scene_infos: Sequence[Mapping[str, Any]],
    frame_token: str,
    views: Sequence[str],
    temporal_window: int,
) -> Tuple[List[str], List[str]]:
    context_frame_tokens = build_context_frame_tokens(scene_infos, frame_token, temporal_window)
    scene_info_by_token = {str(info["token"]): info for info in scene_infos}
    image_paths: List[str] = []
    for context_token in context_frame_tokens:
        info = scene_info_by_token[context_token]
        for view in views:
            image_paths.append(resolve_view_image_path(info, view))
    return context_frame_tokens, image_paths


def build_sidecar_rows(
    qa_pairs: Sequence[Mapping[str, Any]],
    scene_to_split: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sidecar_all: List[Dict[str, Any]] = []
    sidecar_train: List[Dict[str, Any]] = []
    sidecar_val: List[Dict[str, Any]] = []
    for qa_pair in qa_pairs:
        split = scene_to_split[str(qa_pair["scene_id"])]
        row = {
            "question_id": str(qa_pair["question_id"]),
            "scene_id": str(qa_pair["scene_id"]),
            "frame_token": str(qa_pair["frame_token"]),
            "chapter": str(qa_pair["chapter"]),
            "section": str(qa_pair["section"]),
            "subtemplate": str(qa_pair["subtemplate"]),
            "question": str(qa_pair["question"]),
            "answer": str(qa_pair["answer"]),
            "structured_targets": qa_pair.get("structured_targets", {}),
            "split": split,
        }
        sidecar_all.append(row)
        if split == "train":
            sidecar_train.append(row)
        else:
            sidecar_val.append(row)
    return sidecar_all, sidecar_train, sidecar_val


def summarize_split_counts(samples: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "samples": len(samples),
        "scenes": len({str(sample["scene_id"]) for sample in samples}),
        "frames": len({str(sample["frame_token"]) for sample in samples}),
    }


def validate_prepared_sample(sample: Mapping[str, Any], temporal_window: int, views: Sequence[str]) -> None:
    ensure(isinstance(sample.get("system"), str) and sample["system"].strip(), f"Sample {sample.get('id')} is missing a non-empty system prompt.")
    ensure("image" in sample and isinstance(sample["image"], list), f"Sample {sample.get('id')} is missing image list.")
    ensure(len(sample["image"]) == temporal_window * len(views), f"Sample {sample.get('id')} does not contain {temporal_window * len(views)} images.")
    ensure("conversations" in sample and len(sample["conversations"]) == 2, f"Sample {sample.get('id')} has invalid conversation payload.")
    ensure(sample["conversations"][0]["from"] == "human", f"Sample {sample.get('id')} missing human turn.")
    ensure(sample["conversations"][1]["from"] == "gpt", f"Sample {sample.get('id')} missing assistant turn.")
    for image_path in sample["image"]:
        ensure(Path(image_path).is_file(), f"Prepared sample {sample.get('id')} references missing image {image_path}")
