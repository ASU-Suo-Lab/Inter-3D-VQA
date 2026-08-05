#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


IMAGE_FIELDS = [
    ("north_image_path", "north"),
    ("south_image_path", "south"),
    ("east_image_path", "east"),
    ("west_image_path", "west"),
]

ROLE_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}

DATASET_COLUMNS = {
    "messages": "messages",
    "system": "system",
    "images": "images",
    "lidar_object_features": "lidar_object_features",
    "lidar_object_geometry_features": "lidar_object_geometry_features",
}
SUPPORTED_TEMPLATE_VERSIONS = {"v5", "v6"}
RAWSCENE_PROMPT_MODES = {"pointcloud_plus_image", "pointcloud_plus_image_detailed_rawscene"}
RAWPATCH_PROMPT_MODES = {"pointcloud_plus_image_detailed_rawpatch"}
INPUTFUSION_PROMPT_MODES = {"pointcloud_plus_image_detailed_rawscene_inputfusion"}
LIDAR_MEMORY_PROMPT_MODES = RAWSCENE_PROMPT_MODES | RAWPATCH_PROMPT_MODES | INPUTFUSION_PROMPT_MODES
MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS = 4
LIDAR_OBJECT_EVIDENCE_SELECTIONS = {
    "question_reference_xy_then_detector_score",
    "template_aware",
}
OBJECT_REFERENCE_PATTERN = re.compile(r"<o\d+,\d+,(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated intersection QA JSON (v5 or v6) into a LlamaFactory sharegpt multimodal dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa-json",
        type=Path,
        default=Path("intersection_qa_pairs_v5.json"),
        help="Path to the generated intersection QA dataset.",
    )
    parser.add_argument(
        "--dataset-version",
        choices=["v5", "v6"],
        default=None,
        help="Explicit dataset version. When provided, it must match metadata.version in the QA JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target LlamaFactory dataset_dir to populate. Defaults to a version-specific directory based on dataset_version / metadata.version.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "image_only",
            "pointcloud_plus_image",
            "pointcloud_plus_image_detailed_rawscene",
            "pointcloud_plus_image_detailed_rawpatch",
            "pointcloud_plus_image_detailed_rawscene_inputfusion",
        ],
        default="image_only",
        help="Prompt metadata variant to export.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        type=str,
        default=None,
        help="Optional explicit dataset prefix for output filenames and dataset_info entries.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio at the scene_id level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic scene-level splitting.",
    )
    parser.add_argument(
        "--lidar-object-dir",
        type=Path,
        default=None,
        help="Optional directory containing offline LiDAR object token files named by frame_token.",
    )
    parser.add_argument(
        "--lidar-object-geometry-dir",
        type=Path,
        default=None,
        help="Optional directory containing offline object-aligned LiDAR geometry feature files named by frame_token.",
    )
    parser.add_argument(
        "--lidar-scene-dir",
        type=Path,
        default=None,
        help="Optional directory containing offline LiDAR scene token files named by frame_token.",
    )
    parser.add_argument(
        "--lidar-object-evidence-max-objects",
        type=int,
        default=MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS,
        help="Maximum number of LiDAR object evidence rows to include in rawscene prompts.",
    )
    parser.add_argument(
        "--lidar-object-evidence-selection",
        choices=sorted(LIDAR_OBJECT_EVIDENCE_SELECTIONS),
        default="question_reference_xy_then_detector_score",
        help="Selection strategy for LiDAR object evidence rows.",
    )
    parser.add_argument(
        "--lidar-input-token",
        type=str,
        default="<|lidar_pad|>",
        help="Special token string used for LiDAR input-token fusion placeholders.",
    )
    parser.add_argument(
        "--lidar-input-token-count",
        type=int,
        default=80,
        help="Number of LiDAR placeholder tokens to insert for input-token fusion.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"QA JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_list(value: object) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def _squared_xy_distance(box: list, ref_xy: tuple[float, float]) -> float | None:
    if not isinstance(box, list) or len(box) < 2:
        return None
    x, y = float(box[0]), float(box[1])
    ref_x, ref_y = ref_xy
    return (x - ref_x) ** 2 + (y - ref_y) ** 2


def _append_unique_index(indices: list[int], idx: int, max_objects: int) -> None:
    if idx not in indices and len(indices) < max_objects:
        indices.append(idx)


def _select_reference_matched_indices(
    boxes: list,
    object_reference_xy: tuple[tuple[float, float], ...],
    max_objects: int,
) -> list[int]:
    selected_indices: list[int] = []
    for ref_xy in object_reference_xy:
        best_idx = None
        best_distance = None
        for idx, box in enumerate(boxes):
            distance = _squared_xy_distance(box, ref_xy)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_idx = idx
                best_distance = distance
        if best_idx is not None:
            _append_unique_index(selected_indices, best_idx, max_objects)
    return selected_indices


def _rank_by_score(boxes: list, scores: list) -> list[int]:
    return sorted(
        range(len(boxes)),
        key=lambda idx: (-(float(scores[idx]) if idx < len(scores) else 0.0), idx),
    )


def _select_lidar_object_evidence_indices(
    boxes: list,
    scores: list,
    max_objects: int,
    object_reference_xy: tuple[tuple[float, float], ...],
    subtemplate: str | None,
    selection: str,
) -> tuple[list[int], int]:
    selected_indices = _select_reference_matched_indices(boxes, object_reference_xy, max_objects)
    reference_selected_count = len(selected_indices)

    if selection == "template_aware":
        value = (subtemplate or "").lower()
        relation_template = any(
            marker in value
            for marker in (
                "nearest",
                "neighbor",
                "conflict",
                "following",
                "distance",
                "far_edge",
                "stopline",
                "waypoint",
            )
        )
        if relation_template and object_reference_xy:
            for ref_xy in object_reference_xy:
                nearby_indices = sorted(
                    range(len(boxes)),
                    key=lambda idx: (_squared_xy_distance(boxes[idx], ref_xy) is None,
                                     _squared_xy_distance(boxes[idx], ref_xy) or 0.0,
                                     idx),
                )
                for idx in nearby_indices:
                    _append_unique_index(selected_indices, idx, max_objects)
                    if len(selected_indices) >= max_objects:
                        break
                if len(selected_indices) >= max_objects:
                    break

    for idx in _rank_by_score(boxes, scores):
        _append_unique_index(selected_indices, idx, max_objects)
        if len(selected_indices) >= max_objects:
            break

    return selected_indices, reference_selected_count


def _object_label_for_idx(label_names: list, labels: list, idx: int) -> str:
    label = label_names[idx] if idx < len(label_names) and label_names[idx] else None
    if label is None and idx < len(labels):
        label = f"class_{int(labels[idx])}"
    return label or "object"


def _format_reference_geometry_lines(
    boxes: list,
    object_reference_xy: tuple[tuple[float, float], ...],
) -> list[str]:
    if len(object_reference_xy) < 2:
        return []
    ref_a = object_reference_xy[0]
    ref_b = object_reference_xy[1]
    distance = ((ref_a[0] - ref_b[0]) ** 2 + (ref_a[1] - ref_b[1]) ** 2) ** 0.5
    lines = [f"- referenced_distance_m: {distance:.1f}."]
    return lines


@lru_cache(maxsize=8192)
def _build_lidar_object_evidence_text_for_path(
    object_feature_path: str,
    max_objects: int,
    object_reference_xy: tuple[tuple[float, float], ...],
    subtemplate: str | None,
    selection: str,
) -> str | None:
    path = Path(object_feature_path)
    if path.suffix.lower() != ".pt":
        return None

    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    boxes = _to_list(payload.get("pred_boxes"))
    scores = _to_list(payload.get("pred_scores"))
    label_names = payload.get("pred_label_names")
    labels = _to_list(payload.get("pred_labels"))
    if not boxes:
        return "LiDAR object evidence text:\n- no retained LiDAR object tokens above the detector threshold."

    if not isinstance(label_names, list):
        label_names = []

    selected_indices, reference_selected_count = _select_lidar_object_evidence_indices(
        boxes,
        scores,
        max_objects,
        object_reference_xy,
        subtemplate,
        selection,
    )

    pre_filter_count = payload.get("pre_filter_object_count")
    threshold = payload.get("object_score_threshold")
    summary_parts = [f"listed {len(selected_indices)} object tokens"]
    if object_reference_xy:
        summary_parts.append(f"prioritized {reference_selected_count} question-referenced object coordinates")
    if isinstance(pre_filter_count, int):
        summary_parts.append(f"from {pre_filter_count} detector candidates")
    if isinstance(threshold, (int, float)):
        summary_parts.append(f"score_threshold={float(threshold):.2f}")

    lines = [
        "LiDAR object evidence text (detector-token summary; token_index is not the QA object ID):",
        f"- summary: {', '.join(summary_parts)}.",
    ]
    lines.extend(_format_reference_geometry_lines(boxes, object_reference_xy))
    for idx in selected_indices:
        box = boxes[idx]
        if not isinstance(box, list) or len(box) < 7:
            continue
        label = _object_label_for_idx(label_names, labels, idx)
        score = float(scores[idx]) if idx < len(scores) else None
        x, y, z, dx, dy, dz, heading = [float(value) for value in box[:7]]
        line = (
            f"- token_{idx}: type={label}, center_xy=({x:.1f},{y:.1f}) m, z={z:.1f} m, "
            f"size=({dx:.1f},{dy:.1f},{dz:.1f}) m, heading={heading:.2f} rad"
        )
        if len(box) >= 9:
            vx, vy = float(box[7]), float(box[8])
            speed = (vx ** 2 + vy ** 2) ** 0.5
            line += f", velocity_xy=({vx:.1f},{vy:.1f}) m/s, speed={speed:.1f} m/s"
        if object_reference_xy:
            nearest_ref_idx = None
            nearest_ref_distance = None
            for ref_idx, ref_xy in enumerate(object_reference_xy):
                distance = ((x - ref_xy[0]) ** 2 + (y - ref_xy[1]) ** 2) ** 0.5
                if nearest_ref_distance is None or distance < nearest_ref_distance:
                    nearest_ref_idx = ref_idx
                    nearest_ref_distance = distance
            if nearest_ref_idx is not None and nearest_ref_distance is not None:
                ref_x, ref_y = object_reference_xy[nearest_ref_idx]
                line += (
                    f", nearest_question_ref=o{nearest_ref_idx + 1}, "
                    f"relative_xy=({x - ref_x:.1f},{y - ref_y:.1f}) m, "
                    f"distance_to_ref={nearest_ref_distance:.1f} m"
                )
        if score is not None:
            line += f", score={score:.2f}"
        line += "."
        lines.append(line)

    return "\n".join(lines)


def build_lidar_object_evidence_text(
    lidar_object_features: list[str] | None,
    max_objects: int = MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS,
    question: str | None = None,
    subtemplate: str | None = None,
    selection: str = "question_reference_xy_then_detector_score",
) -> str | None:
    if not lidar_object_features:
        return None
    if len(lidar_object_features) != 1:
        return None
    object_reference_xy = tuple(
        (float(match.group(1)), float(match.group(2))) for match in OBJECT_REFERENCE_PATTERN.finditer(question or "")
    )
    return _build_lidar_object_evidence_text_for_path(
        str(lidar_object_features[0]),
        max_objects,
        object_reference_xy,
        subtemplate,
        selection,
    )


def build_system_prompt(
    prompt_metadata: dict,
    prompt_mode: str,
    has_scene_evidence: bool,
    has_object_evidence: bool,
    raw_scene_mode: bool,
) -> str:
    if prompt_mode == "pointcloud_plus_image":
        scene_evidence = "LiDAR-derived scene memory" if raw_scene_mode else "LiDAR scene tokens"
        system_prompt = f"You analyze a four-way intersection using four camera images and {scene_evidence}."
        system_prompt += " Use LiDAR-derived scene evidence mainly for global 3D layout, distance cues, lane geometry, and overall spatial structure."
        if has_object_evidence:
            system_prompt += " Use LiDAR-derived object evidence mainly for object-level geometric cues, object relations, and local risk reasoning."
        system_prompt += " Use the images mainly for visible appearance and camera-specific evidence. Treat the LiDAR inputs as derived memory rather than direct raw point-cloud access. Answer briefly and only from supported evidence."
        return system_prompt
    if prompt_mode in {"pointcloud_plus_image_detailed_rawscene", "pointcloud_plus_image_detailed_rawscene_inputfusion"}:
        system_prompt = prompt_metadata["system_prompt"]
        evidence_lines = [
            "- LiDAR-derived object memory: lidar_object_tokens.",
            "- LiDAR-derived object-local geometry memory: lidar_object_geometry_tokens.",
        ]
        if has_scene_evidence:
            evidence_lines.append("- LiDAR-derived scene token memory: lidar_scene_tokens.")
        system_prompt = system_prompt.replace(
            "- global intersection point cloud: point_cloud",
            "\n".join(evidence_lines),
        )
        system_prompt = system_prompt.replace(
            "Use both the images and the point cloud conservatively as evidence.",
            "Use both the images and the LiDAR-derived memory conservatively as evidence.",
        )
        if has_scene_evidence:
            geometry_guidance = (
                "For 3D geometry, spatial layout, global position, and distance inference, prefer the LiDAR-derived "
                "object, geometry, and scene-token evidence when geometric support is available, with the images as "
                "auxiliary evidence."
            )
        else:
            geometry_guidance = (
                "For 3D geometry, spatial layout, global position, and distance inference, prefer the LiDAR-derived "
                "object and geometry evidence together with the referenced global X,Y coordinates when geometric "
                "support is available, with the images as auxiliary evidence."
            )
        system_prompt = system_prompt.replace(
            "For 3D geometry, spatial layout, global position, and distance inference, primarily rely on the point cloud, with the images as auxiliary evidence.",
            geometry_guidance,
        )
        system_prompt = system_prompt.replace(
            "For visible appearance, visible signal state, local occlusion, and cross-view scene context, primarily rely on the images, with the point cloud as auxiliary evidence.",
            "For visible appearance, visible signal state, local occlusion, and cross-view scene context, primarily rely on the images, with the LiDAR-derived memory as auxiliary evidence.",
        )
        system_prompt = system_prompt.replace("Point-cloud reference rules:", "LiDAR geometric reference rules:")
        system_prompt = system_prompt.replace(
            "X,Y are global point-cloud planar coordinates in meters and come from 3D scene geometry.",
            "X,Y are LiDAR-derived global planar coordinates in meters and come from scene geometry."
        )
        system_prompt = system_prompt.replace(
            "The point-cloud coordinate system is global and shared across frames within the same scene.",
            "The LiDAR geometric coordinate system is global and shared across frames within the same scene.",
        )
        system_prompt = system_prompt.replace(
            "Use the point cloud and the provided object references for geometry-aware reasoning.",
            "Use the provided LiDAR-derived evidence and object references for geometry-aware reasoning.",
        )
        if not has_scene_evidence:
            system_prompt = system_prompt.replace(
                "- For 3D geometry, spatial layout, global position, and distance inference, primarily rely on the LiDAR object, geometry, and scene-token evidence, with the images as auxiliary evidence.",
                geometry_guidance,
            )
        return system_prompt
    if prompt_mode == "pointcloud_plus_image_detailed_rawpatch":
        system_prompt = prompt_metadata["system_prompt"]
        evidence_lines = ["- LiDAR-derived local patch memory: lidar_object_tokens."]
        if has_scene_evidence:
            evidence_lines.append("- LiDAR-derived scene patch memory: lidar_scene_tokens.")
        system_prompt = system_prompt.replace(
            "- global intersection point cloud: point_cloud",
            "\n".join(evidence_lines),
        )
        system_prompt = system_prompt.replace(
            "Use both the images and the point cloud conservatively as evidence.",
            "Use both the images and the LiDAR-derived raw patch memory conservatively as evidence.",
        )
        if has_scene_evidence:
            geometry_guidance = (
                "For 3D geometry, spatial layout, global position, and distance inference, prefer the LiDAR-derived "
                "local patch memory and scene patch memory when geometric support is available, with the images as "
                "auxiliary evidence."
            )
        else:
            geometry_guidance = (
                "For 3D geometry, spatial layout, global position, and distance inference, prefer the LiDAR-derived "
                "local patch memory together with the referenced global X,Y coordinates when geometric support is "
                "available, with the images as auxiliary evidence."
            )
        system_prompt = system_prompt.replace(
            "For 3D geometry, spatial layout, global position, and distance inference, primarily rely on the point cloud, with the images as auxiliary evidence.",
            geometry_guidance,
        )
        system_prompt = system_prompt.replace(
            "For visible appearance, visible signal state, local occlusion, and cross-view scene context, primarily rely on the images, with the point cloud as auxiliary evidence.",
            "For visible appearance, visible signal state, local occlusion, and cross-view scene context, primarily rely on the images, with the LiDAR-derived raw patch memory as auxiliary evidence.",
        )
        system_prompt = system_prompt.replace("Point-cloud reference rules:", "LiDAR geometric reference rules:")
        system_prompt = system_prompt.replace(
            "X,Y are global point-cloud planar coordinates in meters and come from 3D scene geometry.",
            "X,Y are LiDAR-derived global planar coordinates in meters and come from scene geometry."
        )
        system_prompt = system_prompt.replace(
            "The point-cloud coordinate system is global and shared across frames within the same scene.",
            "The LiDAR geometric coordinate system is global and shared across frames within the same scene.",
        )
        system_prompt = system_prompt.replace(
            "Use the point cloud and the provided object references for geometry-aware reasoning.",
            "Use the provided LiDAR-derived raw patch memory and object references for geometry-aware reasoning.",
        )
        return system_prompt

    return prompt_metadata["system_prompt"]


def build_user_prompt(
    qa: dict,
    prompt_metadata: dict,
    prompt_mode: str,
    has_scene_evidence: bool,
    has_object_evidence: bool,
    raw_scene_mode: bool,
    lidar_object_features: list[str] | None = None,
    lidar_object_evidence_max_objects: int = MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS,
    lidar_object_evidence_selection: str = "question_reference_xy_then_detector_score",
) -> str:
    subtemplate = qa["subtemplate"]
    patch = prompt_metadata["subtemplate_patches"][subtemplate]
    if prompt_mode == "pointcloud_plus_image":
        scene_evidence_text = "lidar_scene_tokens" if raw_scene_mode else "lidar_scene_tokens"
        prompt = (
            "Image-to-direction mapping:\n"
            "- image 1 = north view\n"
            "- image 2 = south view\n"
            "- image 3 = east view\n"
            "- image 4 = west view\n"
            f"- available LiDAR scene memory: {scene_evidence_text}\n"
        )
        if has_object_evidence:
            prompt += "- available LiDAR object memory: lidar_object_tokens\n"
        prompt += "\n"
        prompt += (
            f"Subtemplate: {subtemplate}\n"
            f"Question: {qa['question']}\n"
            f"Instruction: {patch}"
        )
        return prompt
    if prompt_mode in {"pointcloud_plus_image_detailed_rawscene", "pointcloud_plus_image_detailed_rawscene_inputfusion"}:
        prompt = (
            "Image-to-direction mapping:\n"
            "- image 1 = north view\n"
            "- image 2 = south view\n"
            "- image 3 = east view\n"
            "- image 4 = west view\n"
        )
        if has_scene_evidence:
            prompt += "- available LiDAR scene memory: lidar_scene_tokens\n"
        if has_object_evidence:
            prompt += "- available LiDAR object memory: lidar_object_tokens\n"
            prompt += "- available LiDAR geometry memory: lidar_object_geometry_tokens\n"
        prompt += "\n"
        prompt += (
            f"Subtemplate: {subtemplate}\n"
            f"Question: {qa['question']}\n"
        )
        object_evidence_text = (
            build_lidar_object_evidence_text(
                lidar_object_features,
                max_objects=lidar_object_evidence_max_objects,
                question=qa["question"],
                subtemplate=subtemplate,
                selection=lidar_object_evidence_selection,
            )
            if has_object_evidence
            else None
        )
        if object_evidence_text:
            prompt += "\n" + object_evidence_text + "\n"
        prompt += f"\nInstruction: {patch}"
        return prompt
    if prompt_mode == "pointcloud_plus_image_detailed_rawpatch":
        prompt = (
            "Image-to-direction mapping:\n"
            "- image 1 = north view\n"
            "- image 2 = south view\n"
            "- image 3 = east view\n"
            "- image 4 = west view\n"
        )
        if has_scene_evidence:
            prompt += "- available LiDAR scene memory: lidar_scene_tokens\n"
        if has_object_evidence:
            prompt += "- available LiDAR object memory: lidar_object_tokens\n"
        prompt += "\n"
        prompt += (
            f"Subtemplate: {subtemplate}\n"
            f"Question: {qa['question']}\n"
            f"Instruction: {patch}"
        )
        return prompt

    chapter = qa.get("chapter", qa.get("category", "unknown"))
    section = qa.get("section", qa.get("scope", "unknown"))
    return prompt_metadata["user_prompt_template"].format(
        chapter=chapter,
        section=section,
        subtemplate=subtemplate,
        question=qa["question"],
        task_rule_block=patch,
        point_cloud="point_cloud",
    )


def validate_rawscene_user_prompt_layout(
    user_prompt: str,
    question_id: object,
    lidar_object_evidence_max_objects: int = MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS,
) -> None:
    subtemplate_idx = user_prompt.find("Subtemplate:")
    question_idx = user_prompt.find("Question:")
    instruction_idx = user_prompt.find("Instruction:")
    require(
        subtemplate_idx >= 0 and question_idx >= 0 and instruction_idx >= 0,
        f"Rawscene prompt is missing task markers for question_id={question_id}",
    )
    require(
        subtemplate_idx < question_idx < instruction_idx,
        f"Rawscene task markers are misordered for question_id={question_id}",
    )

    evidence_idx = user_prompt.find("LiDAR object evidence text")
    if evidence_idx >= 0:
        require(
            question_idx < evidence_idx < instruction_idx,
            f"LiDAR object evidence must be between Question and Instruction for question_id={question_id}",
        )
        evidence_token_lines = [
            line for line in user_prompt[evidence_idx:instruction_idx].splitlines() if line.startswith("- token_")
        ]
        require(
            len(evidence_token_lines) <= lidar_object_evidence_max_objects,
            f"LiDAR object evidence has too many listed tokens for question_id={question_id}: "
            f"{len(evidence_token_lines)} > {lidar_object_evidence_max_objects}",
        )


def normalize_output_dir(path: Path) -> Path:
    return path.resolve()


def detect_template_version(metadata: dict, cli_version: str | None = None) -> str:
    require(isinstance(metadata, dict), "metadata must be a dict")
    version = metadata.get("version")
    require(isinstance(version, str), "metadata.version is required")
    version = version.strip().lower()
    require(version in SUPPORTED_TEMPLATE_VERSIONS, f"Unsupported metadata.version: {version}")
    if cli_version is None:
        return version
    normalized_cli_version = cli_version.strip().lower()
    require(
        normalized_cli_version in SUPPORTED_TEMPLATE_VERSIONS,
        f"Unsupported --dataset-version: {normalized_cli_version}",
    )
    require(
        normalized_cli_version == version,
        f"--dataset-version {normalized_cli_version} does not match metadata.version {version}",
    )
    return normalized_cli_version


def dataset_prefix_for_version(
    template_version: str,
    prompt_mode: str,
    lidar_object_dir: Path | None = None,
    lidar_object_geometry_dir: Path | None = None,
    dataset_name_prefix: str | None = None,
) -> str:
    if dataset_name_prefix:
        return dataset_name_prefix
    if prompt_mode in (RAWSCENE_PROMPT_MODES | INPUTFUSION_PROMPT_MODES):
        if lidar_object_dir is not None and lidar_object_geometry_dir is not None:
            if prompt_mode in INPUTFUSION_PROMPT_MODES:
                return (
                    "intersection_vqa_lidar_rawscene_objroute_inputfusion"
                    if template_version == "v5"
                    else "intersection_vqa_v6_lidar_rawscene_objroute_inputfusion"
                )
            return (
                "intersection_vqa_lidar_rawscene_objroute"
                if template_version == "v5"
                else "intersection_vqa_v6_lidar_rawscene_objroute"
            )
        raise ValueError("pointcloud_plus_image export now requires rawscene LiDAR object and geometry directories.")
    return "intersection_vqa" if template_version == "v5" else "intersection_vqa_v6"


def default_output_dir_for_version(
    template_version: str,
    prompt_mode: str,
    lidar_object_dir: Path | None = None,
    lidar_object_geometry_dir: Path | None = None,
    dataset_name_prefix: str | None = None,
) -> Path:
    return Path("LlamaFactory/data") / dataset_prefix_for_version(
        template_version,
        prompt_mode,
        lidar_object_dir=lidar_object_dir,
        lidar_object_geometry_dir=lidar_object_geometry_dir,
        dataset_name_prefix=dataset_name_prefix,
    )


def resolve_media_path(media_path: Path, qa_json_dir: Path) -> Path:
    if media_path.is_absolute():
        return media_path
    direct = (qa_json_dir / media_path).resolve()
    if direct.exists():
        return direct
    parts = list(media_path.parts)
    if len(parts) >= 2 and parts[0] == "data" and parts[1].startswith("rosbag"):
        suffix = Path(*parts[1:])
        repo_data = (qa_json_dir / "data").resolve()
        candidates = [repo_data / suffix]
        if media_path.suffix.lower() == ".bin":
            candidates.append(repo_data / "lidar" / suffix)
        else:
            candidates.append(repo_data / "images" / suffix)
            candidates.append(repo_data / "202_scenes" / suffix)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return direct


def validate_metadata(metadata: dict, prompt_mode: str) -> dict:
    require("prompt_metadata" in metadata, "metadata.prompt_metadata is required")
    prompt_bundle = metadata["prompt_metadata"]
    prompt_bundle_key = "pointcloud_plus_image" if prompt_mode in LIDAR_MEMORY_PROMPT_MODES else prompt_mode
    if prompt_bundle_key in prompt_bundle:
        require(isinstance(prompt_bundle[prompt_bundle_key], dict), f"prompt_metadata.{prompt_bundle_key} must be a dict")
        prompt_metadata = prompt_bundle[prompt_bundle_key]
    else:
        prompt_metadata = prompt_bundle
    required_keys = {"system_prompt", "user_prompt_template", "subtemplate_patches", "subtemplate_patch_style"}
    missing = sorted(required_keys - set(prompt_metadata.keys()))
    require(not missing, f"prompt_metadata is missing required keys: {missing}")
    require(
        prompt_metadata["subtemplate_patch_style"] == "simple",
        "This converter currently expects simple prompt metadata. Regenerate the QA JSON in simple mode first.",
    )
    return prompt_metadata


def collect_images(qa: dict, qa_json_dir: Path) -> list[str]:
    image_paths = []
    for field_name, _view_name in IMAGE_FIELDS:
        require(field_name in qa, f"Missing image field {field_name} in question_id={qa.get('question_id')}")
        image_path = resolve_media_path(Path(qa[field_name]), qa_json_dir)
        require(image_path.is_file(), f"Image file not found for question_id={qa.get('question_id')}: {image_path}")
        image_paths.append(str(image_path))
    return image_paths


def collect_lidar_object_features(qa: dict, lidar_object_dir: Path | None) -> list[str] | None:
    if lidar_object_dir is None:
        return None

    frame_token = qa.get("frame_token")
    require(frame_token, f"Missing frame_token for LiDAR object lookup in question_id={qa.get('question_id')}")
    for suffix in (".object.pt", ".object.npy", ".object.npz", "_object.pt", "_object.npy", "_object.npz"):
        candidate = lidar_object_dir / f"{frame_token}{suffix}"
        if candidate.is_file():
            return [str(candidate.resolve())]

    raise ValueError(f"LiDAR object feature file not found for frame_token={frame_token} under {lidar_object_dir}")


def collect_lidar_object_geometry_features(qa: dict, lidar_object_geometry_dir: Path | None) -> list[str] | None:
    if lidar_object_geometry_dir is None:
        return None

    frame_token = qa.get("frame_token")
    require(frame_token, f"Missing frame_token for LiDAR object geometry lookup in question_id={qa.get('question_id')}")
    for suffix in (
        ".object_geometry.pt",
        ".object_geometry.npy",
        ".object_geometry.npz",
        "_object_geometry.pt",
        "_object_geometry.npy",
        "_object_geometry.npz",
    ):
        candidate = lidar_object_geometry_dir / f"{frame_token}{suffix}"
        if candidate.is_file():
            return [str(candidate.resolve())]

    raise ValueError(
        f"LiDAR object geometry feature file not found for frame_token={frame_token} under {lidar_object_geometry_dir}"
    )


def collect_lidar_scene_features(qa: dict, lidar_scene_dir: Path | None) -> list[str] | None:
    if lidar_scene_dir is None:
        return None

    frame_token = qa.get("frame_token")
    require(frame_token, f"Missing frame_token for LiDAR scene lookup in question_id={qa.get('question_id')}")
    for suffix in (".scene.pt", ".scene.npy", ".scene.npz", "_scene.pt", "_scene.npy", "_scene.npz"):
        candidate = lidar_scene_dir / f"{frame_token}{suffix}"
        if candidate.is_file():
            return [str(candidate.resolve())]

    raise ValueError(f"LiDAR scene feature file not found for frame_token={frame_token} under {lidar_scene_dir}")


def split_scenes(scene_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    require(0.0 < train_ratio < 1.0, f"train_ratio must be between 0 and 1, got {train_ratio}")
    shuffled = sorted(scene_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * (1.0 - train_ratio))))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    require(val_count > 0, "Need at least one validation scene after splitting")
    val_scenes = set(shuffled[:val_count])
    train_scenes = set(shuffled[val_count:])
    require(train_scenes, "Need at least one training scene after splitting")
    require(not (train_scenes & val_scenes), "train/val scene overlap detected")
    return train_scenes, val_scenes


def is_trainable_qa(qa: dict) -> bool:
    answer = qa.get("answer")
    return (not qa.get("placeholder", False)) and answer not in (None, "")


def make_train_record(
    qa: dict,
    prompt_metadata: dict,
    prompt_mode: str,
    qa_json_dir: Path,
    lidar_object_dir: Path | None,
    lidar_object_geometry_dir: Path | None,
    lidar_scene_dir: Path | None,
    lidar_object_evidence_max_objects: int = MAX_LIDAR_OBJECT_EVIDENCE_OBJECTS,
    lidar_object_evidence_selection: str = "question_reference_xy_then_detector_score",
    lidar_input_token: str = "<|lidar_pad|>",
    lidar_input_token_count: int = 80,
) -> dict:
    require(is_trainable_qa(qa), f"Non-trainable QA reached train export: question_id={qa.get('question_id')}")
    raw_scene_mode = prompt_mode in LIDAR_MEMORY_PROMPT_MODES
    lidar_scene_features = collect_lidar_scene_features(qa, lidar_scene_dir)
    lidar_object_features = collect_lidar_object_features(qa, lidar_object_dir)
    lidar_object_geometry_features = collect_lidar_object_geometry_features(qa, lidar_object_geometry_dir)
    has_scene_evidence = lidar_scene_features is not None
    has_object_evidence = lidar_object_features is not None
    system_prompt = build_system_prompt(
        prompt_metadata,
        prompt_mode,
        has_scene_evidence=has_scene_evidence,
        has_object_evidence=has_object_evidence,
        raw_scene_mode=raw_scene_mode,
    )
    user_prompt = build_user_prompt(
        qa,
        prompt_metadata,
        prompt_mode,
        has_scene_evidence=has_scene_evidence,
        has_object_evidence=has_object_evidence,
        raw_scene_mode=raw_scene_mode,
        lidar_object_features=lidar_object_features,
        lidar_object_evidence_max_objects=lidar_object_evidence_max_objects,
        lidar_object_evidence_selection=lidar_object_evidence_selection,
    )
    if raw_scene_mode:
        validate_rawscene_user_prompt_layout(
            user_prompt,
            qa.get("question_id"),
            lidar_object_evidence_max_objects=lidar_object_evidence_max_objects,
        )
    user_content = "<image><image><image><image>\n" + user_prompt
    if prompt_mode in INPUTFUSION_PROMPT_MODES:
        require(lidar_input_token_count > 0, "lidar_input_token_count must be positive for input-token fusion.")
        lidar_token_line = " ".join([lidar_input_token] * lidar_input_token_count)
        user_content = "<image><image><image><image>\n" + lidar_token_line + "\n" + user_prompt

    record = {
        "system": system_prompt,
        "scene_id": qa["scene_id"],
        "frame_token": qa["frame_token"],
        "messages": [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": qa["answer"],
            },
        ],
        "images": collect_images(qa, qa_json_dir),
        "lidar_object_features": lidar_object_features,
        "lidar_object_geometry_features": lidar_object_geometry_features,
    }
    if lidar_scene_features is not None:
        record["lidar_scene_features"] = lidar_scene_features
    return record


def make_sidecar_record(qa: dict, split: str, qa_json_dir: Path, template_version: str) -> dict:
    return {
        "question_id": qa["question_id"],
        "scene_id": qa["scene_id"],
        "frame_token": qa["frame_token"],
        "chapter": qa["chapter"],
        "section": qa["section"],
        "subtemplate": qa["subtemplate"],
        "question": qa["question"],
        "answer": qa["answer"],
        "structured_targets": qa["structured_targets"],
        "images": collect_images(qa, qa_json_dir),
        "split": split,
        "template_version": template_version,
    }


def make_dataset_info(
    dataset_prefix: str,
    include_lidar_scene_features: bool = False,
) -> dict:
    dataset_columns = dict(DATASET_COLUMNS)
    if include_lidar_scene_features:
        dataset_columns["lidar_scene_features"] = "lidar_scene_features"
    return {
        f"{dataset_prefix}_train": {
            "file_name": f"{dataset_prefix}_train.jsonl",
            "formatting": "sharegpt",
            "columns": dataset_columns,
            "tags": ROLE_TAGS,
        },
        f"{dataset_prefix}_val": {
            "file_name": f"{dataset_prefix}_val.jsonl",
            "formatting": "sharegpt",
            "columns": dataset_columns,
            "tags": ROLE_TAGS,
        },
    }


def main() -> None:
    args = parse_args()
    require(
        args.lidar_object_evidence_max_objects > 0,
        f"--lidar-object-evidence-max-objects must be positive, got {args.lidar_object_evidence_max_objects}",
    )
    data = read_json(args.qa_json)
    qa_json_dir = args.qa_json.resolve().parent
    lidar_object_dir = args.lidar_object_dir.resolve() if args.lidar_object_dir is not None else None
    lidar_object_geometry_dir = (
        args.lidar_object_geometry_dir.resolve() if args.lidar_object_geometry_dir is not None else None
    )
    lidar_scene_dir = args.lidar_scene_dir.resolve() if args.lidar_scene_dir is not None else None
    require(
        lidar_object_geometry_dir is None or lidar_object_dir is not None,
        "--lidar-object-geometry-dir requires --lidar-object-dir so object-local geometry stays paired with object features.",
    )
    if lidar_object_dir is not None:
        require(lidar_object_dir.is_dir(), f"LiDAR object feature directory not found: {lidar_object_dir}")
    if lidar_object_geometry_dir is not None:
        require(
            lidar_object_geometry_dir.is_dir(),
            f"LiDAR object geometry feature directory not found: {lidar_object_geometry_dir}",
        )
    if lidar_scene_dir is not None:
        require(lidar_scene_dir.is_dir(), f"LiDAR scene feature directory not found: {lidar_scene_dir}")
    metadata = data["metadata"]
    prompt_metadata = validate_metadata(metadata, args.prompt_mode)
    template_version = detect_template_version(metadata, args.dataset_version)
    dataset_prefix = dataset_prefix_for_version(
        template_version,
        args.prompt_mode,
        lidar_object_dir=lidar_object_dir,
        lidar_object_geometry_dir=lidar_object_geometry_dir,
        dataset_name_prefix=args.dataset_name_prefix,
    )
    output_dir = normalize_output_dir(
        args.output_dir
        or default_output_dir_for_version(
            template_version,
            args.prompt_mode,
            lidar_object_dir=lidar_object_dir,
            lidar_object_geometry_dir=lidar_object_geometry_dir,
            dataset_name_prefix=args.dataset_name_prefix,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    qas = data["qa_pairs"]
    require(qas, "qa_pairs is empty")
    source_scene_ids = []
    seen_source_scenes = set()
    trainable_qas = []
    filtered_template_counts: Counter[str] = Counter()
    placeholder_count = 0
    empty_answer_count = 0
    for qa in qas:
        for required_key in [
            "question_id",
            "scene_id",
            "frame_token",
            "chapter",
            "section",
            "subtemplate",
            "question",
            "answer",
            "structured_targets",
        ]:
            require(required_key in qa, f"Missing {required_key} in QA sample")
        if qa["scene_id"] not in seen_source_scenes:
            seen_source_scenes.add(qa["scene_id"])
            source_scene_ids.append(qa["scene_id"])
        if qa.get("placeholder", False):
            placeholder_count += 1
        if qa.get("answer") in (None, ""):
            empty_answer_count += 1
        if is_trainable_qa(qa):
            trainable_qas.append(qa)
        else:
            filtered_template_counts[qa["subtemplate"]] += 1

    require(trainable_qas, "No trainable QA samples remain after filtering placeholders / empty answers")

    trainable_scene_ids = []
    seen_trainable_scenes = set()
    for qa in trainable_qas:
        if qa["scene_id"] not in seen_trainable_scenes:
            seen_trainable_scenes.add(qa["scene_id"])
            trainable_scene_ids.append(qa["scene_id"])

    train_scenes, val_scenes = split_scenes(trainable_scene_ids, args.train_ratio, args.seed)
    train_rows = []
    val_rows = []
    sidecar_rows = []
    for qa in trainable_qas:
        split = "train" if qa["scene_id"] in train_scenes else "val"
        record = make_train_record(
            qa,
            prompt_metadata,
            args.prompt_mode,
            qa_json_dir,
            lidar_object_dir,
            lidar_object_geometry_dir,
            lidar_scene_dir,
            lidar_object_evidence_max_objects=args.lidar_object_evidence_max_objects,
            lidar_object_evidence_selection=args.lidar_object_evidence_selection,
            lidar_input_token=args.lidar_input_token,
            lidar_input_token_count=args.lidar_input_token_count,
        )
        if split == "train":
            train_rows.append(record)
        else:
            val_rows.append(record)
        sidecar_rows.append(make_sidecar_record(qa, split, qa_json_dir, template_version))

    require(train_rows, "No training rows were generated")
    require(val_rows, "No validation rows were generated")

    train_file = f"{dataset_prefix}_train.jsonl"
    val_file = f"{dataset_prefix}_val.jsonl"
    sidecar_file = f"{dataset_prefix}_eval_sidecar.jsonl"

    write_jsonl(output_dir / train_file, train_rows)
    write_jsonl(output_dir / val_file, val_rows)
    write_jsonl(output_dir / sidecar_file, sidecar_rows)
    write_json(
        output_dir / "dataset_info.json",
        make_dataset_info(
            dataset_prefix,
            include_lidar_scene_features=lidar_scene_dir is not None,
        ),
    )
    write_json(
        output_dir / "split_summary.json",
        {
            "source_qa_json": str(args.qa_json.resolve()),
            "template_version": template_version,
            "dataset_prefix": dataset_prefix,
            "dataset_files": {
                "train": train_file,
                "val": val_file,
                "eval_sidecar": sidecar_file,
            },
            "prompt_mode": args.prompt_mode,
            "lidar_input_token": args.lidar_input_token if args.prompt_mode in INPUTFUSION_PROMPT_MODES else None,
            "lidar_input_token_count": args.lidar_input_token_count if args.prompt_mode in INPUTFUSION_PROMPT_MODES else None,
            "lidar_object_evidence_max_objects": args.lidar_object_evidence_max_objects,
            "lidar_object_evidence_selection": args.lidar_object_evidence_selection,
            "lidar_object_dir": str(lidar_object_dir) if lidar_object_dir is not None else None,
            "lidar_object_geometry_dir": str(lidar_object_geometry_dir) if lidar_object_geometry_dir is not None else None,
            "lidar_scene_dir": str(lidar_scene_dir) if lidar_scene_dir is not None else None,
            "prompt_style": prompt_metadata["subtemplate_patch_style"],
            "image_order": [view_name for _field_name, view_name in IMAGE_FIELDS],
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "scene_counts": {
                "source_total": len(source_scene_ids),
                "trainable_total": len(trainable_scene_ids),
                "train": len(train_scenes),
                "val": len(val_scenes),
            },
            "sample_counts": {
                "source_total": len(qas),
                "trainable_total": len(trainable_qas),
                "filtered_non_trainable": len(qas) - len(trainable_qas),
                "placeholder_total": placeholder_count,
                "empty_answer_total": empty_answer_count,
                "train": len(train_rows),
                "val": len(val_rows),
                "exported_total": len(train_rows) + len(val_rows),
            },
            "filtered_counts_by_subtemplate": dict(sorted(filtered_template_counts.items())),
        },
    )

    print(f"Wrote dataset_dir: {output_dir}")
    print(f"Template version: {template_version} | dataset prefix: {dataset_prefix}")
    print(f"Train scenes: {len(train_scenes)}, val scenes: {len(val_scenes)}")
    print(
        "Source samples: "
        f"{len(qas)} | trainable: {len(trainable_qas)} | filtered: {len(qas) - len(trainable_qas)}"
    )
    print(f"Train samples: {len(train_rows)}, val samples: {len(val_rows)}")


if __name__ == "__main__":
    main()
