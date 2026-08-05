from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geovlm_intersection.data.v5_io import PreparedSample


SPECIAL_SUBTEMPLATE_PROMPTS: dict[str, str] = {
    "3_1_1_current_motion_state": "\n".join(
        [
            "Subtemplate-specific instruction:",
            "- Title: Current Motion State",
            "- You must answer with one complete sentence only.",
            "- Allowed output forms:",
            "  1. The {object_type} on the {north|south|east|west} approach is {motion_state} at {speed} m/s.",
            "  2. The {object_type} in the center area is {motion_state} at {speed} m/s.",
            "  3. The {object_type} ... is {motion_state} at {speed} m/s and {accelerating|decelerating} at {acceleration} m/s^2.",
            "- Use one decimal place for speed and acceleration.",
            "- Do not omit the object type.",
            "- Do not omit the location phrase.",
            "- Do not use shorthand such as 'stopped, 0.0 m/s' or 'moving, ~10 m/s'.",
            "- Do not describe future intent or maneuver.",
            "- Correct example: The truck on the east approach is stopped at 0.2 m/s.",
            "- Correct example: The golf cart on the south approach is braking at 5.2 m/s and decelerating at -0.3 m/s^2.",
            "- Wrong example: stopped, 0.0 m/s",
            "- Wrong example: moving, ~10 m/s",
            "- Return only the final answer.",
        ]
    ),
    "4_2_1_speeding_risk": "\n".join(
        [
            "Subtemplate-specific instruction:",
            "- Title: Speeding Risk",
            "- You must answer with one complete sentence only.",
            "- If there is no speeding risk, output exactly: No.",
            "- If there is a speeding risk, output exactly one of these forms:",
            "  1. Yes, a {vehicle_type} on the {north|south|east|west} approach of the intersection is still moving at about {speed} m/s.",
            "  2. Yes, a {vehicle_type} in the center area of the intersection is still moving at about {speed} m/s.",
            "- Use one decimal place for speed.",
            "- Do not give a general explanation about traffic moving quickly.",
            "- Do not mention multiple vehicles.",
            "- Do not omit the vehicle type, location, or speed when the answer is Yes.",
            "- Correct example: Yes, a truck on the east approach of the intersection is still moving at about 18.2 m/s.",
            "- Correct example: No.",
            "- Wrong example: Yes, there is still a risk of speeding because several vehicles appear to be moving quickly.",
            "- Return only the final answer.",
        ]
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_prompt_bundle(qa_json_path: str | Path) -> dict[str, Any]:
    qa_json = Path(qa_json_path).resolve()
    _require(qa_json.is_file(), f"Missing QA JSON: {qa_json}")
    payload = json.loads(qa_json.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"QA JSON root must be an object: {qa_json}")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), f"QA JSON metadata must be an object: {qa_json}")
    prompt_metadata = metadata.get("prompt_metadata")
    _require(isinstance(prompt_metadata, dict), f"metadata.prompt_metadata must be an object: {qa_json}")
    prompt_mode = "pointcloud_plus_image"
    mode_config = prompt_metadata.get(prompt_mode)
    _require(isinstance(mode_config, dict), f"metadata.prompt_metadata.{prompt_mode} must be an object: {qa_json}")
    system_prompt = mode_config.get("system_prompt")
    user_prompt_template = mode_config.get("user_prompt_template")
    _require(
        isinstance(system_prompt, str) and system_prompt.strip(),
        f"metadata.prompt_metadata.{prompt_mode}.system_prompt must be a non-empty string: {qa_json}",
    )
    _require(
        isinstance(user_prompt_template, str) and user_prompt_template.strip(),
        f"metadata.prompt_metadata.{prompt_mode}.user_prompt_template must be a non-empty string: {qa_json}",
    )
    subtemplate_patches = mode_config.get("subtemplate_patches") or {}
    strict_answer_schemas = mode_config.get("strict_answer_schemas") or {}
    _require(
        isinstance(subtemplate_patches, dict),
        f"metadata.prompt_metadata.{prompt_mode}.subtemplate_patches must be an object: {qa_json}",
    )
    _require(
        isinstance(strict_answer_schemas, dict),
        f"metadata.prompt_metadata.{prompt_mode}.strict_answer_schemas must be an object: {qa_json}",
    )
    return {
        "qa_json": qa_json,
        "prompt_mode": prompt_mode,
        "prompt_version": mode_config.get("version"),
        "subtemplate_patch_style": mode_config.get("subtemplate_patch_style"),
        "system_prompt": system_prompt.strip(),
        "user_prompt_template": user_prompt_template,
        "subtemplate_patches": subtemplate_patches,
        "strict_answer_schemas": strict_answer_schemas,
    }


def validate_subtemplates(subtemplates: list[str] | tuple[str, ...] | set[str], prompt_bundle: dict[str, Any]) -> None:
    patches = prompt_bundle["subtemplate_patches"]
    for subtemplate in sorted({str(item) for item in subtemplates}):
        _require(
            subtemplate in patches and isinstance(patches[subtemplate], str) and patches[subtemplate].strip(),
            f"Missing pointcloud_plus_image subtemplate_patches entry for subtemplate={subtemplate}",
        )


def build_system_prompt(prompt_bundle: dict[str, Any]) -> str:
    return (
        f"{prompt_bundle['system_prompt'].rstrip()}\n\n"
        "Additional output rules:\n"
        "- Return only the final answer text.\n"
        "- Do not include explanation, markdown, or role labels."
    )


def _record_mapping(sample: PreparedSample) -> dict[str, object]:
    return {
        "chapter": sample.chapter,
        "section": sample.section,
        "subtemplate": sample.subtemplate,
        "question": sample.question,
        "point_cloud": "point_cloud",
    }


def _build_image_mapping_block(sample: PreparedSample) -> str:
    lines = ["Image-to-direction mapping:"]
    for image_idx, view in enumerate(sample.camera_views, start=1):
        lines.append(f"- image {image_idx} = {view.view_direction} view")
    lines.append("- global intersection point cloud: point_cloud")
    return "\n".join(lines)


def build_subtemplate_prompt(sample: PreparedSample, prompt_bundle: dict[str, Any]) -> str:
    special = SPECIAL_SUBTEMPLATE_PROMPTS.get(sample.subtemplate)
    if special is not None:
        return special
    patch = prompt_bundle["subtemplate_patches"].get(sample.subtemplate)
    _require(
        isinstance(patch, str) and patch.strip(),
        f"Missing pointcloud_plus_image subtemplate_patches entry for subtemplate={sample.subtemplate}",
    )
    lines = ["Subtemplate-specific instruction:", patch.strip()]
    strict_schema = prompt_bundle["strict_answer_schemas"].get(sample.subtemplate)
    if isinstance(strict_schema, str) and strict_schema.strip():
        lines.append(f"- Strict answer schema: {strict_schema.strip()}")
    lines.append("- Return only the final answer.")
    return "\n".join(lines)


def build_user_prompt(sample: PreparedSample, prompt_bundle: dict[str, Any]) -> str:
    task_rule_block = build_subtemplate_prompt(sample, prompt_bundle)
    user_prompt_template = str(prompt_bundle["user_prompt_template"])
    if "Task metadata:" in user_prompt_template:
        _, template_tail = user_prompt_template.split("Task metadata:", 1)
        normalized_template = f"{_build_image_mapping_block(sample)}\n\nTask metadata:{template_tail}"
    else:
        normalized_template = f"{_build_image_mapping_block(sample)}\n\n{user_prompt_template.lstrip()}"
    return normalized_template.format(task_rule_block=task_rule_block, **_record_mapping(sample)).rstrip()


def build_full_prompt(sample: PreparedSample, prompt_bundle: dict[str, Any]) -> tuple[str, str, str]:
    system_prompt = build_system_prompt(prompt_bundle)
    user_prompt = build_user_prompt(sample, prompt_bundle)
    full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}".rstrip()
    return system_prompt, user_prompt, full_prompt
