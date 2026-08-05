from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


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
    default_mode = prompt_metadata.get("default_mode")
    _require(isinstance(default_mode, str) and default_mode.strip(), f"metadata.prompt_metadata.default_mode must be a non-empty string: {qa_json}")
    mode_config = prompt_metadata.get(default_mode)
    _require(isinstance(mode_config, dict), f"metadata.prompt_metadata.{default_mode} must be an object: {qa_json}")
    system_prompt = mode_config.get("system_prompt")
    user_prompt_template = mode_config.get("user_prompt_template")
    _require(isinstance(system_prompt, str) and system_prompt.strip(), f"metadata.prompt_metadata.{default_mode}.system_prompt must be a non-empty string: {qa_json}")
    _require(
        isinstance(user_prompt_template, str) and user_prompt_template.strip(),
        f"metadata.prompt_metadata.{default_mode}.user_prompt_template must be a non-empty string: {qa_json}",
    )
    template_registry = metadata.get("template_registry")
    _require(isinstance(template_registry, dict) and template_registry, f"metadata.template_registry must be a non-empty object: {qa_json}")
    subtemplate_patches = mode_config.get("subtemplate_patches") or {}
    strict_answer_schemas = mode_config.get("strict_answer_schemas") or {}
    _require(isinstance(subtemplate_patches, dict), f"metadata.prompt_metadata.{default_mode}.subtemplate_patches must be an object: {qa_json}")
    _require(isinstance(strict_answer_schemas, dict), f"metadata.prompt_metadata.{default_mode}.strict_answer_schemas must be an object: {qa_json}")
    return {
        "qa_json": qa_json,
        "default_mode": default_mode,
        "prompt_version": mode_config.get("version"),
        "subtemplate_patch_style": mode_config.get("subtemplate_patch_style"),
        "system_prompt": system_prompt.strip(),
        "user_prompt_template": user_prompt_template,
        "subtemplate_patches": subtemplate_patches,
        "strict_answer_schemas": strict_answer_schemas,
        "template_registry": template_registry,
    }


def _get_template_spec(subtemplate: str, prompt_bundle: dict[str, Any]) -> dict[str, Any]:
    registry = prompt_bundle["template_registry"]
    spec = registry.get(subtemplate)
    _require(isinstance(spec, dict), f"Missing template_registry entry for subtemplate={subtemplate}")
    title = spec.get("title")
    question_schema = spec.get("question_schema")
    answer_schema = spec.get("answer_schema")
    _require(isinstance(title, str) and title.strip(), f"template_registry[{subtemplate}].title must be a non-empty string")
    _require(
        isinstance(question_schema, str) and question_schema.strip(),
        f"template_registry[{subtemplate}].question_schema must be a non-empty string",
    )
    _require(
        isinstance(answer_schema, str) and answer_schema.strip(),
        f"template_registry[{subtemplate}].answer_schema must be a non-empty string",
    )
    return spec


def validate_subtemplates(subtemplates: Iterable[str], prompt_bundle: dict[str, Any]) -> None:
    for subtemplate in sorted({str(item) for item in subtemplates}):
        _get_template_spec(subtemplate, prompt_bundle)


def build_system_prompt(prompt_bundle: dict[str, Any]) -> str:
    return (
        f"{prompt_bundle['system_prompt'].rstrip()}\n\n"
        "Additional output rules:\n"
        "- Return only the final answer text.\n"
        "- Do not include explanation, markdown, or role labels."
    )


def build_subtemplate_prompt(record: dict[str, object], prompt_bundle: dict[str, Any]) -> str:
    subtemplate = str(record.get("subtemplate", ""))
    special = SPECIAL_SUBTEMPLATE_PROMPTS.get(subtemplate)
    if special is not None:
        return special
    spec = _get_template_spec(subtemplate, prompt_bundle)
    lines = [
        "Subtemplate-specific instruction:",
        f"- Title: {spec['title']}",
        f"- Expected question schema: {spec['question_schema']}",
        f"- Expected answer schema: {spec['answer_schema']}",
    ]
    strict_schema = prompt_bundle["strict_answer_schemas"].get(subtemplate)
    if isinstance(strict_schema, str) and strict_schema.strip():
        lines.append(f"- Strict answer schema: {strict_schema.strip()}")
    patch = prompt_bundle["subtemplate_patches"].get(subtemplate)
    if isinstance(patch, str) and patch.strip():
        lines.append(f"- Extra template guidance: {patch.strip()}")
    lines.append("- Follow the expected answer shape as closely as possible.")
    lines.append("- Return only the final answer.")
    return "\n".join(lines)


def build_user_prompt(record: dict[str, object], prompt_bundle: dict[str, Any]) -> str:
    chapter = str(record.get("chapter", ""))
    section = str(record.get("section", ""))
    subtemplate = str(record.get("subtemplate", ""))
    question = str(record["question"])
    task_rule_block = build_subtemplate_prompt(record, prompt_bundle)
    return prompt_bundle["user_prompt_template"].format(
        chapter=chapter,
        section=section,
        subtemplate=subtemplate,
        question=question,
        task_rule_block=task_rule_block,
    ).rstrip()
