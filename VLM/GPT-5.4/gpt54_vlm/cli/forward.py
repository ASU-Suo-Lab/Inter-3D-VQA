from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from tqdm import tqdm

from gpt54_vlm.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL,
    FORWARD_DEFAULTS,
    PROVIDER_NAME,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from gpt54_vlm.utils.client import GPT54Client
from gpt54_vlm.utils.io import append_jsonl, dump_json, dump_jsonl, ensure, load_json, load_jsonl
from gpt54_vlm.utils.prompting import build_system_prompt, build_user_prompt, load_prompt_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-5.1 intersection forward.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=FORWARD_DEFAULTS["max_output_tokens"])
    parser.add_argument("--request-timeout-seconds", type=int, default=FORWARD_DEFAULTS["request_timeout_seconds"])
    parser.add_argument("--max-retries", type=int, default=FORWARD_DEFAULTS["max_retries"])
    parser.add_argument("--retry-backoff-seconds", type=float, default=FORWARD_DEFAULTS["retry_backoff_seconds"])
    return parser.parse_args()


def build_forward_run_payload(
    *,
    run_id: str,
    provider: str,
    model: str,
    dataset_version: str,
    prepared_dir: Path,
    requested_count: int,
    success_count: int,
    failed_count: int,
    status: str,
    last_question_id: str | None,
    error_type: str | None,
    error_message: str | None,
    started_at: str,
    finished_at: str | None,
    qa_json: Path,
    prompt_mode: str,
    prompt_version: str | None,
    subtemplate_patch_style: str | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "dataset_version": dataset_version,
        "prepared_dir": str(prepared_dir),
        "qa_json": str(qa_json),
        "prompt_mode": prompt_mode,
        "prompt_version": prompt_version,
        "subtemplate_patch_style": subtemplate_patch_style,
        "requested_count": requested_count,
        "processed_count": success_count + failed_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "prediction_count": success_count,
        "status": status,
        "last_question_id": last_question_id,
        "error_type": error_type,
        "error_message": error_message,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def format_decimal(value: str) -> str:
    return f"{float(value):.1f}"


def normalize_location_phrase(location: str) -> str:
    normalized = " ".join(location.strip().split()).lower().rstrip(".")
    if normalized == "in the center area":
        return "in the center area"
    if normalized == "in the center area of the intersection":
        return "in the center area of the intersection"
    match = re.fullmatch(r"on the (north|south|east|west) approach(?: of the intersection)?", normalized)
    if match:
        side = match.group(1)
        suffix = " of the intersection" if "intersection" in normalized else ""
        return f"on the {side} approach{suffix}"
    return normalized


def normalize_motion_prediction(prediction: str) -> tuple[str, list[str], bool]:
    text = " ".join(prediction.strip().split())
    diagnostics: list[str] = []
    if not text:
        return text, diagnostics, False
    lowered = text.lower()
    full_match = re.fullmatch(
        r"(?:the\s+)?(?P<object>[a-z0-9_ /-]+?)\s+"
        r"(?P<location>on the (?:north|south|east|west) approach|in the center area)\s+"
        r"is\s+(?P<state>[a-z-]+)\s+at\s+(?:about\s+)?[~≈]?(?P<speed>-?\d+(?:\.\d+)?)\s*m/s"
        r"(?:\s+and\s+(?P<acc_state>accelerating|decelerating)\s+at\s+(?:about\s+)?[~≈]?(?P<acc>-?\d+(?:\.\d+)?)\s*m/s\^2)?\.?",
        lowered,
    )
    if full_match:
        object_type = " ".join(full_match.group("object").split())
        location = normalize_location_phrase(full_match.group("location"))
        state = full_match.group("state")
        speed = format_decimal(full_match.group("speed"))
        acc_state = full_match.group("acc_state")
        acc = full_match.group("acc")
        if acc_state and acc is not None:
            normalized = (
                f"The {object_type} {location} is {state} at {speed} m/s and "
                f"{acc_state} at {format_decimal(acc)} m/s^2."
            )
        else:
            normalized = f"The {object_type} {location} is {state} at {speed} m/s."
        return normalized, diagnostics, normalized != prediction
    if re.fullmatch(
        r"[a-z-]+\s*,\s*(?:about\s+)?[~≈]?-?\d+(?:\.\d+)?\s*m/s"
        r"(?:\s+and\s+(?:accelerating|decelerating)\s+at\s+-?\d+(?:\.\d+)?\s*m/s\^2)?\.?",
        lowered,
    ):
        diagnostics.extend(["shorthand_motion_answer", "missing_object_type", "missing_location_phrase"])
        if "m/s^2" in lowered:
            diagnostics.append("missing_acceleration_payload")
        return text, diagnostics, False
    if "m/s" not in lowered:
        diagnostics.append("missing_speed_value")
    if "center area" not in lowered and "approach" not in lowered:
        diagnostics.append("missing_location_phrase")
    if not lowered.startswith("the "):
        diagnostics.append("missing_object_type")
    return text, diagnostics, False


def normalize_speeding_risk_prediction(prediction: str) -> tuple[str, list[str], bool]:
    text = " ".join(prediction.strip().split())
    diagnostics: list[str] = []
    if not text:
        return text, diagnostics, False
    lowered = text.lower().rstrip()
    if lowered in {"no", "no.", "no,"}:
        normalized = "No."
        return normalized, diagnostics, normalized != prediction
    yes_match = re.fullmatch(
        r"yes[,.]?\s+(?:a\s+)?(?P<object>[a-z0-9_ /-]+?)\s+"
        r"(?P<location>on the (?:north|south|east|west) approach(?: of the intersection)?|in the center area(?: of the intersection)?)\s+"
        r"(?:is\s+(?:still\s+)?(?:moving|traveling)|appears\s+to\s+be\s+(?:moving|traveling)(?:\s+at\s+high\s+speed)?|appears\s+to\s+be\s+traveling\s+at\s+high\s+speed)\s+"
        r"at(?:\s+about)?\s+(?P<speed>-?\d+(?:\.\d+)?)\s*m/s\.?",
        lowered,
    )
    if yes_match:
        object_type = " ".join(yes_match.group("object").split())
        location = normalize_location_phrase(yes_match.group("location"))
        if location == "in the center area":
            location = "in the center area of the intersection"
        elif location.startswith("on the ") and not location.endswith("of the intersection"):
            location = f"{location} of the intersection"
        normalized = f"Yes, a {object_type} {location} is still moving at about {format_decimal(yes_match.group('speed'))} m/s."
        return normalized, diagnostics, normalized != prediction
    if lowered.startswith("yes"):
        diagnostics.append("generic_explanatory_answer")
        if "m/s" not in lowered:
            diagnostics.append("missing_speed_value")
        if "center area" not in lowered and "approach" not in lowered:
            diagnostics.append("missing_location_phrase")
        if not re.search(r"\byes[,.]?\s+a\s+[a-z0-9_ /-]+", lowered):
            diagnostics.append("missing_object_type")
        return text, diagnostics, False
    if lowered.startswith("no"):
        normalized = "No."
        diagnostics.append("noncanonical_no_answer")
        return normalized, diagnostics, normalized != prediction
    diagnostics.append("invalid_speeding_answer")
    return text, diagnostics, False


def normalize_prediction_for_subtemplate(subtemplate: str, prediction: str) -> tuple[str, list[str], bool]:
    if subtemplate == "3_1_1_current_motion_state":
        return normalize_motion_prediction(prediction)
    if subtemplate == "4_2_1_speeding_risk":
        return normalize_speeding_risk_prediction(prediction)
    return prediction.strip(), [], prediction.strip() != prediction


def load_existing_predictions(
    *,
    worktree: dict[str, Path],
    dataset_version: str,
    model: str,
    provider: str,
    allowed_question_ids: set[str],
) -> list[dict[str, object]]:
    state_path = worktree["forward_run_json"]
    if state_path.is_file():
        payload = load_json(state_path)
        ensure(isinstance(payload, dict), "forward_run.json must be a JSON object.")
        if payload.get("dataset_version") is not None:
            ensure(
                payload.get("dataset_version") == dataset_version,
                f"Existing predictions dataset_version={payload.get('dataset_version')} does not match requested {dataset_version}.",
            )
        if payload.get("model") is not None:
            ensure(payload.get("model") == model, f"Existing predictions model={payload.get('model')} does not match requested {model}.")
        if payload.get("provider") is not None:
            ensure(
                payload.get("provider") == provider,
                f"Existing predictions provider={payload.get('provider')} does not match requested {provider}.",
            )

    source_path: Path | None = None
    if worktree["predictions_merged"].is_file():
        source_path = worktree["predictions_merged"]
    elif worktree["predictions_rank0"].is_file():
        source_path = worktree["predictions_rank0"]
    if source_path is None:
        return []

    loaded_rows = load_jsonl(source_path)
    rows: list[dict[str, object]] = []
    seen_question_ids: set[str] = set()
    for row in loaded_rows:
        ensure(isinstance(row, dict), f"Prediction rows in {source_path} must be JSON objects.")
        question_id = str(row.get("question_id", ""))
        ensure(question_id, f"Prediction rows in {source_path} must include question_id.")
        ensure(question_id in allowed_question_ids, f"Existing prediction question_id={question_id} is not part of the requested evaluation set.")
        ensure(question_id not in seen_question_ids, f"Duplicate existing prediction for question_id={question_id}.")
        seen_question_ids.add(question_id)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    qa_json = Path(resolved["qa_json"]).resolve()
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
    prompt_bundle = load_prompt_bundle(qa_json)
    manifest = load_json(prepared_dir / "split_manifest.json")
    ensure(isinstance(manifest, dict), "split_manifest.json must be a JSON object.")
    ensure(manifest.get("dataset_version") == resolved["dataset_version"], f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.")
    records = load_json(prepared_dir / "val_eval.json")
    ensure(isinstance(records, list) and records, f"No evaluation rows found in {prepared_dir / 'val_eval.json'}")
    if args.limit is not None:
        records = records[: max(int(args.limit), 0)]
    question_ids = [str(row.get("question_id", "")) for row in records if isinstance(row, dict)]
    ensure(len(question_ids) == len(records), "Each val_eval row must be an object with question_id.")
    ensure(len(set(question_ids)) == len(question_ids), "val_eval.json contains duplicate question_id values.")
    client = GPT54Client(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    system_prompt = build_system_prompt(prompt_bundle)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    existing_rows = load_existing_predictions(
        worktree=worktree,
        dataset_version=str(resolved["dataset_version"]),
        model=args.model,
        provider=PROVIDER_NAME,
        allowed_question_ids=set(question_ids),
    )
    if existing_rows:
        dump_jsonl(worktree["predictions_rank0"], existing_rows)
        dump_jsonl(worktree["predictions_merged"], existing_rows)
    else:
        for prediction_path in (worktree["predictions_rank0"], worktree["predictions_merged"]):
            prediction_path.unlink(missing_ok=True)
    completed_question_ids = {str(row["question_id"]) for row in existing_rows}
    pending_records = [row for row in records if str(row["question_id"]) not in completed_question_ids]
    success_count = len(existing_rows)
    failed_count = 0
    dump_json(
        worktree["forward_run_json"],
        build_forward_run_payload(
            run_id=run_id,
            provider=PROVIDER_NAME,
            model=args.model,
            dataset_version=str(resolved["dataset_version"]),
            prepared_dir=prepared_dir,
            requested_count=len(records),
            success_count=success_count,
            failed_count=failed_count,
            status="running",
            last_question_id=None,
            error_type=None,
            error_message=None,
            started_at=started_at,
            finished_at=None,
            qa_json=qa_json,
            prompt_mode=str(prompt_bundle["default_mode"]),
            prompt_version=prompt_bundle.get("prompt_version"),
            subtemplate_patch_style=prompt_bundle.get("subtemplate_patch_style"),
        ),
    )
    if not pending_records:
        dump_json(
            worktree["forward_run_json"],
            build_forward_run_payload(
                run_id=run_id,
                provider=PROVIDER_NAME,
                model=args.model,
                dataset_version=str(resolved["dataset_version"]),
                prepared_dir=prepared_dir,
                requested_count=len(records),
                success_count=success_count,
                failed_count=failed_count,
                status="completed",
                last_question_id=question_ids[-1] if question_ids else None,
                error_type=None,
                error_message=None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                qa_json=qa_json,
                prompt_mode=str(prompt_bundle["default_mode"]),
                prompt_version=prompt_bundle.get("prompt_version"),
                subtemplate_patch_style=prompt_bundle.get("subtemplate_patch_style"),
            ),
        )
        return
    progress = tqdm(total=len(records), initial=success_count, desc="Forward", unit="sample")
    for row in pending_records:
        question_id = None
        try:
            ensure(isinstance(row, dict), "Each val_eval row must be an object.")
            question_id = str(row.get("question_id", ""))
            progress.set_postfix_str(question_id)
            image_paths = [Path(str(image)).resolve() for image in row["images"]]
            for image_path in image_paths:
                ensure(image_path.is_file(), f"Missing image: {image_path}")
            prediction = client.answer(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(row, prompt_bundle),
                image_paths=image_paths,
            )
            normalized_prediction, prediction_diagnostics, prediction_normalized = normalize_prediction_for_subtemplate(
                str(row["subtemplate"]), prediction
            )
            output_row = {
                "question_id": row["question_id"],
                "scene_id": row["scene_id"],
                "frame_token": row["frame_token"],
                "chapter": row["chapter"],
                "section": row["section"],
                "subtemplate": row["subtemplate"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "prediction": normalized_prediction,
                "raw_prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "prediction_diagnostics": prediction_diagnostics,
            }
            append_jsonl(worktree["predictions_rank0"], output_row)
            append_jsonl(worktree["predictions_merged"], output_row)
            success_count += 1
            progress.update(1)
            dump_json(
                worktree["forward_run_json"],
                build_forward_run_payload(
                    run_id=run_id,
                    provider=PROVIDER_NAME,
                    model=args.model,
                    dataset_version=str(resolved["dataset_version"]),
                    prepared_dir=prepared_dir,
                    requested_count=len(records),
                    success_count=success_count,
                    failed_count=failed_count,
                    status="running",
                    last_question_id=question_id,
                    error_type=None,
                    error_message=None,
                    started_at=started_at,
                    finished_at=None,
                    qa_json=qa_json,
                    prompt_mode=str(prompt_bundle["default_mode"]),
                    prompt_version=prompt_bundle.get("prompt_version"),
                    subtemplate_patch_style=prompt_bundle.get("subtemplate_patch_style"),
                ),
            )
        except Exception as exc:
            failed_count += 1
            dump_json(
                worktree["forward_run_json"],
                build_forward_run_payload(
                    run_id=run_id,
                    provider=PROVIDER_NAME,
                    model=args.model,
                    dataset_version=str(resolved["dataset_version"]),
                    prepared_dir=prepared_dir,
                    requested_count=len(records),
                    success_count=success_count,
                    failed_count=failed_count,
                    status="interrupted",
                    last_question_id=question_id,
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    qa_json=qa_json,
                    prompt_mode=str(prompt_bundle["default_mode"]),
                    prompt_version=prompt_bundle.get("prompt_version"),
                    subtemplate_patch_style=prompt_bundle.get("subtemplate_patch_style"),
                ),
            )
            raise
    progress.close()
    dump_json(
        worktree["forward_run_json"],
        build_forward_run_payload(
            run_id=run_id,
            provider=PROVIDER_NAME,
            model=args.model,
            dataset_version=str(resolved["dataset_version"]),
            prepared_dir=prepared_dir,
            requested_count=len(records),
            success_count=success_count,
            failed_count=failed_count,
            status="completed",
            last_question_id=str(records[-1].get("question_id", "")) if records else None,
            error_type=None,
            error_message=None,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            qa_json=qa_json,
            prompt_mode=str(prompt_bundle["default_mode"]),
            prompt_version=prompt_bundle.get("prompt_version"),
            subtemplate_patch_style=prompt_bundle.get("subtemplate_patch_style"),
        ),
    )


if __name__ == "__main__":
    main()
