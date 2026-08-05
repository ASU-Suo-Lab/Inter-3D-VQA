from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

PACKAGE_NAME = "gemini31pro_vlm"
PROVIDER_NAME = "Gemini"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_ENV_NAME = "mla"
API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_PREPARED_DIR = VLM_ROOT / "NuScenes-QA" / "data" / "intersection_nuscenesqa_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "gemini31pro_v5"
LOCAL_API_KEY_FILE = REPO_ROOT / "api_keys.local.json"

FORWARD_DEFAULTS = {
    "temperature": 0.0,
    "max_output_tokens": 512,
    "request_timeout_seconds": 180,
    "check_image_count": 4,
    "media_resolution": "MEDIA_RESOLUTION_LOW",
    "max_retries": 4,
    "retry_backoff_seconds": 2.0,
}

DATASET_VERSION_DEFAULTS: dict[str, dict[str, Any]] = {
    "v5": {
        "qa_json": DEFAULT_QA_JSON,
        "evaluator": DEFAULT_EVALUATOR,
        "prepared_dir": DEFAULT_PREPARED_DIR,
        "work_dir": DEFAULT_WORK_DIR,
    },
    "v6": {
        "qa_json": LLM_ROOT / "intersection_qa_pairs_v6.json",
        "evaluator": LLM_ROOT / "utils" / "evaluate_intersection_vqa_v6.py",
        "prepared_dir": VLM_ROOT / "NuScenes-QA" / "data" / "intersection_nuscenesqa_v6",
        "work_dir": VLM_ROOT / "work_dirs" / "gemini31pro_v6",
    },
}


def validate_dataset_version(dataset_version: str) -> str:
    version = str(dataset_version).strip().lower()
    if version not in DATASET_VERSION_DEFAULTS:
        supported = ", ".join(sorted(DATASET_VERSION_DEFAULTS))
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Expected one of: {supported}")
    return version


def resolve_dataset_version_paths(
    dataset_version: str = DEFAULT_DATASET_VERSION,
    *,
    qa_json: str | Path | None = None,
    evaluator: str | Path | None = None,
    prepared_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    version = validate_dataset_version(dataset_version)
    defaults = DATASET_VERSION_DEFAULTS[version]
    return {
        "dataset_version": version,
        "qa_json": Path(qa_json).resolve() if qa_json is not None else Path(defaults["qa_json"]).resolve(),
        "evaluator": Path(evaluator).resolve() if evaluator is not None else Path(defaults["evaluator"]).resolve(),
        "prepared_dir": Path(prepared_dir).resolve() if prepared_dir is not None else Path(defaults["prepared_dir"]).resolve(),
        "work_dir": Path(work_dir).resolve() if work_dir is not None else Path(defaults["work_dir"]).resolve(),
    }


def worktree_paths(work_dir: Path) -> dict[str, Path]:
    predictions = work_dir / "predictions"
    metrics = work_dir / "metrics"
    logs = work_dir / "logs"
    return {
        "work_dir": work_dir,
        "predictions": predictions,
        "predictions_rank0": predictions / "predictions_rank0.jsonl",
        "predictions_merged": predictions / "merged_predictions.jsonl",
        "forward_run_json": predictions / "forward_run.json",
        "metrics": metrics,
        "metrics_run_json": metrics / "metrics_run.json",
        "partial_sidecar_jsonl": metrics / "_partial_sidecar.jsonl",
        "logs": logs,
    }


def ensure_worktree_layout(work_dir: Path) -> dict[str, Path]:
    paths = worktree_paths(work_dir)
    for key in ("work_dir", "predictions", "metrics", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths
