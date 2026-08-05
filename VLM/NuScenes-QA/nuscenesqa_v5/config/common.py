from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

DEFAULT_ENV_NAME = "nuscenesqa-cu128"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "intersection_nuscenesqa_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "nuscenesqa_v5"
DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_RATIO = 0.145
DEFAULT_CAMERA_ORDER = ("north", "south", "east", "west")

TRAINING_DEFAULTS = {
    "seed": 3407,
    "epochs": 4,
    "batch_size_per_gpu": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "num_workers": 8,
    "log_every": 20,
    "eval_every": 1,
    "max_question_chars": 224,
    "max_answer_chars": 196,
}

MODEL_DEFAULTS = {
    "object_limit": 100,
    "hidden_size": 256,
    "char_embed_size": 128,
    "bbox_embed_size": 128,
    "layers": 4,
    "multi_head": 4,
    "ff_size": 1024,
    "dropout": 0.1,
    "flat_mlp_size": 256,
    "flat_glimpses": 1,
}

FEATURE_DEFAULTS = {
    "output_subdir": "features",
    "bbox_feature_dim": 7,
}

DATASET_VERSION_DEFAULTS: dict[str, dict[str, Any]] = {
    "v5": {
        "qa_json": DEFAULT_QA_JSON,
        "evaluator": DEFAULT_EVALUATOR,
        "prepared_dir": DEFAULT_PREPARED_DIR,
        "work_dir": DEFAULT_WORK_DIR,
        "annotation_key": "v5_manual_annotations",
    },
    "v6": {
        "qa_json": LLM_ROOT / "intersection_qa_pairs_v6.json",
        "evaluator": LLM_ROOT / "utils" / "evaluate_intersection_vqa_v6.py",
        "prepared_dir": REPO_ROOT / "data" / "intersection_nuscenesqa_v6",
        "work_dir": VLM_ROOT / "work_dirs" / "nuscenesqa_v6",
        # The current infos bundle still stores manual annotations under the v5 key.
        "annotation_key": "v5_manual_annotations",
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
        "annotation_key": str(defaults["annotation_key"]),
    }


def worktree_paths(work_dir: Path) -> Dict[str, Path]:
    checkpoints = work_dir / "checkpoints"
    logs = work_dir / "logs"
    plots = work_dir / "plots"
    predictions = work_dir / "predictions"
    metrics = work_dir / "metrics"
    features = work_dir / FEATURE_DEFAULTS["output_subdir"]
    return {
        "work_dir": work_dir,
        "checkpoints": checkpoints,
        "best_checkpoint": checkpoints / "best.pt",
        "logs": logs,
        "loss_history_jsonl": logs / "loss_history.jsonl",
        "loss_history_json": logs / "loss_history.json",
        "best_metrics_json": logs / "best_metrics.json",
        "train_summary_json": logs / "train_summary.json",
        "plots": plots,
        "loss_curve_png": plots / "loss_curves.png",
        "predictions": predictions,
        "metrics": metrics,
        "features": features,
        "feature_manifest": features / "feature_manifest.json",
    }


def ensure_worktree_layout(work_dir: Path) -> Dict[str, Path]:
    paths = worktree_paths(work_dir)
    for key in ("checkpoints", "logs", "plots", "predictions", "metrics", "features"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (paths["features"] / "train").mkdir(parents=True, exist_ok=True)
    (paths["features"] / "val").mkdir(parents=True, exist_ok=True)
    return paths
