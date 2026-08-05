from __future__ import annotations

import os.path as osp
from pathlib import Path


REPO_ROOT = Path("/home/suolab/LLM/VLM/OpenDriveVLA").resolve()
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

DEFAULT_ENV_NAME = "opendrivevla-cu128"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_MODEL_PATH = REPO_ROOT / "weights"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "sunlakes_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "openDriveVLA_v5"
DEFAULT_FEATURE_TRAIN_DIR = DEFAULT_WORK_DIR / "features_train"
DEFAULT_FEATURE_VAL_DIR = DEFAULT_WORK_DIR / "features_val"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"
DEFAULT_PLOT_DIR = DEFAULT_WORK_DIR / "plots"
DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_DIR / "checkpoints"
DEFAULT_PREDICTION_DIR = DEFAULT_WORK_DIR / "predictions"
DEFAULT_METRIC_DIR = DEFAULT_WORK_DIR / "metrics"
DEFAULT_TRAINING_TEMP_DIR = DEFAULT_WORK_DIR / "_trainer"
DEFAULT_LOSS_HISTORY_JSON = DEFAULT_LOG_DIR / "loss_history.json"
DEFAULT_LOSS_HISTORY_JSONL = DEFAULT_LOG_DIR / "loss_history.jsonl"
DEFAULT_LOSS_CURVE_PNG = DEFAULT_PLOT_DIR / "loss_curves.png"
DEFAULT_BEST_DIR = DEFAULT_CHECKPOINT_DIR / "best.pth"
DEFAULT_LAST_DIR = DEFAULT_CHECKPOINT_DIR / "last.pth"
DEFAULT_BEST_METRICS_JSON = DEFAULT_CHECKPOINT_DIR / "best_metrics.json"
DEFAULT_PREDS_JSONL = DEFAULT_PREDICTION_DIR / "merged_predictions.jsonl"
DEFAULT_UNIAD_CONFIG = REPO_ROOT / "projects" / "configs" / "stage1_track_map" / "sunlakes_intersection_v5_4cam.py"

STRICT_CAM_ORDER = ("CAM_NORTH", "CAM_EAST", "CAM_SOUTH", "CAM_WEST")
DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_RATIO = 0.10
CAN_BUS_DIM = 18

TRAIN_QAS = 22104
VAL_QAS = 2285
TRAIN_FRAMES = 3008
VAL_FRAMES = 306

TRAINING_DEFAULTS = {
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "logging_steps": 100,
    "eval_steps": 400,
    "save_steps": 400,
    "save_total_limit": 2,
    "seed": 42,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_train_samples": None,
}

EVAL_DEFAULTS = {
    "batch_size": 1,
    "max_new_tokens": 1024,
    "bertscore_model": "roberta-large",
    "sim_model": "sentence-transformers/all-mpnet-base-v2",
}

DATASET_VERSION_DEFAULTS = {
    "v5": {
        "qa_json": DEFAULT_QA_JSON,
        "evaluator": DEFAULT_EVALUATOR,
        "prepared_dir": DEFAULT_PREPARED_DIR,
        "work_dir": DEFAULT_WORK_DIR,
    },
    "v6": {
        "qa_json": LLM_ROOT / "intersection_qa_pairs_v6.json",
        "evaluator": LLM_ROOT / "utils" / "evaluate_intersection_vqa_v6.py",
        "prepared_dir": REPO_ROOT / "data" / "sunlakes_v6",
        "work_dir": VLM_ROOT / "work_dirs" / "openDriveVLA_v6",
    },
}


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_worktree_layout(work_dir: Path) -> None:
    for directory in (
        work_dir,
        work_dir / "checkpoints",
        work_dir / "logs",
        work_dir / "plots",
        work_dir / "predictions",
        work_dir / "metrics",
        work_dir / "features_train",
        work_dir / "features_val",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def default_env_setup_script() -> Path:
    return REPO_ROOT / "scripts" / "setup_intersection_v5_env.sh"


def default_env_file() -> Path:
    return REPO_ROOT / "env" / "environment_intersection_v5.yml"


def resolve_dataset_version_paths(
    dataset_version: str,
    *,
    qa_json: str | Path | None = None,
    evaluator: str | Path | None = None,
    prepared_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Path | str]:
    if dataset_version not in DATASET_VERSION_DEFAULTS:
        raise ValueError(f"Unsupported dataset_version={dataset_version}")
    defaults = DATASET_VERSION_DEFAULTS[dataset_version]
    return {
        "dataset_version": dataset_version,
        "qa_json": resolve_path(qa_json) if qa_json is not None else resolve_path(defaults["qa_json"]),
        "evaluator": resolve_path(evaluator) if evaluator is not None else resolve_path(defaults["evaluator"]),
        "prepared_dir": resolve_path(prepared_dir) if prepared_dir is not None else resolve_path(defaults["prepared_dir"]),
        "work_dir": resolve_path(work_dir) if work_dir is not None else resolve_path(defaults["work_dir"]),
    }
