from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path("/home/suolab/LLM/VLM/Senna").resolve()
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

DEFAULT_ENV_NAME = "senna-cu128"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"

DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "intersection_senna_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "senna_v5"
DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_DIR / "checkpoints"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"
DEFAULT_PLOT_DIR = DEFAULT_WORK_DIR / "plots"
DEFAULT_RESULTS_DIR = DEFAULT_WORK_DIR / "predictions"
DEFAULT_EVAL_DIR = DEFAULT_WORK_DIR / "metrics"

DEFAULT_BEST_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "best"
DEFAULT_BEST_METRICS_JSON = DEFAULT_CHECKPOINT_DIR / "best_metrics.json"
DEFAULT_TRAIN_SUMMARY_JSON = DEFAULT_LOG_DIR / "train_summary.json"
DEFAULT_LOSS_HISTORY_JSON = DEFAULT_LOG_DIR / "loss_history.json"
DEFAULT_LOSS_HISTORY_JSONL = DEFAULT_LOG_DIR / "loss_history.jsonl"
DEFAULT_LOSS_CURVE_PNG = DEFAULT_PLOT_DIR / "loss_curves.png"
DEFAULT_MERGED_PREDICTIONS = DEFAULT_RESULTS_DIR / "merged_predictions.jsonl"

DEFAULT_MODEL_NAME_OR_PATH = REPO_ROOT / "weights" / "senna"
DEFAULT_VISION_TOWER = REPO_ROOT / "weights" / "clip-vit-large-patch14-336"
DEFAULT_PROMPT_VERSION = "v1"

DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_RATIO = 0.10
DEFAULT_CAMERA_ORDER = ("north", "south", "east", "west")

TRAINING_DEFAULTS = {
    "mode": "lora",
    "epochs": 3,
    "batch_size": 1,
    "eval_batch_size": 1,
    "grad_accum": 8,
    "learning_rate": 1e-4,
    "mm_projector_lr": 1e-4,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 20,
    "model_max_length": 4096,
    "workers": 4,
    "lora_r": 8,
    "lora_alpha": 16,
    "warmup_ratio": 0.1,
    "save_total_limit": 2,
    "lr_scheduler_type": "cosine",
}

INFERENCE_DEFAULTS = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": None,
    "num_beams": 1,
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
        "prepared_dir": REPO_ROOT / "data" / "intersection_senna_v6",
        "work_dir": VLM_ROOT / "work_dirs" / "senna_v6",
    },
}


def worktree_paths(work_dir: Path) -> dict[str, Path]:
    work_dir = work_dir.resolve()
    checkpoints = work_dir / "checkpoints"
    logs = work_dir / "logs"
    plots = work_dir / "plots"
    predictions = work_dir / "predictions"
    metrics = work_dir / "metrics"
    return {
        "root": work_dir,
        "checkpoints": checkpoints,
        "logs": logs,
        "plots": plots,
        "predictions": predictions,
        "metrics": metrics,
        "best_checkpoint": checkpoints / "best",
        "best_metrics_json": checkpoints / "best_metrics.json",
        "train_summary_json": logs / "train_summary.json",
        "loss_history_json": logs / "loss_history.json",
        "loss_history_jsonl": logs / "loss_history.jsonl",
        "loss_curve_png": plots / "loss_curves.png",
        "merged_predictions": predictions / "merged_predictions.jsonl",
    }


def ensure_worktree_layout(work_dir: Path) -> dict[str, Path]:
    paths = worktree_paths(work_dir)
    for directory in (
        paths["root"],
        paths["checkpoints"],
        paths["logs"],
        paths["plots"],
        paths["predictions"],
        paths["metrics"],
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def default_env_file() -> Path:
    return REPO_ROOT / "env" / "environment_intersection_v5.yml"


def default_env_setup_script() -> Path:
    return REPO_ROOT / "scripts" / "setup_intersection_v5_env.sh"


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
        "qa_json": Path(qa_json).expanduser().resolve() if qa_json is not None else Path(defaults["qa_json"]).resolve(),
        "evaluator": Path(evaluator).expanduser().resolve() if evaluator is not None else Path(defaults["evaluator"]).resolve(),
        "prepared_dir": Path(prepared_dir).expanduser().resolve() if prepared_dir is not None else Path(defaults["prepared_dir"]).resolve(),
        "work_dir": Path(work_dir).expanduser().resolve() if work_dir is not None else Path(defaults["work_dir"]).resolve(),
    }
