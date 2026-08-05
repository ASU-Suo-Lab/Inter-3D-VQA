from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

DEFAULT_ENV_NAME = "bevllm-v5"
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "intersection_bev_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "bevLLM_v5"
DEFAULT_OPENPCDET_ROOT = Path("/home/suolab/OpenPCDet")
DEFAULT_BEVFUSION_CONFIG = DEFAULT_OPENPCDET_ROOT / "output" / "sunlakes_models" / "bevfusion" / "v03_high_lidar_b4_e20" / "bevfusion.yaml"
DEFAULT_BEVFUSION_CKPT = DEFAULT_OPENPCDET_ROOT / "output" / "sunlakes_models" / "bevfusion" / "v03_high_lidar_b4_e20" / "ckpt" / "checkpoint_epoch_20.pth"
DEFAULT_BEV_FEATURE_KEY = "spatial_features_2d"

DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_DIR / "checkpoints"
DEFAULT_BEST_DIR = DEFAULT_CHECKPOINT_DIR / "best"
DEFAULT_BEST_CHECKPOINT = DEFAULT_BEST_DIR / "checkpoint.pt"
DEFAULT_BEST_CONFIG_JSON = DEFAULT_BEST_DIR / "config.json"
DEFAULT_BEST_METRICS_JSON = DEFAULT_BEST_DIR / "best_metrics.json"

DEFAULT_FEATURE_DIR = DEFAULT_WORK_DIR / "features"
DEFAULT_FEATURE_TRAIN_DIR = DEFAULT_FEATURE_DIR / "train"
DEFAULT_FEATURE_VAL_DIR = DEFAULT_FEATURE_DIR / "val"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"
DEFAULT_PLOT_DIR = DEFAULT_WORK_DIR / "plots"
DEFAULT_PREDICTION_DIR = DEFAULT_WORK_DIR / "predictions"
DEFAULT_METRIC_DIR = DEFAULT_WORK_DIR / "metrics"

DEFAULT_LOSS_HISTORY_JSON = DEFAULT_LOG_DIR / "loss_history.json"
DEFAULT_LOSS_HISTORY_JSONL = DEFAULT_LOG_DIR / "loss_history.jsonl"
DEFAULT_LOSS_CURVE_PNG = DEFAULT_PLOT_DIR / "loss_curves.png"
DEFAULT_TRAIN_SUMMARY_JSON = DEFAULT_LOG_DIR / "train_summary.json"
DEFAULT_PREDS_JSONL = DEFAULT_PREDICTION_DIR / "merged_predictions.jsonl"
DEFAULT_RESULTS_JSON = DEFAULT_METRIC_DIR / "metrics.json"

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MODEL_CACHE_DIR = LLM_ROOT / "cache"
DEFAULT_QFORMER_MODEL_ID = "bert-base-uncased"
DEFAULT_TOKENIZER_MODEL_MAX_LENGTH = 2048
DEFAULT_TOKENIZER_PADDING_SIDE = "right"
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in traffic-scene analysis. "
    "Answer the user's question using the BEV representation of the scene."
)
DEFAULT_VIEW_MODE = "global"
DEFAULT_VIEW_ID = 6

DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_RATIO = 0.10
DEFAULT_EXPECTED_GPUS = 4

MODEL_DEFAULTS = {
    "model_id": DEFAULT_MODEL_ID,
    "access_token": None,
    "cache_dir": str(DEFAULT_MODEL_CACHE_DIR),
    "use_lora": True,
    "num_query_token": 32,
    "bev_channels": 512,
    "cross_attention_freq": 2,
    "pos_encoding_scale": 0.06,
    "qformer_model_id": DEFAULT_QFORMER_MODEL_ID,
    "tokenizer_model_max_length": DEFAULT_TOKENIZER_MODEL_MAX_LENGTH,
    "tokenizer_padding_side": DEFAULT_TOKENIZER_PADDING_SIDE,
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    },
}

TRAINING_DEFAULTS = {
    "epochs": 3,
    "max_steps": -1,
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "warmup_ratio": 0.03,
    "num_workers": 0,
    "pin_memory": True,
    "seed": 42,
    "log_every": 10,
    "expected_gpus": DEFAULT_EXPECTED_GPUS,
}

INFERENCE_DEFAULTS = {
    "batch_size": 1,
    "num_workers": 4,
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
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
        "prepared_dir": REPO_ROOT / "data" / "intersection_bev_v6",
        "work_dir": VLM_ROOT / "work_dirs" / "bevLLM_v6",
    },
}


def resolve(path_like) -> Path:
    return Path(path_like).expanduser().resolve()


def worktree_paths(work_dir: Path) -> dict[str, Path]:
    root = resolve(work_dir)
    checkpoints = root / "checkpoints"
    best = checkpoints / "best"
    logs = root / "logs"
    plots = root / "plots"
    features = root / "features"
    predictions = root / "predictions"
    metrics = root / "metrics"
    return {
        "root": root,
        "checkpoints": checkpoints,
        "best": best,
        "features": features,
        "feature_train": features / "train",
        "feature_val": features / "val",
        "logs": logs,
        "plots": plots,
        "predictions": predictions,
        "metrics": metrics,
        "best_checkpoint": best / "checkpoint.pt",
        "best_config_json": best / "config.json",
        "best_metrics_json": best / "best_metrics.json",
        "loss_history_json": logs / "loss_history.json",
        "loss_history_jsonl": logs / "loss_history.jsonl",
        "loss_curve_png": plots / "loss_curves.png",
        "train_summary_json": logs / "train_summary.json",
        "merged_predictions": predictions / "merged_predictions.jsonl",
        "metrics_json": metrics / "metrics.json",
    }


def ensure_worktree_layout(work_dir: Path) -> dict[str, Path]:
    paths = worktree_paths(work_dir)
    for key in ("root", "checkpoints", "best", "features", "feature_train", "feature_val", "logs", "plots", "predictions", "metrics"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


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
        "qa_json": resolve(qa_json) if qa_json is not None else resolve(defaults["qa_json"]),
        "evaluator": resolve(evaluator) if evaluator is not None else resolve(defaults["evaluator"]),
        "prepared_dir": resolve(prepared_dir) if prepared_dir is not None else resolve(defaults["prepared_dir"]),
        "work_dir": resolve(work_dir) if work_dir is not None else resolve(defaults["work_dir"]),
    }
