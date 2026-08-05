from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent

DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_PREPARED_DIR = DEFAULT_DATA_DIR / "intersectionqa_v5_3frame"

DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "traffiXQwen_v5"
DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_DIR / "checkpoints"
DEFAULT_OUTPUT_ROOT = DEFAULT_CHECKPOINT_DIR
DEFAULT_MODEL_OUTPUT_DIR = DEFAULT_CHECKPOINT_DIR / "best"
DEFAULT_RESULTS_DIR = DEFAULT_WORK_DIR / "predictions"
DEFAULT_EVAL_DIR = DEFAULT_WORK_DIR / "metrics"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"
DEFAULT_PLOT_DIR = DEFAULT_WORK_DIR / "plots"

DEFAULT_MODEL_NAME_OR_PATH = REPO_ROOT / "weights" / "Qwen2-0.5B"
DEFAULT_VISION_TOWER = REPO_ROOT / "weights" / "siglip-so400m"
DEFAULT_MM_PROJECTOR = REPO_ROOT / "weights" / "mm_projector.bin"

DEFAULT_TEMPORAL_WINDOW = 3
DEFAULT_VIEWS = ("north", "south", "east", "west")
DEFAULT_VAL_SCENES = 20
DEFAULT_VAL_RATIO = 0.10
DEFAULT_PROMPT_VERSION = "qwen_intersection_qa"
DEFAULT_MODEL_MAX_LENGTH = 12288
DEFAULT_IMAGE_TOKEN_COST = 730
DEFAULT_FINETUNE_MODE = "adapter_lm_lora"
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_TRAIN_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_TRAIN_EPOCHS = 3
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_DATALOADER_NUM_WORKERS = 4
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 500
DEFAULT_SAVE_STEPS = 500
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_INFERENCE_BATCH_SIZE = 1

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
        "prepared_dir": DEFAULT_DATA_DIR / "intersectionqa_v6_3frame",
        "work_dir": VLM_ROOT / "work_dirs" / "traffiXQwen_v6",
    },
}


def resolve(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


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
