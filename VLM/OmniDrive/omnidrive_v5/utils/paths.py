from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent
DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "omnidrive_v5" / "config" / "train.py"
DEFAULT_EVAL_CONFIG = REPO_ROOT / "omnidrive_v5" / "config" / "eval.py"
DEFAULT_EVALUATOR = LLM_ROOT / "utils" / "evaluate_intersection_vqa.py"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "sunlakes_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "omniDrive_v5"

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
        "work_dir": VLM_ROOT / "work_dirs" / "omniDrive_v6",
    },
}


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


DEFAULT_SIDECAR_JSONL = DEFAULT_PREPARED_DIR / "intersection_vqa_eval_sidecar.jsonl"
DATA_DIR = DEFAULT_PREPARED_DIR


def resolve(path_like) -> Path:
    return Path(path_like).expanduser().resolve()


def layout(work_dir) -> dict[str, Path]:
    root = resolve(work_dir)
    return {
        "root": root,
        "logs": root / "logs",
        "plots": root / "plots",
        "checkpoints": root / "checkpoints",
        "predictions": root / "predictions",
        "metrics": root / "metrics",
    }


def ensure_layout(work_dir) -> dict[str, Path]:
    paths = layout(work_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
