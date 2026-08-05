from __future__ import annotations

from pathlib import Path
from typing import Any

from geovlm_intersection.utils import ensure, load_json

PACKAGE_NAME = "geovlm_intersection"
REPO_ROOT = Path(__file__).resolve().parents[2]
VLM_ROOT = REPO_ROOT.parent
LLM_ROOT = VLM_ROOT.parent
OPENPCDET_ROOT = Path("/home/suolab/OpenPCDet")
HF_CACHE_ROOT = LLM_ROOT / "cache" / "huggingface" / "hub"

DEFAULT_DATASET_VERSION = "v5"
DEFAULT_QA_JSON = LLM_ROOT / "intersection_qa_pairs_v5.json"
DEFAULT_INFO_PKL = LLM_ROOT / "sunlakes_infos_trainval.pkl"
DEFAULT_SOURCE_PREPARED_DIR = VLM_ROOT / "NuScenes-QA" / "data" / "intersection_nuscenesqa_v5"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "intersection_geovlm_v5"
DEFAULT_WORK_DIR = VLM_ROOT / "work_dirs" / "geovlm_intersection_v5"
DEFAULT_ENV_NAME = "geovlm-cu128"
DEFAULT_QWEN3_VL_MODEL_DIR = (
    HF_CACHE_ROOT / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
DEFAULT_LION_CONFIG = OPENPCDET_ROOT / "tools" / "cfgs" / "sunlakes_models" / "lion.yaml"
DEFAULT_LION_QUALITY = "high"
DEFAULT_LION_CHECKPOINTS = {
    "low": OPENPCDET_ROOT / "output" / "sunlakes_models" / "lion" / "v03_low_lidar_b4_e20" / "ckpt" / "checkpoint_epoch_20.pth",
    "mid": OPENPCDET_ROOT / "output" / "sunlakes_models" / "lion" / "v03_mid_lidar_b4_e20" / "ckpt" / "checkpoint_epoch_20.pth",
    "high": OPENPCDET_ROOT / "output" / "sunlakes_models" / "lion" / "v03_high_lidar_b4_e20" / "ckpt" / "checkpoint_epoch_20.pth",
}
FRAME_ONLY_FEATURE_STORAGE = "frame_only_runtime_question_text"
FEATURE_LAYOUT_VERSION = "coarse_bev_v2_enriched_object_v2"

DATASET_VERSION_DEFAULTS = {
    "v5": {
        "qa_json": DEFAULT_QA_JSON,
        "source_prepared_dir": DEFAULT_SOURCE_PREPARED_DIR,
        "prepared_dir": DEFAULT_PREPARED_DIR,
        "evaluator": LLM_ROOT / "utils" / "evaluate_intersection_vqa.py",
        "work_dir": DEFAULT_WORK_DIR,
    },
}


def validate_dataset_version(dataset_version: str) -> str:
    version = str(dataset_version).strip().lower()
    if version not in DATASET_VERSION_DEFAULTS:
        supported = ", ".join(sorted(DATASET_VERSION_DEFAULTS))
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Expected one of: {supported}")
    return version


def resolve_dataset_version_paths(
    dataset_version: str,
    *,
    qa_json: str | Path | None = None,
    source_prepared_dir: str | Path | None = None,
    prepared_dir: str | Path | None = None,
    evaluator: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Path | str]:
    version = validate_dataset_version(dataset_version)
    defaults = DATASET_VERSION_DEFAULTS[version]
    return {
        "dataset_version": version,
        "qa_json": Path(qa_json).resolve() if qa_json is not None else Path(defaults["qa_json"]).resolve(),
        "source_prepared_dir": Path(source_prepared_dir).resolve()
        if source_prepared_dir is not None
        else Path(defaults["source_prepared_dir"]).resolve(),
        "prepared_dir": Path(prepared_dir).resolve() if prepared_dir is not None else Path(defaults["prepared_dir"]).resolve(),
        "evaluator": Path(evaluator).resolve() if evaluator is not None else Path(defaults["evaluator"]).resolve(),
        "work_dir": Path(work_dir).resolve() if work_dir is not None else Path(defaults["work_dir"]).resolve(),
    }


def ensure_worktree_layout(work_dir: Path) -> dict[str, Path]:
    resolved = work_dir.resolve()
    features = resolved / "features"
    checkpoints = resolved / "checkpoints"
    predictions = resolved / "predictions"
    metrics = resolved / "metrics"
    logs = resolved / "logs"
    resolved.mkdir(parents=True, exist_ok=True)
    for directory in (features, checkpoints, predictions, metrics, logs):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "work_dir": resolved,
        "features": features,
        "checkpoints": checkpoints,
        "predictions": predictions,
        "metrics": metrics,
        "logs": logs,
        "features_manifest": features / "feature_manifest.json",
        "feature_index_train": features / "train_index.json",
        "feature_index_val": features / "val_index.json",
        "feature_index_val_eval": features / "val_eval_index.json",
        "best_checkpoint": checkpoints / "best.pt",
        "last_checkpoint": checkpoints / "last.pt",
        "train_args_json": logs / "train_args.json",
        "train_summary_json": logs / "train_summary.json",
        "predictions_rank0": predictions / "predictions_rank0.jsonl",
        "predictions_merged": predictions / "merged_predictions.jsonl",
        "forward_run_json": predictions / "forward_run.json",
        "metrics_json": metrics / "metrics.json",
        "metrics_run_json": metrics / "metrics_run.json",
        "partial_sidecar_jsonl": metrics / "_partial_sidecar.jsonl",
    }


def validate_features_manifest_payload(
    manifest: object,
    *,
    dataset_version: str,
    features_dir: Path | None = None,
    required_splits: tuple[str, ...] = (),
) -> dict[str, Any]:
    ensure(isinstance(manifest, dict), "Feature manifest must be an object.")
    ensure(
        manifest.get("dataset_version") == dataset_version,
        f"Feature manifest dataset_version={manifest.get('dataset_version')} does not match requested {dataset_version}.",
    )
    ensure(
        manifest.get("feature_storage") == FRAME_ONLY_FEATURE_STORAGE,
        f"Feature manifest is incomplete or stale; expected top-level feature_storage={FRAME_ONLY_FEATURE_STORAGE}. "
        "Re-run extract with the current code.",
    )
    ensure(
        manifest.get("feature_layout_version") == FEATURE_LAYOUT_VERSION,
        f"Feature manifest is incomplete or stale; expected feature_layout_version={FEATURE_LAYOUT_VERSION}. "
        "Re-run extract with the current code.",
    )
    split_aliases = manifest.get("split_aliases")
    ensure(
        isinstance(split_aliases, dict),
        "Feature manifest is incomplete or stale; expected top-level split_aliases object.",
    )
    splits = manifest.get("splits")
    ensure(isinstance(splits, dict), "Feature manifest is incomplete or stale; expected top-level splits object.")
    for split in required_splits:
        ensure(
            isinstance(splits.get(split), dict),
            f"Feature manifest is missing required split metadata for {split}. Re-run extract with the current code.",
        )

    if "val" in splits or "val_eval" in splits or split_aliases.get("val_eval") is not None:
        ensure(
            split_aliases.get("val_eval") == "val",
            "Feature layout is stale; val_eval must alias val. Re-run extract with the current code.",
        )
        val_meta = splits.get("val")
        val_eval_meta = splits.get("val_eval")
        ensure(
            isinstance(val_meta, dict) and isinstance(val_eval_meta, dict),
            "Feature layout is stale; expected both val and val_eval split metadata.",
        )
        ensure(
            val_eval_meta.get("alias_of") == "val",
            "Feature layout is stale; val_eval metadata must declare alias_of=val.",
        )
        ensure(
            val_eval_meta.get("physical_split") == "val",
            "Feature layout is stale; val_eval physical_split must be val.",
        )
        ensure(
            Path(str(val_meta.get("frame_feature_dir"))).resolve()
            == Path(str(val_eval_meta.get("frame_feature_dir"))).resolve(),
            "Feature layout is stale; val_eval must reuse val frame_feature_dir.",
        )
        if features_dir is not None:
            stale_val_eval_frame_dir = features_dir.resolve() / "val_eval" / "frames"
            if stale_val_eval_frame_dir.is_dir():
                has_payloads = any(path.is_file() for path in stale_val_eval_frame_dir.iterdir())
                ensure(
                    not has_payloads,
                    "Feature layout is stale; val_eval must alias val and should not have its own frame directory. "
                    "Remove stale features/val_eval/frames and re-run extract.",
                )
    return manifest


def load_validated_features_manifest(
    worktree: dict[str, Path],
    *,
    dataset_version: str,
    required_splits: tuple[str, ...] = (),
) -> dict[str, Any]:
    manifest = load_json(worktree["features_manifest"])
    return validate_features_manifest_payload(
        manifest,
        dataset_version=dataset_version,
        features_dir=worktree["features"],
        required_splits=required_splits,
    )


def resolve_feature_split_alias(features_manifest: dict[str, Any], split: str) -> str:
    split_aliases = features_manifest.get("split_aliases")
    ensure(isinstance(split_aliases, dict), "Feature manifest must contain split_aliases.")
    alias = split_aliases.get(split)
    return str(alias) if isinstance(alias, str) and alias else split
