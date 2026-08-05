import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from omnidrive_v5.utils.paths import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_INFO_PKL,
    DEFAULT_TRAIN_CONFIG,
    REPO_ROOT,
    ensure_layout,
    layout,
    resolve,
    resolve_dataset_version_paths,
)


WORK_MANIFEST_NAME = "dataset_manifest.json"


def parse_pipeline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OmniDrive SunLakes V5/V6 training/evaluation pipeline.")
    parser.add_argument("--stage", choices=["prepare", "train", "forward", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--sidecar-jsonl", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--validate-during-train", action="store_true")
    parser.add_argument("--train-cfg-options", nargs="*", default=None)
    parser.add_argument("--eval-cfg-options", nargs="*", default=None)
    parser.add_argument("--eval-split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--sim-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--skip-semantic-metrics", action="store_true")
    parser.add_argument("--skip-env-check", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[sunlakes-pipeline] {message}", flush=True)


def run_command(command, cwd, env):
    log("Running command:")
    log(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_runtime_env(resolved: dict[str, Path | str]) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    env["OMNIDRIVE_DATASET_VERSION"] = str(resolved["dataset_version"])
    env["OMNIDRIVE_QA_JSON"] = str(resolve(resolved["qa_json"]))
    env["OMNIDRIVE_PREPARED_DIR"] = str(resolve(resolved["prepared_dir"]))
    env["OMNIDRIVE_WORK_DIR"] = str(resolve(resolved["work_dir"]))
    env["OMNIDRIVE_EVALUATOR"] = str(resolve(resolved["evaluator"]))
    return env


def resolve_checkpoint_path(requested_checkpoint, work_dir):
    checkpoint_dir = layout(work_dir)["checkpoints"]
    if requested_checkpoint is not None:
        checkpoint = resolve(requested_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    for candidate_name in ("best.pth", "last.pth"):
        candidate = checkpoint_dir / candidate_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found under {checkpoint_dir}. Expected best.pth or last.pth."
    )


def work_manifest_path(paths: dict[str, Path]) -> Path:
    return paths["root"] / WORK_MANIFEST_NAME


def expected_work_manifest(resolved: dict[str, Path | str]) -> dict[str, str]:
    return {
        "dataset_version": str(resolved["dataset_version"]),
        "qa_json": str(resolve(resolved["qa_json"])),
        "prepared_dir": str(resolve(resolved["prepared_dir"])),
        "work_dir": str(resolve(resolved["work_dir"])),
        "evaluator": str(resolve(resolved["evaluator"])),
    }


def ensure_work_manifest(paths: dict[str, Path], resolved: dict[str, Path | str]) -> None:
    manifest_path = work_manifest_path(paths)
    expected = expected_work_manifest(resolved)
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("dataset_version") != expected["dataset_version"]:
            raise ValueError(
                "OmniDrive work-dir dataset_version mismatch: "
                f"expected {expected['dataset_version']}, got {manifest.get('dataset_version')}"
            )
        return

    artifact_dirs = [paths["checkpoints"], paths["predictions"], paths["metrics"], paths["logs"], paths["plots"]]
    if any(path.exists() and any(path.iterdir()) for path in artifact_dirs):
        raise ValueError(
            "OmniDrive work_dir already contains artifacts but no dataset manifest. "
            f"Refusing to reuse {paths['root']} without an explicit version record."
        )
    dump_json(manifest_path, expected)


def build_prepare_command(args, resolved):
    return [
        args.python,
        "-m",
        "omnidrive_v5.cli.prepare",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--qa-json",
        str(resolve(resolved["qa_json"])),
        "--info-pkl",
        str(resolve(args.info_pkl)),
        "--output-dir",
        str(resolve(resolved["prepared_dir"])),
    ]


def build_env_check_command(args, resolved):
    return [
        args.python,
        "-m",
        "omnidrive_v5.cli.check_env",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--qa-json",
        str(resolve(resolved["qa_json"])),
        "--info-pkl",
        str(resolve(args.info_pkl)),
        "--prepared-dir",
        str(resolve(resolved["prepared_dir"])),
        "--work-dir",
        str(resolve(resolved["work_dir"])),
        "--evaluator",
        str(resolve(resolved["evaluator"])),
        "--require-cuda",
        "--require-prepared",
    ]


def build_train_command(args, work_dir):
    if args.nproc_per_node > 1:
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            str(REPO_ROOT / "tools" / "train.py"),
            str(resolve(args.train_config)),
            "--launcher",
            "pytorch",
            "--work-dir",
            str(resolve(work_dir)),
        ]
    else:
        command = [
            args.python,
            str(REPO_ROOT / "tools" / "train.py"),
            str(resolve(args.train_config)),
            "--work-dir",
            str(resolve(work_dir)),
        ]
    if not args.validate_during_train:
        command.append("--no-validate")
    if args.train_cfg_options:
        command.append("--cfg-options")
        command.extend(args.train_cfg_options)
    return command


def build_forward_command(args, checkpoint, work_dir):
    prediction_dir = layout(work_dir)["predictions"]
    cfg_options = [
        f"work_dir={str(resolve(work_dir)).replace(os.sep, '/')}",
        f"model.save_path={str(prediction_dir).replace(os.sep, '/')}",
    ]
    if args.eval_cfg_options:
        cfg_options.extend(args.eval_cfg_options)

    if args.nproc_per_node > 1:
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port + 1}",
            str(REPO_ROOT / "tools" / "test.py"),
            str(resolve(args.eval_config)),
            str(checkpoint),
            "--launcher",
            "pytorch",
            "--eval",
            "bbox",
        ]
    else:
        command = [
            args.python,
            str(REPO_ROOT / "tools" / "test.py"),
            str(resolve(args.eval_config)),
            str(checkpoint),
            "--launcher",
            "none",
            "--eval",
            "bbox",
        ]
    if cfg_options:
        command.append("--cfg-options")
        command.extend(cfg_options)
    return command


def build_metric_command(args, resolved, work_dir):
    paths = layout(work_dir)
    sidecar_jsonl = resolve(args.sidecar_jsonl) if args.sidecar_jsonl is not None else resolve(resolved["prepared_dir"]) / "intersection_vqa_eval_sidecar.jsonl"
    command = [
        args.python,
        "-m",
        "omnidrive_v5.cli.evaluate",
        "--dataset-version",
        str(resolved["dataset_version"]),
        "--prepared-dir",
        str(resolve(resolved["prepared_dir"])),
        "--work-dir",
        str(resolve(resolved["work_dir"])),
        "--evaluator",
        str(resolve(resolved["evaluator"])),
        "--pred-dir",
        str(paths["predictions"]),
        "--sidecar-jsonl",
        str(sidecar_jsonl),
        "--output-dir",
        str(paths["metrics"]),
        "--split",
        args.eval_split,
        "--bertscore-model",
        args.bertscore_model,
        "--sim-model",
        args.sim_model,
        "--batch-size",
        str(args.eval_batch_size),
        "--device",
        args.eval_device,
    ]
    if args.skip_semantic_metrics:
        command.append("--skip-semantic-metrics")
    return command


def prepared_artifacts_need_refresh(args, resolved):
    prepared_dir = resolve(resolved["prepared_dir"])
    required_paths = (
        prepared_dir / "infos_train.pkl",
        prepared_dir / "infos_val.pkl",
        prepared_dir / "sidecar_train.jsonl",
        prepared_dir / "sidecar_val.jsonl",
        prepared_dir / "intersection_vqa_eval_sidecar.jsonl",
        prepared_dir / "split_manifest.json",
    )
    if any(not path.exists() for path in required_paths):
        return True, "prepared artifacts are missing"

    manifest = load_json(prepared_dir / "split_manifest.json")
    if manifest.get("dataset_version") != resolved["dataset_version"]:
        return True, (
            "prepared artifacts were built for "
            f"{manifest.get('dataset_version')} instead of {resolved['dataset_version']}"
        )

    source_paths = (resolve(resolved["qa_json"]), resolve(args.info_pkl))
    newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    oldest_output_mtime = min(path.stat().st_mtime for path in required_paths)
    if oldest_output_mtime < newest_source_mtime:
        return True, "prepared artifacts are older than current source files"

    with resolve(resolved["qa_json"]).open("r", encoding="utf-8") as file:
        qa_payload = json.load(file)
    raw_qids = {str(row["question_id"]) for row in qa_payload["qa_pairs"]}
    sidecar_paths = (
        prepared_dir / "sidecar_train.jsonl",
        prepared_dir / "sidecar_val.jsonl",
        prepared_dir / "intersection_vqa_eval_sidecar.jsonl",
    )
    for sidecar_path in sidecar_paths:
        with sidecar_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                question_id = str(json.loads(line)["question_id"])
                if question_id not in raw_qids:
                    return True, f"prepared artifact {sidecar_path.name} contains stale question_id={question_id}"
    return False, None


def clear_prediction_dir(work_dir):
    prediction_dir = layout(work_dir)["predictions"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for path in prediction_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def ensure_prepared_artifacts(args, resolved, env):
    needs_refresh, reason = prepared_artifacts_need_refresh(args, resolved)
    if needs_refresh:
        log(f"Refreshing prepared {resolved['dataset_version']} artifacts because {reason}.")
        run_command(build_prepare_command(args, resolved), REPO_ROOT, env)
        log("Preparation refresh finished.")


def main() -> None:
    args = parse_pipeline_args()
    stage = args.stage
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        evaluator=args.evaluator,
        prepared_dir=args.prepared_dir,
        work_dir=args.work_dir,
    )
    paths = ensure_layout(resolved["work_dir"])
    ensure_work_manifest(paths, resolved)
    env = build_runtime_env(resolved)

    if stage in {"prepare", "all"}:
        log(f"Preparing {resolved['dataset_version']} data artifacts.")
        run_command(build_prepare_command(args, resolved), REPO_ROOT, env)
        log("Preparation stage finished.")

    if stage in {"train", "forward", "evaluate", "all"}:
        ensure_prepared_artifacts(args, resolved, env)

    if stage in {"train", "forward", "all"} and not args.skip_env_check:
        log(f"Checking runtime environment for {resolved['dataset_version']}.")
        run_command(build_env_check_command(args, resolved), REPO_ROOT, env)
        log("Environment check finished.")

    if stage in {"train", "all"}:
        log(f"Starting training stage for {resolved['dataset_version']}.")
        run_command(build_train_command(args, paths["root"]), REPO_ROOT, env)
        log("Training stage finished.")

    if stage in {"forward", "evaluate", "all"}:
        checkpoint = resolve_checkpoint_path(args.checkpoint, paths["root"])
        log(f"Using checkpoint: {checkpoint}")
        log("Clearing stale prediction outputs.")
        clear_prediction_dir(paths["root"])
        log("Starting forward pass.")
        run_command(build_forward_command(args, checkpoint, paths["root"]), REPO_ROOT, env)
        log("Forward pass finished.")

    if stage in {"evaluate", "all"}:
        log(f"Computing {resolved['dataset_version']} metrics.")
        run_command(build_metric_command(args, resolved, paths["root"]), REPO_ROOT, env)
        log("Evaluation stage finished.")


if __name__ == "__main__":
    main()
