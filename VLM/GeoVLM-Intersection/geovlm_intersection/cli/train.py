from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from geovlm_intersection.backbones.qwen3_vl_adapter import load_qwen3_vl_runtime
from geovlm_intersection.config.common import (
    DEFAULT_DATASET_VERSION,
    ensure_worktree_layout,
    load_validated_features_manifest,
    resolve_dataset_version_paths,
)
from geovlm_intersection.data import GeoVLMFeatureDataset, build_info_index, collate_feature_batch, load_feature_payload
from geovlm_intersection.models.semantic_decoder import (
    SEMANTIC_GENERATION_SUBTEMPLATES,
    HYBRID_GENERATION_SUBTEMPLATES,
    STRUCTURED_FINAL_SUBTEMPLATES,
    build_frozen_semantic_decoder_runtime,
    compute_decoder_text_loss,
    generate_decoder_outputs,
)
from geovlm_intersection.pipeline_core import build_model_from_feature_dims, compute_batch_supervision_loss, save_checkpoint
from geovlm_intersection.rendering import compute_final_text_match, resolve_final_prediction
from geovlm_intersection.utils import dump_json, ensure, load_json


LOSS_COMPONENT_NAMES = (
    "object_selection",
    "object_type",
    "side",
    "motion_state",
    "risk_reason",
    "intersection_action",
    "side_action",
    "lane_action",
    "object_action",
    "lane_function",
    "camera",
    "binary",
    "count",
    "distance",
    "speed",
    "acceleration",
    "position_3d",
    "image_ref",
    "binary_speed_consistency",
    "decoder_text",
)

VALIDATION_METRIC_NAMES = (
    "semantic_text_score",
    "hybrid_text_score",
    "structured_text_score",
    "m_position",
    "m_image_ref",
)

STAGE1_COMPONENT_WEIGHTS = {
    "intersection_action": 0.0,
    "side_action": 0.0,
    "lane_action": 0.0,
    "count": 0.25,
    "distance": 0.5,
    "risk_reason": 0.5,
    "binary": 0.5,
    "lane_function": 0.25,
    "object_action": 0.75,
    "object_selection": 1.0,
    "object_type": 1.25,
    "side": 1.1,
    "camera": 1.25,
    "position_3d": 1.25,
    "image_ref": 1.0,
    "speed": 1.0,
    "acceleration": 1.0,
    "binary_speed_consistency": 0.75,
}

STAGE3_SUBTEMPLATE_WEIGHTS = {
    "1_1_1_fine_type": 2.0,
    "1_1_4_relative_neighbor_type": 2.0,
    "3_1_1_current_motion_state": 2.0,
    "3_4_2_nearest_conflict_participant": 2.0,
    "3_4_3_primary_risk_subject": 2.0,
    "4_2_1_speeding_risk": 2.0,
    "4_3_1_intersection_action": 1.5,
    "4_3_2_side_action": 1.5,
    "4_3_3_lane_action": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GeoVLM fusion core on extracted v5 features.")
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _is_distributed() -> bool:
    return _env_int("WORLD_SIZE", 1) > 1


def _init_distributed(timeout_seconds: int) -> tuple[bool, int, int, int, str]:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return False, world_size, rank, local_rank, "cpu"
    ensure(torch.cuda.is_available(), "Distributed GeoVLM training requires CUDA.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    return True, world_size, rank, local_rank, f"cuda:{local_rank}"


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _dist_barrier(local_rank: int) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _maybe_relaunch_with_torchrun(args: argparse.Namespace) -> None:
    if _is_distributed():
        return
    if os.getenv("GEOVLM_DISABLE_AUTO_TORCHRUN") == "1":
        return
    if not str(args.device).startswith("cuda"):
        return
    gpu_count = torch.cuda.device_count()
    nproc = min(4, gpu_count)
    if nproc <= 1:
        return
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        "29531",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)
    raise SystemExit(0)


def _build_model_from_index(index_path: Path, device: str) -> torch.nn.Module:
    index_rows = load_json(index_path)
    ensure(isinstance(index_rows, list) and index_rows, f"Feature index must be a non-empty JSON list: {index_path}")
    first_row = index_rows[0]
    ensure(
        first_row.get("frame_feature_path"),
        f"Feature index row must contain frame_feature_path: {index_path}",
    )
    first_feature = load_feature_payload(Path(first_row["frame_feature_path"]).resolve())
    qwen_runtime = load_qwen3_vl_runtime()
    return build_model_from_feature_dims(
        image_token_dim=int(first_feature["image_tokens"].shape[-1]),
        bev_token_dim=int(first_feature["bev_tokens"].shape[-1]),
        object_token_dim=int(first_feature["object_tokens"].shape[-1]),
        question_token_dim=int(qwen_runtime.text_hidden_size),
        device=device,
    )


def _run_epoch(
    *,
    model: torch.nn.Module,
    semantic_decoder_runtime,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    distributed: bool,
    show_progress: bool,
    progress_desc: str,
    component_weights: dict[str, float] | None = None,
    subtemplate_loss_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    component_sums = {name: 0.0 for name in LOSS_COMPONENT_NAMES}

    iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False) if show_progress else loader
    for batch in iterator:
        image_tokens = batch["image_tokens"].to(device)
        bev_tokens = batch["bev_tokens"].to(device)
        object_tokens = batch["object_tokens"].to(device)
        question_tokens = batch["question_tokens"].to(device)
        outputs = model(
            image_tokens=image_tokens,
            bev_tokens=bev_tokens,
            object_tokens=object_tokens,
            raw_object_tokens=batch["raw_object_tokens"].to(device),
            question_tokens=question_tokens,
            subtemplate_ids=batch["subtemplate_index"].to(device),
        )
        structured_loss_output = compute_batch_supervision_loss(
            outputs,
            batch["supervision"],
            device=device,
            component_weights=component_weights,
            subtemplate_loss_weights=subtemplate_loss_weights,
        )
        batch_total_loss = structured_loss_output.total_loss
        component_values = dict(structured_loss_output.components)
        decoder_loss_output = compute_decoder_text_loss(
            runtime=semantic_decoder_runtime,
            semantic_prefix_tokens=outputs["semantic_prefix_tokens"],
            prompt_texts=batch["decoder_prompt_text"],
            answer_texts=batch["answer_text"],
            subtemplates=batch["subtemplate"],
            batch_size=len(batch["question_id"]),
        )
        if decoder_loss_output.batch_mean_loss is not None:
            batch_total_loss = batch_total_loss + decoder_loss_output.batch_mean_loss
            component_values["decoder_text"] = decoder_loss_output.batch_mean_loss.detach()
        if training:
            optimizer.zero_grad(set_to_none=True)
            batch_total_loss.backward()
            optimizer.step()
        batch_size = len(batch["question_id"])
        total_loss_value = float(batch_total_loss.detach().cpu().item())
        total_loss += total_loss_value * batch_size
        total_items += batch_size
        for name, value in component_values.items():
            component_sums[name] += float(value.detach().cpu().item()) * batch_size
        if show_progress:
            iterator.set_postfix(loss=f"{total_loss_value:.4f}")

    ensure(total_items > 0, "Training loader produced zero samples.")
    if distributed:
        stats = torch.tensor(
            [total_loss, float(total_items), *[component_sums[name] for name in LOSS_COMPONENT_NAMES]],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_items = int(stats[1].item())
        component_sums = {
            name: float(stats[2 + index].item())
            for index, name in enumerate(LOSS_COMPONENT_NAMES)
        }
    mean_components = {
        name: value / total_items
        for name, value in component_sums.items()
        if value > 0.0
    }
    return total_loss / total_items, mean_components


def _resolve_stage_schedule(total_epochs: int) -> list[tuple[str, int, dict[str, float] | None, dict[str, float] | None]]:
    ensure(total_epochs > 0, f"epochs must be positive, got: {total_epochs}")
    if total_epochs == 1:
        return [("joint", 1, None, None)]
    if total_epochs == 2:
        return [("grounding_warmup", 1, STAGE1_COMPONENT_WEIGHTS, None), ("joint", 1, None, None)]
    base = [2, 4, 2]
    stage_names = ["grounding_warmup", "joint", "alignment"]
    raw = [total_epochs * value / sum(base) for value in base]
    counts = [int(value) for value in raw]
    for index in range(min(total_epochs, 3)):
        counts[index] = max(1, counts[index])
    while sum(counts) < total_epochs:
        index = max(range(3), key=lambda idx: raw[idx] - counts[idx])
        counts[index] += 1
    while sum(counts) > total_epochs:
        index = max(range(3), key=lambda idx: (counts[idx], -idx))
        if counts[index] > 1:
            counts[index] -= 1
        else:
            break
    component_by_stage = {
        "grounding_warmup": STAGE1_COMPONENT_WEIGHTS,
        "joint": None,
        "alignment": None,
    }
    subtemplate_by_stage = {
        "grounding_warmup": None,
        "joint": None,
        "alignment": STAGE3_SUBTEMPLATE_WEIGHTS,
    }
    return [
        (name, count, component_by_stage[name], subtemplate_by_stage[name])
        for name, count in zip(stage_names, counts)
        if count > 0
    ]


def _select_single_output(outputs: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            selected[key] = value[index : index + 1]
        else:
            selected[key] = value
    return selected


def _compute_selection_score(metrics: dict[str, float]) -> float:
    return float(metrics.get("overall_val_score", 0.0))


def _run_validation_epoch(
    *,
    model: torch.nn.Module,
    semantic_decoder_runtime,
    loader: DataLoader,
    device: str,
    distributed: bool,
    show_progress: bool,
    component_weights: dict[str, float] | None,
    subtemplate_loss_weights: dict[str, float] | None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    component_sums = {name: 0.0 for name in LOSS_COMPONENT_NAMES}
    counts = defaultdict(int)
    sums = defaultdict(float)

    progress = tqdm(loader, desc="Val", unit="batch", leave=False) if show_progress else loader
    with torch.inference_mode():
        for batch in progress:
            outputs = model(
                image_tokens=batch["image_tokens"].to(device),
                bev_tokens=batch["bev_tokens"].to(device),
                object_tokens=batch["object_tokens"].to(device),
                raw_object_tokens=batch["raw_object_tokens"].to(device),
                question_tokens=batch["question_tokens"].to(device),
                subtemplate_ids=batch["subtemplate_index"].to(device),
            )
            structured_loss_output = compute_batch_supervision_loss(
                outputs,
                batch["supervision"],
                device=device,
                component_weights=component_weights,
                subtemplate_loss_weights=subtemplate_loss_weights,
            )
            batch_total_loss = structured_loss_output.total_loss
            component_values = dict(structured_loss_output.components)
            decoder_loss_output = compute_decoder_text_loss(
                runtime=semantic_decoder_runtime,
                semantic_prefix_tokens=outputs["semantic_prefix_tokens"],
                prompt_texts=batch["decoder_prompt_text"],
                answer_texts=batch["answer_text"],
                subtemplates=batch["subtemplate"],
                batch_size=len(batch["question_id"]),
            )
            if decoder_loss_output.batch_mean_loss is not None:
                batch_total_loss = batch_total_loss + decoder_loss_output.batch_mean_loss
                component_values["decoder_text"] = decoder_loss_output.batch_mean_loss.detach()
            batch_size = len(batch["question_id"])
            total_loss += float(batch_total_loss.detach().cpu().item()) * batch_size
            total_items += batch_size
            for name, value in component_values.items():
                component_sums[name] += float(value.detach().cpu().item()) * batch_size

            decoder_outputs = generate_decoder_outputs(
                runtime=semantic_decoder_runtime,
                semantic_prefix_tokens=outputs["semantic_prefix_tokens"],
                prompt_texts=batch["decoder_prompt_text"],
                subtemplates=batch["subtemplate"],
            )

            for index, supervision in enumerate(batch["supervision"]):
                subtemplate = supervision.subtemplate
                final_result = resolve_final_prediction(
                    subtemplate=subtemplate,
                    outputs=_select_single_output(outputs, index),
                    decoder_raw_output=decoder_outputs[index],
                )
                decoded_payload = final_result.decoded_payload or {}
                target = batch["prepared_record"][index].get("structured_targets") or {}

                if final_result.prediction:
                    text_match = compute_final_text_match(batch["answer_text"][index], final_result.prediction)
                    if subtemplate in SEMANTIC_GENERATION_SUBTEMPLATES:
                        sums["semantic_text_score"] += text_match
                        counts["semantic_text_score"] += 1
                    elif subtemplate in HYBRID_GENERATION_SUBTEMPLATES:
                        sums["hybrid_text_score"] += text_match
                        counts["hybrid_text_score"] += 1
                    elif subtemplate in STRUCTURED_FINAL_SUBTEMPLATES:
                        sums["structured_text_score"] += text_match
                        counts["structured_text_score"] += 1

                decoded_position = decoded_payload.get("position_3d")
                if supervision.position_3d is not None and isinstance(decoded_position, (list, tuple)) and len(decoded_position) == 2:
                    dx = float(decoded_position[0]) - supervision.position_3d[0]
                    dy = float(decoded_position[1]) - supervision.position_3d[1]
                    sums["m_position"] += float((dx * dx + dy * dy) ** 0.5 <= 8.0)
                    counts["m_position"] += 1
                decoded_image_ref = decoded_payload.get("image_ref")
                if supervision.image_ref is not None and isinstance(decoded_image_ref, (list, tuple)) and len(decoded_image_ref) == 2:
                    dx = float(decoded_image_ref[0]) - supervision.image_ref[0]
                    dy = float(decoded_image_ref[1]) - supervision.image_ref[1]
                    sums["m_image_ref"] += float((dx * dx + dy * dy) ** 0.5 <= 150.0)
                    counts["m_image_ref"] += 1

            if show_progress:
                progress.set_postfix(loss=f"{float(batch_total_loss.detach().cpu().item()):.4f}")

    if distributed:
        stats = torch.tensor(
            [total_loss, float(total_items), *[component_sums[name] for name in LOSS_COMPONENT_NAMES]],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_items = int(stats[1].item())
        component_sums = {
            name: float(stats[2 + index].item())
            for index, name in enumerate(LOSS_COMPONENT_NAMES)
        }
        metric_stats = torch.tensor(
            [
                value
                for name in VALIDATION_METRIC_NAMES
                for value in (sums[name], float(counts[name]))
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metric_stats, op=dist.ReduceOp.SUM)
        for index, name in enumerate(VALIDATION_METRIC_NAMES):
            sums[name] = float(metric_stats[2 * index].item())
            counts[name] = int(metric_stats[2 * index + 1].item())
    ensure(total_items > 0, "Validation loader produced zero samples.")
    mean_components = {
        name: value / total_items
        for name, value in component_sums.items()
        if value > 0.0
    }
    metrics = {
        key: sums[key] / counts[key]
        for key in counts
        if counts[key] > 0
    }
    numeric_parts = [metrics[key] for key in ("m_position", "m_image_ref") if key in metrics]
    if numeric_parts:
        metrics["structured_numeric_score"] = sum(numeric_parts) / len(numeric_parts)
    overall_parts = [
        metrics[key]
        for key in ("semantic_text_score", "hybrid_text_score", "structured_text_score", "structured_numeric_score")
        if key in metrics
    ]
    if overall_parts:
        metrics["overall_val_score"] = sum(overall_parts) / len(overall_parts)
    metrics["selection_score"] = _compute_selection_score(metrics)
    return total_loss / total_items, mean_components, metrics


def main() -> None:
    args = parse_args()
    _maybe_relaunch_with_torchrun(args)
    distributed, world_size, rank, local_rank, ddp_device = _init_distributed(args.ddp_timeout_seconds)
    device = ddp_device if distributed else args.device
    try:
        resolved = resolve_dataset_version_paths(
            args.dataset_version,
            prepared_dir=args.prepared_dir,
            work_dir=args.work_dir,
        )
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        manifest = load_json(prepared_dir / "split_manifest.json")
        ensure(isinstance(manifest, dict), f"split_manifest.json must be an object: {prepared_dir / 'split_manifest.json'}")
        ensure(
            manifest.get("dataset_version") == resolved["dataset_version"],
            f"Prepared dataset_version={manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
        )
        worktree = ensure_worktree_layout(Path(resolved["work_dir"]).resolve())
        load_validated_features_manifest(
            worktree,
            dataset_version=resolved["dataset_version"],
            required_splits=("train", "val"),
        )

        info_index = build_info_index()
        train_dataset = GeoVLMFeatureDataset(
            prepared_dir=prepared_dir,
            work_dir=worktree["work_dir"],
            split="train",
            info_index=info_index,
        )
        val_dataset = GeoVLMFeatureDataset(
            prepared_dir=prepared_dir,
            work_dir=worktree["work_dir"],
            split="val",
            info_index=info_index,
        )

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_feature_batch,
            pin_memory=device.startswith("cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=(
                DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                if distributed
                else None
            ),
            num_workers=args.num_workers,
            collate_fn=collate_feature_batch,
            pin_memory=device.startswith("cuda"),
        )

        model = _build_model_from_index(worktree["feature_index_train"], device)
        semantic_decoder_runtime = build_frozen_semantic_decoder_runtime(device=device)
        raw_model = model
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                # GeoVLM uses sparse template-specific supervision. Many heads are
                # intentionally inactive on a given rank/batch, so DDP must track
                # unused parameters instead of assuming every parameter receives
                # gradient every iteration.
                find_unused_parameters=True,
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        history: list[dict[str, object]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        global_step = 0
        stage_schedule = _resolve_stage_schedule(args.epochs)
        train_args = {
            "dataset_version": resolved["dataset_version"],
            "prepared_dir": str(prepared_dir),
            "work_dir": str(worktree["work_dir"]),
            "epochs": args.epochs,
            "stage_schedule": [
                {
                    "name": name,
                    "epochs": epoch_count,
                    "component_weights": component_weights or {},
                    "subtemplate_loss_weights": subtemplate_weights or {},
                }
                for name, epoch_count, component_weights, subtemplate_weights in stage_schedule
            ],
            "per_device_batch_size": args.batch_size,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "learning_rate": args.learning_rate,
            "device": device,
            "num_workers": args.num_workers,
            "distributed": distributed,
            "training_scope": "fusion core + structured heads + frozen qwen3-vl semantic decoder prefixing",
        }
        if _is_main_process(rank):
            dump_json(worktree["train_args_json"], train_args)

        epoch_counter = 0
        best_selection_score = float("-inf")
        for stage_name, stage_epoch_count, component_weights, subtemplate_loss_weights in stage_schedule:
            for stage_epoch_index in range(stage_epoch_count):
                epoch_counter += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch_counter)
                train_loss, train_components = _run_epoch(
                    model=model,
                    semantic_decoder_runtime=semantic_decoder_runtime,
                    loader=train_loader,
                    device=device,
                    optimizer=optimizer,
                    distributed=distributed,
                    show_progress=_is_main_process(rank),
                    progress_desc=f"Train {stage_name} E{epoch_counter}",
                    component_weights=component_weights,
                    subtemplate_loss_weights=subtemplate_loss_weights,
                )
                stage_final_epoch = stage_epoch_index == stage_epoch_count - 1
                val_loss: float | None = None
                val_components: dict[str, float] = {}
                val_metrics: dict[str, float] = {}
                if stage_final_epoch:
                    val_loss, val_components, val_metrics = _run_validation_epoch(
                        model=model,
                        semantic_decoder_runtime=semantic_decoder_runtime,
                        loader=val_loader,
                        device=device,
                        distributed=distributed,
                        show_progress=_is_main_process(rank),
                        component_weights=component_weights,
                        subtemplate_loss_weights=subtemplate_loss_weights,
                    )
                global_step += len(train_loader)
                if _is_main_process(rank):
                    history.append(
                        {
                            "epoch": epoch_counter,
                            "stage": stage_name,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_components": train_components,
                            "val_components": val_components,
                            "val_metrics": val_metrics,
                            "selection_score": val_metrics.get("selection_score"),
                        }
                    )
                    save_checkpoint(
                        worktree["last_checkpoint"],
                        model=raw_model,
                        config=raw_model.config,
                        epoch=epoch_counter,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        dataset_version=resolved["dataset_version"],
                            extra_state={"train_args": train_args},
                    )
                    selection_score = float(val_metrics.get("selection_score", float("-inf")))
                    if stage_final_epoch and val_loss is not None and val_loss < best_val_loss:
                        best_selection_score = selection_score
                        best_val_loss = val_loss
                        best_epoch = epoch_counter
                        save_checkpoint(
                            worktree["best_checkpoint"],
                            model=raw_model,
                            config=raw_model.config,
                            epoch=epoch_counter,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            dataset_version=resolved["dataset_version"],
                            extra_state={"train_args": train_args, "best_selection_score": best_selection_score},
                        )
                if distributed:
                    _dist_barrier(local_rank)

        if _is_main_process(rank):
            dump_json(
                worktree["train_summary_json"],
                {
                    "dataset_version": resolved["dataset_version"],
                    "epochs_completed": args.epochs,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "world_size": world_size,
                    "history": history,
                },
            )
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
