from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from traffixqwen_v5.config.common import (
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_MAX_LENGTH,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_PREPARED_DIR,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    resolve_dataset_version_paths,
)
from traffixqwen_v5.data.common import ensure, load_json
from traffixqwen_v5.data.dataset import StrictIntersectionV5EvalDataset, StrictIntersectionV5InferenceCollator
from traffixqwen_v5.utils.dist import cleanup_distributed, init_distributed


def clean_prediction_text(text: str) -> str:
    cleaned = text
    if "<|im_end|>" in cleaned:
        cleaned = cleaned.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in cleaned:
        cleaned = cleaned.split("<|im_start|>", 1)[0]
    cleaned = cleaned.replace("<|endoftext|>", "")
    return cleaned.strip()


def resolve_run_id(rank: int) -> str:
    run_id_holder: List[Any] = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(run_id_holder, src=0)
    run_id = run_id_holder[0]
    ensure(isinstance(run_id, str) and run_id, "Failed to resolve a valid TraffiX-Qwen V5 forward run id.")
    return run_id


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    ensure(dtype_name in dtype_map, f"Unsupported TraffiX-Qwen V5 forward dtype: {dtype_name}")
    return dtype_map[dtype_name]


def infer_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def tensor_has_non_finite(tensor: torch.Tensor | None) -> bool:
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return not torch.isfinite(tensor).all().item()


def summarize_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not any(char.isalnum() for char in stripped)


def collect_worker_outputs(output_dir: Path, world_size: int, run_id: str) -> tuple[List[Path], int]:
    worker_outputs: List[Path] = []
    total_generated = 0
    expected_dataset_version: str | None = None
    for worker_rank in range(world_size):
        worker_file = output_dir / f"predictions_rank{worker_rank}.jsonl"
        worker_meta_path = output_dir / f"predictions_rank{worker_rank}.meta.json"
        ensure(worker_file.is_file(), f"Missing rank output file: {worker_file}")
        ensure(worker_meta_path.is_file(), f"Missing rank metadata file: {worker_meta_path}")
        worker_meta = load_json(worker_meta_path)
        ensure(
            isinstance(worker_meta, dict),
            f"Invalid rank metadata payload in {worker_meta_path}",
        )
        ensure(
            worker_meta.get("run_id") == run_id,
            "TraffiX-Qwen V5 forward detected stale rank output metadata. "
            f"Expected run_id={run_id}, got {worker_meta.get('run_id')} in {worker_meta_path}",
        )
        ensure(
            worker_meta.get("rank") == worker_rank,
            f"Rank metadata mismatch in {worker_meta_path}: expected rank {worker_rank}, "
            f"got {worker_meta.get('rank')}",
        )
        ensure(
            worker_meta.get("world_size") == world_size,
            f"World-size mismatch in {worker_meta_path}: expected {world_size}, "
            f"got {worker_meta.get('world_size')}",
        )
        dataset_version = worker_meta.get("dataset_version")
        ensure(isinstance(dataset_version, str) and dataset_version, f"Invalid dataset_version in {worker_meta_path}: {dataset_version}")
        if expected_dataset_version is None:
            expected_dataset_version = dataset_version
        else:
            ensure(
                dataset_version == expected_dataset_version,
                f"Prediction shard dataset_version mismatch in {worker_meta_path}: expected {expected_dataset_version}, got {dataset_version}",
            )
        generated_rows = worker_meta.get("generated_count")
        ensure(
            isinstance(generated_rows, int) and generated_rows >= 0,
            f"Invalid generated_count in {worker_meta_path}: {generated_rows}",
        )
        actual_rows = count_jsonl_rows(worker_file)
        ensure(
            actual_rows == generated_rows,
            f"Rank output row count mismatch for {worker_file}: metadata={generated_rows}, actual={actual_rows}",
        )
        worker_outputs.append(worker_file)
        total_generated += generated_rows
    return worker_outputs, total_generated


def wait_for_worker_outputs(
    output_dir: Path,
    world_size: int,
    run_id: str,
    expected_total: int,
    timeout_seconds: int = 3600,
    poll_interval_seconds: float = 2.0,
) -> List[Path]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            worker_outputs, total_generated = collect_worker_outputs(output_dir, world_size, run_id)
            ensure(
                total_generated == expected_total,
                f"TraffiX-Qwen V5 forward expected {expected_total} predictions across all ranks, "
                f"but found {total_generated}.",
            )
            return worker_outputs
        except Exception as exc:
            last_error = str(exc)
            time.sleep(poll_interval_seconds)
    raise TimeoutError(
        "Timed out waiting for TraffiX-Qwen V5 rank outputs to finish. "
        f"Last observed issue: {last_error or 'unknown'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict distributed inference for TraffiX-Qwen Intersection V5.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29542)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-max-length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--debug-forward", action="store_true")
    parser.add_argument("--conv-mode", default=DEFAULT_PROMPT_VERSION)
    return parser.parse_args()


def has_tokenizer_files(model_path: Path) -> bool:
    required = ("tokenizer.json", "tokenizer_config.json")
    return all((model_path / filename).is_file() for filename in required)


def is_lora_checkpoint(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").is_file() or (model_path / "non_lora_trainables.bin").is_file()


def resolve_model_base(model_path: Path, explicit_model_base: str | None) -> str | None:
    if explicit_model_base is not None:
        resolved = Path(explicit_model_base).resolve()
        ensure(resolved.exists(), f"Model base path not found: {resolved}")
        return str(resolved)

    if is_lora_checkpoint(model_path):
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.is_file():
            with adapter_config_path.open("r", encoding="utf-8") as file:
                adapter_config = json.load(file)
            candidate = adapter_config.get("base_model_name_or_path")
            if isinstance(candidate, str) and candidate.strip():
                candidate_path = Path(candidate).expanduser()
                if candidate_path.exists():
                    return str(candidate_path.resolve())

    if has_tokenizer_files(model_path):
        return None

    config_path = model_path / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        candidate = config.get("_name_or_path")
        if isinstance(candidate, str) and candidate.strip():
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path.resolve())

    default_base = Path(DEFAULT_MODEL_NAME_OR_PATH).resolve()
    ensure(default_base.exists(), f"Default base model path not found: {default_base}")
    return str(default_base)


def launch_distributed_forward(args: argparse.Namespace) -> None:
    ensure(torch.cuda.is_available(), "CUDA is required for TraffiX-Qwen V5 inference.")
    ensure(args.num_gpus >= 1, "--num-gpus must be at least 1.")
    ensure(
        torch.cuda.device_count() >= args.num_gpus,
        f"Requested {args.num_gpus} GPUs for TraffiX-Qwen V5 forward, "
        f"but only {torch.cuda.device_count()} CUDA devices are visible.",
    )
    command = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_port={args.master_port}",
        "-m",
        "traffixqwen_v5.cli.forward",
        "--model-path",
        args.model_path,
        "--data-path",
        args.data_path,
        "--output-dir",
        args.output_dir,
        "--num-gpus",
        str(args.num_gpus),
        "--master-port",
        str(args.master_port),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--model-max-length",
        str(args.model_max_length),
        "--attn-implementation",
        args.attn_implementation,
        "--dtype",
        args.dtype,
        "--conv-mode",
        args.conv_mode,
    ]
    if args.debug_forward:
        command.append("--debug-forward")
    if args.model_base is not None:
        command.extend(["--model-base", args.model_base])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    if args.model_path is None:
        args.model_path = str(work_dir / "checkpoints" / "best")
    if args.data_path is None:
        args.data_path = str(prepared_dir / "val.json")
    if args.output_dir is None:
        args.output_dir = str(work_dir / "predictions")
    if "WORLD_SIZE" not in os.environ:
        launch_distributed_forward(args)
        return
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model_path = Path(args.model_path).resolve()
    ensure(model_path.exists(), f"Model path not found: {model_path}")
    data_path = Path(args.data_path).resolve()
    ensure(data_path.is_file(), f"Inference data not found: {data_path}")
    manifest_path = prepared_dir / "split_manifest.json"
    ensure(manifest_path.is_file(), f"Prepared split manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )
    ensure(args.model_max_length > 0, "--model-max-length must be a positive integer.")
    if args.dtype == "bf16":
        ensure(torch.cuda.is_bf16_supported(), "Requested --dtype bf16 but CUDA bf16 is not supported on this system.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = resolve_run_id(rank)

    try:
        ensure(torch.cuda.is_available(), "CUDA is required for TraffiX-Qwen V5 inference.")
        from llava.model.builder import load_pretrained_model
        from llava.train import train as base_train

        resolved_model_base = resolve_model_base(model_path, args.model_base)
        model_name = "llava_qwen_lora" if resolved_model_base and is_lora_checkpoint(model_path) else "llava_qwen"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(model_path),
            resolved_model_base,
            model_name,
            device_map=None,
            attn_implementation=args.attn_implementation,
            multimodal=True,
        )
        meta_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
        ensure(
            not meta_parameters,
            "TraffiX-Qwen V5 forward loaded a checkpoint with unresolved meta tensors. "
            f"Example parameters: {meta_parameters[:8]}",
        )
        inference_dtype = resolve_torch_dtype(args.dtype)
        model.config.debug_forward = args.debug_forward
        model.config.forward_debug_batch_limit = 2
        model.config.forward_debug_rank = rank
        model.config.forward_attn_implementation = args.attn_implementation
        model.config.forward_inference_dtype = args.dtype
        # Decoder-only batched generation must use left padding.
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = args.model_max_length
        if tokenizer.pad_token_id is None and tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        model.config.tokenizer_padding_side = "left"
        model.config.tokenizer_model_max_length = args.model_max_length
        if hasattr(model.config, "max_sequence_length"):
            model.config.max_sequence_length = args.model_max_length
        model.to(device=device, dtype=inference_dtype)
        model.eval()
        model_dtype = infer_model_dtype(model)
        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_token_ids = [
            token_id
            for token_id in {tokenizer.eos_token_id, im_end_token_id}
            if isinstance(token_id, int) and token_id >= 0
        ]
        ensure(eos_token_ids, "TraffiX-Qwen V5 forward could not resolve any EOS token ids.")
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.unk_token_id
        ensure(pad_token_id is not None, "TraffiX-Qwen V5 forward requires a valid pad token id.")

        data_args = base_train.DataArguments(
            data_path=str(data_path),
            image_folder="",
            image_aspect_ratio="pad",
            is_multimodal=True,
        )
        data_args.image_processor = image_processor

        all_samples = load_json(data_path)
        ensure(isinstance(all_samples, list) and all_samples, f"{data_path} does not contain a non-empty sample list.")
        shard = all_samples[rank::world_size]
        print(
            f"[forward] rank={rank} local_rank={local_rank} device={device} shard={len(shard)}/{len(all_samples)} "
            f"output_dir={output_dir} run_id={run_id}"
        )
        print(
            f"[forward] rank={rank} model_dtype={model_dtype} attn={args.attn_implementation} "
            f"tokenizer_max_length={tokenizer.model_max_length} pad_token_id={pad_token_id} eos_token_ids={eos_token_ids}"
        )
        dataset = StrictIntersectionV5EvalDataset(samples=shard, tokenizer=tokenizer, data_args=data_args, conv_mode=args.conv_mode)
        collator = StrictIntersectionV5InferenceCollator(tokenizer=tokenizer)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=False,
        )

        rank_output = output_dir / f"predictions_rank{rank}.jsonl"
        rank_meta_path = output_dir / f"predictions_rank{rank}.meta.json"
        generated_count = 0
        empty_prediction_count = 0
        empty_rows: List[Dict[str, Any]] = []
        unique_predictions: set[str] = set()
        with rank_output.open("w", encoding="utf-8") as output_file:
            iterator = tqdm(data_loader, desc="Forward", leave=True) if rank == 0 else data_loader
            with torch.inference_mode():
                for batch_idx, batch in enumerate(iterator):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    image_tensors = [image_tensor.to(device=device, dtype=model_dtype) for image_tensor in batch["images"]]
                    model.config.forward_debug_context = {
                        "rank": rank,
                        "batch_idx": batch_idx,
                        "question_ids": [record.question_id for record in batch["records"][: min(2, len(batch["records"]))]],
                        "frame_tokens": [record.frame_token for record in batch["records"][: min(2, len(batch["records"]))]],
                        "subtemplates": [record.subtemplate for record in batch["records"][: min(2, len(batch["records"]))]],
                    }
                    if args.debug_forward and batch_idx < 2:
                        attention_lengths = attention_mask.sum(dim=1).tolist()
                        image_shapes = [tuple(image_tensor.shape) for image_tensor in image_tensors[: min(2, len(image_tensors))]]
                        print(
                            f"[forward-debug] rank={rank} batch={batch_idx} input_shape={tuple(input_ids.shape)} "
                            f"attention_lengths={attention_lengths} image_count={len(image_tensors)} "
                            f"image_shapes={image_shapes} image_sizes={batch['image_sizes'][: min(2, len(batch['image_sizes']))]}"
                        )
                    generated = model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        images=image_tensors,
                        image_sizes=batch["image_sizes"],
                        modalities=batch["modalities"],
                        do_sample=False,
                        temperature=1.0,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0],
                        pad_token_id=pad_token_id,
                        return_dict_in_generate=True,
                    )
                    sequences = generated["sequences"] if isinstance(generated, dict) else generated.sequences
                    prompt_length = input_ids.shape[1]
                    if sequences.shape[1] > prompt_length:
                        new_tokens = sequences[:, prompt_length:]
                    else:
                        new_tokens = sequences
                    ensure(
                        new_tokens.shape[1] > 0,
                        f"generate() produced zero-length decoded tokens for prompt_length={prompt_length} "
                        f"and sequences_length={sequences.shape[1]} in TraffiX-Qwen V5 inference.",
                    )
                    predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    cleaned_predictions = [clean_prediction_text(prediction) for prediction in predictions]
                    batch_unique_nonempty = {prediction for prediction in cleaned_predictions if prediction}
                    batch_punctuation_only = sum(1 for prediction in cleaned_predictions if is_punctuation_only(prediction))
                    if args.debug_forward and batch_idx < 2:
                        print(
                            f"[forward-debug] rank={rank} batch={batch_idx} sequences_shape={tuple(sequences.shape)} "
                            f"new_tokens_shape={tuple(new_tokens.shape)} non_finite_new_tokens={tensor_has_non_finite(new_tokens)} "
                            f"batch_unique_nonempty={len(batch_unique_nonempty)} punctuation_only={batch_punctuation_only}/{len(cleaned_predictions)} "
                            f"sample_prediction={summarize_text(cleaned_predictions[0] if cleaned_predictions else '')}"
                        )
                    if (
                        len(cleaned_predictions) >= 2
                        and len(batch_unique_nonempty) == 1
                        and batch_punctuation_only == len(cleaned_predictions)
                    ):
                        first_record = batch["records"][0]
                        raise ValueError(
                            "TraffiX-Qwen V5 forward detected a collapsed punctuation-only batch on "
                            f"rank {rank}, batch {batch_idx}, question_id={first_record.question_id}, "
                            f"subtemplate={first_record.subtemplate}, question={summarize_text(first_record.question)}, "
                            f"new_token_ids={new_tokens[0, : min(16, new_tokens.shape[1])].tolist()}, "
                            f"decoded={summarize_text(next(iter(batch_unique_nonempty)))}"
                        )
                    for record, cleaned_prediction in zip(batch["records"], cleaned_predictions):
                        generated_count += 1
                        unique_predictions.add(cleaned_prediction)
                        if not cleaned_prediction:
                            empty_prediction_count += 1
                        row: Dict[str, Any] = {
                            "question_id": record.question_id,
                            "scene_id": record.scene_id,
                            "frame_token": record.frame_token,
                            "chapter": record.chapter,
                            "section": record.section,
                            "subtemplate": record.subtemplate,
                            "question": record.question,
                            "reference_answer": record.answer,
                            "prediction": cleaned_prediction,
                        }
                        if not cleaned_prediction:
                            empty_rows.append(dict(row))
                        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        unique_nonempty_predictions = {prediction for prediction in unique_predictions if prediction}
        rank_meta = {
            "run_id": run_id,
            "rank": rank,
            "world_size": world_size,
            "dataset_version": str(resolved["dataset_version"]),
            "generated_count": generated_count,
            "empty_prediction_count": empty_prediction_count,
            "unique_prediction_count": len(unique_predictions),
            "unique_nonempty_prediction_count": len(unique_nonempty_predictions),
            "output_file": rank_output.name,
        }
        with rank_meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(rank_meta, meta_file, ensure_ascii=False, indent=2)

        print(
            f"[forward] rank={rank} generated={generated_count} empty_predictions={empty_prediction_count} "
            f"unique_predictions={len(unique_predictions)}"
        )
        if empty_rows:
            diagnostics_path = output_dir / f"empty_predictions_rank{rank}.jsonl"
            with diagnostics_path.open("w", encoding="utf-8") as diagnostics_file:
                for row in empty_rows:
                    diagnostics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[forward] rank={rank} wrote empty prediction diagnostics to {diagnostics_path}")
        ensure(
            empty_prediction_count < generated_count,
            f"TraffiX-Qwen V5 forward produced only empty predictions on rank {rank}.",
        )
        ensure(
            not (generated_count >= 64 and len(unique_nonempty_predictions) == 1),
            "TraffiX-Qwen V5 forward collapsed to a single repeated non-empty prediction on "
            f"rank {rank}: {next(iter(unique_nonempty_predictions))!r}",
        )

        if rank == 0:
            worker_outputs = wait_for_worker_outputs(output_dir, world_size, run_id, len(all_samples))
            merged_output = output_dir / "merged_predictions.jsonl"
            with merged_output.open("w", encoding="utf-8") as merged_file:
                for worker_file in worker_outputs:
                    merged_file.write(worker_file.read_text(encoding="utf-8"))
            print(f"Wrote merged predictions to {merged_output}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
