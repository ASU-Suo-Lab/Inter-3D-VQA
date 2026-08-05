from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from opendrivevla_v5.config.common import (
    DEFAULT_BEST_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_FEATURE_TRAIN_DIR,
    DEFAULT_FEATURE_VAL_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLOT_DIR,
    DEFAULT_PREPARED_DIR,
    DEFAULT_TRAINING_TEMP_DIR,
    DEFAULT_WORK_DIR,
    ensure_worktree_layout,
    resolve_dataset_version_paths,
)
from opendrivevla_v5.config.train import (
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_STEPS,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from opendrivevla_v5.data.collator import IntersectionV5Collator
from opendrivevla_v5.data.dataset import IntersectionV5QADataset
from opendrivevla_v5.engine.callbacks import LossHistoryCallback, materialize_best_and_last
from opendrivevla_v5.utils.modeling import enable_mm_projector_training, infer_lora_target_modules, load_trainable_model
from opendrivevla_v5.utils.tensors import change_tensor_to_bfloat16, change_tensor_to_float16, change_tensor_to_float32
from opendrivevla_v5.utils.io import ensure, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for OpenDriveVLA on Intersection V5 QA.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--qa-train-json", default=None)
    parser.add_argument("--qa-val-json", default=None)
    parser.add_argument("--infos-train-pkl", default=None)
    parser.add_argument("--infos-val-pkl", default=None)
    parser.add_argument("--feature-train-dir", default=None)
    parser.add_argument("--feature-val-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=SAVE_TOTAL_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--train-mm-projector", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntersectionV5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "uniad_pth" in inputs:
            model_dtype = next(model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                inputs["uniad_pth"] = change_tensor_to_bfloat16(inputs["uniad_pth"])
            elif model_dtype == torch.float16:
                inputs["uniad_pth"] = change_tensor_to_float16(inputs["uniad_pth"])
            elif model_dtype == torch.float32:
                inputs["uniad_pth"] = change_tensor_to_float32(inputs["uniad_pth"])
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def build_datasets_and_model(args):
    tokenizer, model = load_trainable_model(
        model_path=str(Path(args.model_path).resolve()),
        device=args.device,
        attn_implementation=args.attn_implementation,
        vision_tower_test_mode=True,
    )
    model.train()
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if not args.disable_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = infer_lora_target_modules(model)
        if not target_modules:
            raise RuntimeError("Failed to infer LoRA target modules from model.")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["mm_projector_scene", "mm_projector_track", "mm_projector_map"],
        )
        model = get_peft_model(model, peft_config)

    if args.train_mm_projector:
        enable_mm_projector_training(model)

    model.to(dtype=torch.bfloat16)

    prepared_dir = Path(args.prepared_dir).resolve()
    train_dataset = IntersectionV5QADataset(
        tokenizer=tokenizer,
        qa_json=str(Path(args.qa_train_json).resolve() if args.qa_train_json else prepared_dir / "qa_train.json"),
        infos_pkl=str(Path(args.infos_train_pkl).resolve() if args.infos_train_pkl else prepared_dir / "infos_train.pkl"),
        device=torch.device("cpu"),
        llava_train_mode=True,
        include_visual_tokens=True,
        uniad_pth_dir=str(Path(args.feature_train_dir).resolve()),
        max_samples=args.max_train_samples,
    )
    eval_dataset = IntersectionV5QADataset(
        tokenizer=tokenizer,
        qa_json=str(Path(args.qa_val_json).resolve() if args.qa_val_json else prepared_dir / "qa_val.json"),
        infos_pkl=str(Path(args.infos_val_pkl).resolve() if args.infos_val_pkl else prepared_dir / "infos_val.pkl"),
        device=torch.device("cpu"),
        llava_train_mode=True,
        include_visual_tokens=True,
        uniad_pth_dir=str(Path(args.feature_val_dir).resolve()),
        max_samples=args.max_eval_samples,
    )
    collator = IntersectionV5Collator(tokenizer=tokenizer, llava_train_mode=True)
    return tokenizer, model, train_dataset, eval_dataset, collator


def main():
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError("Intersection V5 training supports only --per-device-train-batch-size 1.")
    if args.eval_steps != args.save_steps:
        raise ValueError("--eval-steps and --save-steps must match for best+last checkpoint management.")
    if not args.device.startswith("cuda"):
        raise ValueError("Intersection V5 training requires CUDA because precision is fixed to BF16 AMP.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but OpenDriveVLA V5 training requires BF16 AMP on GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Current CUDA device does not support BF16, but OpenDriveVLA V5 training requires BF16 AMP.")

    resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    work_dir = Path(resolved["work_dir"]).resolve()
    args.prepared_dir = str(prepared_dir)
    args.work_dir = str(work_dir)
    if args.feature_train_dir is None:
        args.feature_train_dir = str(work_dir / DEFAULT_FEATURE_TRAIN_DIR.name)
    if args.feature_val_dir is None:
        args.feature_val_dir = str(work_dir / DEFAULT_FEATURE_VAL_DIR.name)
    manifest_path = prepared_dir / "split_manifest.json"
    ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    ensure(
        str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
        f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
    )
    ensure_worktree_layout(work_dir)
    trainer_output_dir = work_dir / DEFAULT_TRAINING_TEMP_DIR.name
    is_primary = int(os.environ.get("RANK", "0")) == 0
    if is_primary:
        if trainer_output_dir.exists():
            shutil.rmtree(trainer_output_dir)
        trainer_output_dir.mkdir(parents=True, exist_ok=True)
        print("[train] precision=bf16_amp", flush=True)

    set_seed(args.seed)
    tokenizer, model, train_dataset, eval_dataset, collator = build_datasets_and_model(args)
    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=max(args.save_total_limit, 2),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=False,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        ddp_find_unused_parameters=True,
        prediction_loss_only=True,
    )

    trainer = IntersectionV5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[LossHistoryCallback(Path(work_dir / DEFAULT_LOG_DIR.name), Path(work_dir / DEFAULT_PLOT_DIR.name))],
    )
    trainer.train()

    if trainer.is_world_process_zero():
        best_payload = materialize_best_and_last(
            trainer=trainer,
            tokenizer=tokenizer,
            trainer_output_dir=trainer_output_dir,
            checkpoint_dir=Path(work_dir / DEFAULT_CHECKPOINT_DIR.name),
            log_history=trainer.state.log_history,
        )
        train_summary = {
            "num_train_samples": len(train_dataset),
            "num_eval_samples": len(eval_dataset),
            "work_dir": str(work_dir),
            "best_checkpoint": str(work_dir / DEFAULT_CHECKPOINT_DIR.name / DEFAULT_BEST_DIR.name),
            "last_checkpoint": str(work_dir / DEFAULT_CHECKPOINT_DIR.name / "last.pth"),
            "best_metrics": best_payload,
            "use_lora": not args.disable_lora,
            "train_mm_projector": args.train_mm_projector,
            "dataset_version": str(resolved["dataset_version"]),
        }
        with (work_dir / DEFAULT_LOG_DIR.name / "train_summary.json").open("w", encoding="utf-8") as file:
            json.dump(train_summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
