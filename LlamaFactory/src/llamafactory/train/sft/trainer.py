# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import inspect
from collections import Counter, defaultdict
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.lidar_template import lidar_template_family_name_from_subtemplate
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)
_SUBTEMPLATE_PATTERN = re.compile(r"Subtemplate:\s*([A-Za-z0-9_]+)")
_PREDICT_SUPERVISION_KEYS = (
    "numeric_count_targets",
    "numeric_count_mask",
    "numeric_motion_targets",
    "numeric_motion_mask",
    "numeric_coord_targets",
    "numeric_coord_mask",
)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        self._prediction_lidar_diagnostics: list[dict[str, Any]] = []
        self._collect_prediction_lidar_diagnostics_enabled = False

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)
        self._last_lidar_aux_log_step = -1

    def _maybe_log_lidar_aux_metrics(self, outputs: Any) -> None:
        if not getattr(self.model, "training", False):
            return

        metrics = None
        if isinstance(outputs, dict):
            metrics = outputs.get("lidar_aux_metrics")
        else:
            metrics = getattr(outputs, "lidar_aux_metrics", None)

        if not isinstance(metrics, dict) or not metrics:
            return

        next_step = self.state.global_step + 1
        if self.args.logging_steps <= 0 or next_step % self.args.logging_steps != 0:
            return
        if self._last_lidar_aux_log_step == next_step:
            return

        self._last_lidar_aux_log_step = next_step
        self.log({key: float(value) for key, value in metrics.items()})

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return_outputs = kwargs.pop("return_outputs", False)
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            loss = self.compute_loss_func(outputs, inputs["labels"], ref_logits)
            return (loss, outputs) if return_outputs else loss
        else:
            outputs = model(**inputs)
            self._maybe_log_lidar_aux_metrics(outputs)
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        raw_lidar_presence = self._summarize_prediction_input_lidar_presence(inputs)
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
            for key in _PREDICT_SUPERVISION_KEYS:
                inputs.pop(key, None)
            prepare_inputs_for_generation = getattr(model, "prepare_inputs_for_generation", None)
            if prepare_inputs_for_generation is None:
                prepare_inputs_for_generation = getattr(
                    self.accelerator.unwrap_model(model),
                    "prepare_inputs_for_generation",
                    None,
                )
            if prepare_inputs_for_generation is not None:
                signature = inspect.signature(prepare_inputs_for_generation)
                if "rope_deltas" not in signature.parameters:
                    inputs.pop("rope_deltas", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        if self._collect_prediction_lidar_diagnostics_enabled:
            self._capture_prediction_lidar_diagnostics(model, inputs, raw_lidar_presence=raw_lidar_presence)

        return loss, generated_tokens, labels

    @override
    def predict(self, *args, **kwargs):
        self._prediction_lidar_diagnostics = []
        self._collect_prediction_lidar_diagnostics_enabled = True
        try:
            return super().predict(*args, **kwargs)
        finally:
            self._collect_prediction_lidar_diagnostics_enabled = False

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)
        lidar_diagnostics = self._prediction_lidar_diagnostics
        prediction_rows: list[dict[str, Any]] = []

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for idx, (text, pred, label) in enumerate(zip(decoded_inputs, decoded_preds, decoded_labels)):
                match = _SUBTEMPLATE_PATTERN.search(text)
                subtemplate = match.group(1) if match is not None else None
                row = {
                    "sample_index": idx,
                    "subtemplate": subtemplate,
                    "lidar_template_family": lidar_template_family_name_from_subtemplate(subtemplate),
                    "prompt": text,
                    "predict": pred,
                    "label": label,
                }
                if idx < len(lidar_diagnostics):
                    row.update(lidar_diagnostics[idx])
                prediction_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._save_lidar_diagnostics_summary(prediction_rows)

    def _save_lidar_diagnostics_summary(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
            status_counts = Counter(row.get("lidar_diagnostics_status", "missing") for row in group_rows)
            object_rate = float(np.mean([bool(row.get("has_lidar_object_features", False)) for row in group_rows]))
            scene_rate = float(np.mean([bool(row.get("has_lidar_scene_features", False)) for row in group_rows]))
            metric_values: dict[str, list[float]] = defaultdict(list)
            for row in group_rows:
                diagnostics = row.get("lidar_diagnostics")
                if not isinstance(diagnostics, dict):
                    continue
                for key, value in diagnostics.items():
                    if isinstance(value, (int, float)):
                        metric_values[key].append(float(value))

            return {
                "count": len(group_rows),
                "status_counts": dict(status_counts),
                "lidar_object_presence_rate": object_rate,
                "lidar_scene_presence_rate": scene_rate,
                "metric_means": {
                    key: float(np.mean(values)) for key, values in sorted(metric_values.items()) if values
                },
            }

        by_subtemplate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_subtemplate[str(row.get("subtemplate") or "unknown")].append(row)
            by_family[str(row.get("lidar_template_family") or "unknown")].append(row)

        summary = {
            "overall": summarize(rows),
            "by_template_family": {
                key: summarize(group_rows) for key, group_rows in sorted(by_family.items())
            },
            "by_subtemplate": {
                key: summarize(group_rows) for key, group_rows in sorted(by_subtemplate.items())
            },
        }
        output_file = os.path.join(self.args.output_dir, "lidar_diagnostics_summary.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _capture_prediction_lidar_diagnostics(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        raw_lidar_presence: Optional[dict[str, Any]] = None,
    ) -> None:
        model_for_diagnostics = self.accelerator.unwrap_model(self.model)
        if not getattr(getattr(model_for_diagnostics, "config", None), "use_lidar_modality", False):
            return

        lower_model = getattr(model_for_diagnostics, "model", None)

        batch_inputs = {key: value for key, value in inputs.items()}
        batch_inputs.pop("labels", None)
        batch_inputs = self._prepare_inputs(batch_inputs)
        input_ids = batch_inputs.get("input_ids")
        batch_size = int(input_ids.size(0)) if torch.is_tensor(input_ids) else 0
        if batch_size == 0:
            return

        raw_lidar_presence = raw_lidar_presence or {}
        raw_object_presence = list(raw_lidar_presence.get("object_presence", []))
        raw_scene_presence = list(raw_lidar_presence.get("scene_presence", []))
        raw_has_object_key = bool(raw_lidar_presence.get("has_object_key", False))
        raw_has_scene_key = bool(raw_lidar_presence.get("has_scene_key", False))
        if len(raw_object_presence) != batch_size:
            raw_object_presence = [False] * batch_size
        if len(raw_scene_presence) != batch_size:
            raw_scene_presence = [False] * batch_size

        prepared_object_presence = self._sample_lidar_presence(
            batch_inputs.get("lidar_object_feature_mask"),
            batch_inputs.get("lidar_object_features"),
        )
        prepared_scene_presence = self._sample_lidar_presence(
            batch_inputs.get("lidar_scene_feature_mask"),
            batch_inputs.get("lidar_scene_features"),
        )
        if len(prepared_object_presence) != batch_size:
            prepared_object_presence = [False] * batch_size
        if len(prepared_scene_presence) != batch_size:
            prepared_scene_presence = [False] * batch_size

        clear_cache = getattr(lower_model, "clear_lidar_decoder_adapter_cache", None)
        if callable(clear_cache):
            clear_cache()

        with torch.no_grad():
            outputs = model(**batch_inputs, output_lidar_diagnostics=True, use_cache=False)

        if isinstance(outputs, dict):
            sample_metrics = outputs.get("lidar_aux_sample_metrics")
        else:
            sample_metrics = getattr(outputs, "lidar_aux_sample_metrics", None)

        local_rows: list[dict[str, Any]] = []
        if not isinstance(sample_metrics, dict) or not sample_metrics:
            for local_idx in range(batch_size):
                has_raw_object = raw_object_presence[local_idx]
                has_raw_scene = raw_scene_presence[local_idx]
                has_object = prepared_object_presence[local_idx]
                has_scene = prepared_scene_presence[local_idx]
                status = "missing_lidar_features"
                reason = None
                if (has_raw_object or has_raw_scene) and not (has_object or has_scene):
                    status = "lost_lidar_features_before_diagnostics_forward"
                    reason = "prediction_step_inputs_have_lidar_but_prepared_batch_missing"
                elif has_object or has_scene:
                    status = "missing_adapter_metrics"
                local_rows.append(
                    {
                        "lidar_diagnostics_status": status,
                        "lidar_diagnostics_reason": reason,
                        "prediction_step_has_lidar_object_key": raw_has_object_key,
                        "prediction_step_has_lidar_scene_key": raw_has_scene_key,
                        "prediction_step_has_lidar_object_features": has_raw_object,
                        "prediction_step_has_lidar_scene_features": has_raw_scene,
                        "has_lidar_object_features": has_object,
                        "has_lidar_scene_features": has_scene,
                        "prepared_has_lidar_object_features": has_object,
                        "prepared_has_lidar_scene_features": has_scene,
                    }
                )
        else:
            for local_idx in range(batch_size):
                row: dict[str, Any] = {
                    "prediction_step_has_lidar_object_key": raw_has_object_key,
                    "prediction_step_has_lidar_scene_key": raw_has_scene_key,
                    "prediction_step_has_lidar_object_features": raw_object_presence[local_idx],
                    "prediction_step_has_lidar_scene_features": raw_scene_presence[local_idx],
                    "has_lidar_object_features": prepared_object_presence[local_idx],
                    "has_lidar_scene_features": prepared_scene_presence[local_idx],
                    "prepared_has_lidar_object_features": prepared_object_presence[local_idx],
                    "prepared_has_lidar_scene_features": prepared_scene_presence[local_idx],
                }
                metric_values: dict[str, float] = {}
                for key, tensor in sample_metrics.items():
                    if torch.is_tensor(tensor) and tensor.ndim >= 1 and local_idx < tensor.size(0):
                        metric_values[key] = float(tensor[local_idx].item())
                has_lidar = row["has_lidar_object_features"] or row["has_lidar_scene_features"]
                had_raw_lidar = row["prediction_step_has_lidar_object_features"] or row["prediction_step_has_lidar_scene_features"]
                has_nonzero_metrics = any(abs(value) > 1.0e-12 for value in metric_values.values())
                if has_nonzero_metrics:
                    row["lidar_diagnostics_status"] = "ok"
                    row["lidar_diagnostics"] = metric_values
                elif has_lidar:
                    row["lidar_diagnostics_status"] = "invalid_all_zero"
                    row["lidar_diagnostics_reason"] = "lidar_present_but_adapter_metrics_zero"
                elif had_raw_lidar:
                    row["lidar_diagnostics_status"] = "lost_lidar_features_before_diagnostics_forward"
                    row["lidar_diagnostics_reason"] = "prediction_step_inputs_have_lidar_but_prepared_batch_missing"
                else:
                    row["lidar_diagnostics_status"] = "missing_lidar_features"
                local_rows.append(row)

        gathered_rows = self.accelerator.gather_for_metrics(local_rows, use_gather_object=True)
        if self.is_world_process_zero():
            self._prediction_lidar_diagnostics.extend(gathered_rows)

    def _summarize_prediction_input_lidar_presence(
        self,
        inputs: dict[str, Union["torch.Tensor", Any]],
    ) -> dict[str, Any]:
        input_ids = inputs.get("input_ids")
        batch_size = int(input_ids.size(0)) if torch.is_tensor(input_ids) else 0
        has_object_key = "lidar_object_features" in inputs or "lidar_object_feature_mask" in inputs
        has_scene_key = "lidar_scene_features" in inputs or "lidar_scene_feature_mask" in inputs
        object_presence = self._sample_lidar_presence(
            inputs.get("lidar_object_feature_mask"),
            inputs.get("lidar_object_features"),
        )
        scene_presence = self._sample_lidar_presence(
            inputs.get("lidar_scene_feature_mask"),
            inputs.get("lidar_scene_features"),
        )
        if len(object_presence) != batch_size:
            object_presence = [False] * batch_size
        if len(scene_presence) != batch_size:
            scene_presence = [False] * batch_size
        return {
            "has_object_key": has_object_key,
            "has_scene_key": has_scene_key,
            "object_presence": object_presence,
            "scene_presence": scene_presence,
        }

    def _sample_lidar_presence(
        self,
        mask_tensor: Optional[torch.Tensor],
        value_tensor: Optional[torch.Tensor],
    ) -> list[bool]:
        if torch.is_tensor(mask_tensor):
            mask_tensor = mask_tensor.to(dtype=torch.bool)
            if mask_tensor.ndim == 1:
                return [bool(x) for x in mask_tensor.detach().cpu().tolist()]
            if mask_tensor.ndim >= 2:
                flat_mask = mask_tensor.view(mask_tensor.size(0), -1)
                return [bool(x) for x in flat_mask.any(dim=1).detach().cpu().tolist()]

        if torch.is_tensor(value_tensor):
            flat_values = value_tensor.detach().view(value_tensor.size(0), -1)
            return [bool(x) for x in flat_values.ne(0).any(dim=1).cpu().tolist()]

        return []
