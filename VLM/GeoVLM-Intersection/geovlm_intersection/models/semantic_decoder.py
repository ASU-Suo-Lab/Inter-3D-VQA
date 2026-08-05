from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from geovlm_intersection.backbones.qwen3_vl_adapter import Qwen3VLModelRuntime, build_qwen3_vl_model_runtime


SEMANTIC_GENERATION_SUBTEMPLATES = {
    "1_1_1_fine_type",
    "4_3_1_intersection_action",
    "4_3_2_side_action",
    "4_3_3_lane_action",
    "4_3_4_object_action",
}

HYBRID_GENERATION_SUBTEMPLATES = {
    "1_1_4_relative_neighbor_type",
    "3_1_1_current_motion_state",
    "3_4_2_nearest_conflict_participant",
    "3_4_3_primary_risk_subject",
    "4_2_1_speeding_risk",
}

GENERATION_SUBTEMPLATES = SEMANTIC_GENERATION_SUBTEMPLATES | HYBRID_GENERATION_SUBTEMPLATES

MAX_DECODER_TARGET_TOKENS = 128
MAX_DECODER_GENERATION_TOKENS = 96


@dataclass(frozen=True)
class SemanticDecoderLossOutput:
    loss: torch.Tensor | None
    loss_sum: torch.Tensor | None
    batch_mean_loss: torch.Tensor | None
    active_count: int


def uses_decoder_training(subtemplate: str) -> bool:
    return subtemplate in GENERATION_SUBTEMPLATES


def uses_decoder_final_prediction(subtemplate: str) -> bool:
    return subtemplate in SEMANTIC_GENERATION_SUBTEMPLATES


def uses_hybrid_structured_render(subtemplate: str) -> bool:
    return subtemplate in HYBRID_GENERATION_SUBTEMPLATES


STRUCTURED_FINAL_SUBTEMPLATES = {
    "1_1_2_side_exists",
    "1_1_3_side_count",
    "2_1_2_ped_to_far_edge",
    "2_1_4_nearest_vehicle_to_ped",
}


def build_decoder_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_text: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(user_prompt).strip()})
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": str(assistant_text).strip()})
    return messages


def build_decoder_chat_prompt_text(
    *,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages = build_decoder_messages(system_prompt=system_prompt, user_prompt=user_prompt)
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    ).rstrip()


def build_decoder_prompt_text(prompt_text: str) -> str:
    return str(prompt_text).rstrip()


def build_frozen_semantic_decoder_runtime(
    *,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Qwen3VLModelRuntime:
    effective_dtype = torch_dtype
    if not str(device).startswith("cuda"):
        effective_dtype = torch.float32
    runtime = build_qwen3_vl_model_runtime(device=device, torch_dtype=effective_dtype)
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    return runtime


def filter_structured_targets_for_training(subtemplate: str, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if subtemplate not in GENERATION_SUBTEMPLATES:
        return targets

    keep_keys = {"subtemplate_index"}
    if subtemplate in {"1_1_1_fine_type", "1_1_4_relative_neighbor_type", "3_4_2_nearest_conflict_participant", "3_4_3_primary_risk_subject", "4_3_4_object_action"}:
        keep_keys.add("object_selection_index")
    if subtemplate in {"1_1_4_relative_neighbor_type", "3_4_2_nearest_conflict_participant", "3_4_3_primary_risk_subject"}:
        keep_keys.update({"position_3d", "camera_index", "image_ref"})
    if subtemplate in HYBRID_GENERATION_SUBTEMPLATES:
        if subtemplate == "3_1_1_current_motion_state":
            keep_keys.update(
                {
                    "object_selection_index",
                    "object_type_index",
                    "side_index",
                    "motion_state_index",
                    "speed_value",
                    "acceleration_value",
                }
            )
        elif subtemplate == "4_2_1_speeding_risk":
            keep_keys.update(
                {
                    "object_selection_index",
                    "object_type_index",
                    "side_index",
                    "binary_answer",
                    "speed_value",
                }
            )
        elif subtemplate in {"1_1_4_relative_neighbor_type", "3_4_2_nearest_conflict_participant", "3_4_3_primary_risk_subject"}:
            keep_keys.update({"object_selection_index", "position_3d", "camera_index", "image_ref"})

    return {key: value for key, value in targets.items() if key in keep_keys}


def _clean_generated_text(text: str, prompt_text: str) -> str:
    cleaned = str(text).strip()
    prompt_stripped = prompt_text.strip()
    if prompt_stripped and cleaned.startswith(prompt_stripped):
        cleaned = cleaned[len(prompt_stripped) :].lstrip()
    for prefix in ("Answer:", "Final answer:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
    return cleaned.strip()


def normalize_answer_text(text: str) -> str:
    cleaned = str(text).strip().lower()
    for old, new in {
        "\n": " ",
        "\t": " ",
        ",": " ",
        ".": " ",
        ";": " ",
        ":": " ",
    }.items():
        cleaned = cleaned.replace(old, new)
    return " ".join(cleaned.split())


def _tokenizer(runtime: Qwen3VLModelRuntime):
    tokenizer = runtime.base_runtime.processor.tokenizer
    return tokenizer


def compute_decoder_text_loss(
    *,
    runtime: Qwen3VLModelRuntime,
    semantic_prefix_tokens: torch.Tensor,
    prompt_texts: Sequence[str],
    answer_texts: Sequence[str],
    subtemplates: Sequence[str],
    batch_size: int | None = None,
) -> SemanticDecoderLossOutput:
    active_indices = [
        index
        for index, subtemplate in enumerate(subtemplates)
        if uses_decoder_training(str(subtemplate)) and str(answer_texts[index]).strip()
    ]
    if not active_indices:
        return SemanticDecoderLossOutput(loss=None, loss_sum=None, batch_mean_loss=None, active_count=0)

    tokenizer = _tokenizer(runtime)
    embed_layer = runtime.model.get_input_embeddings()
    language_model = runtime.model.model.language_model
    lm_head = runtime.model.lm_head
    eos_token_id = tokenizer.eos_token_id
    prompt_text_batch = [build_decoder_prompt_text(str(prompt_texts[index])) for index in active_indices]
    answer_text_batch = [str(answer_texts[index]).strip() for index in active_indices]

    prompt_encoding = tokenizer(prompt_text_batch, add_special_tokens=False, padding=False)
    answer_encoding = tokenizer(answer_text_batch, add_special_tokens=False, padding=False)

    prefix_batch = semantic_prefix_tokens[active_indices].to(device=runtime.device, dtype=runtime.torch_dtype)
    batch_size = len(active_indices)
    hidden_size = int(prefix_batch.shape[-1])
    prefix_length = int(prefix_batch.shape[1])

    sequence_embeds: list[torch.Tensor] = []
    label_tensors: list[torch.Tensor] = []
    for local_index in range(batch_size):
        prompt_ids = torch.tensor(prompt_encoding["input_ids"][local_index], dtype=torch.long, device=runtime.device)
        answer_ids_list = list(answer_encoding["input_ids"][local_index])[:MAX_DECODER_TARGET_TOKENS]
        if eos_token_id is not None:
            answer_ids_list.append(int(eos_token_id))
        answer_ids = torch.tensor(answer_ids_list, dtype=torch.long, device=runtime.device)

        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids.unsqueeze(0)).squeeze(0).to(dtype=runtime.torch_dtype)
            answer_embeds = embed_layer(answer_ids.unsqueeze(0)).squeeze(0).to(dtype=runtime.torch_dtype)

        prefix_tokens = prefix_batch[local_index]
        sequence = torch.cat([prompt_embeds, prefix_tokens, answer_embeds], dim=0)
        labels = torch.full((sequence.shape[0],), -100, dtype=torch.long, device=runtime.device)
        labels[prompt_embeds.shape[0] + prefix_length :] = answer_ids
        sequence_embeds.append(sequence)
        label_tensors.append(labels)

    max_length = max(int(sequence.shape[0]) for sequence in sequence_embeds)
    inputs_embeds = torch.zeros((batch_size, max_length, hidden_size), dtype=runtime.torch_dtype, device=runtime.device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=runtime.device)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long, device=runtime.device)
    for index, (sequence, sample_labels) in enumerate(zip(sequence_embeds, label_tensors)):
        seq_length = int(sequence.shape[0])
        inputs_embeds[index, :seq_length] = sequence
        attention_mask[index, :seq_length] = 1
        labels[index, :seq_length] = sample_labels

    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = lm_head(outputs.last_hidden_state)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    ensure_valid = int(valid_mask.sum().item())
    if ensure_valid <= 0:
        return SemanticDecoderLossOutput(loss=None, loss_sum=None, batch_mean_loss=None, active_count=0)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    effective_batch = int(batch_size if batch_size is not None else len(subtemplates))
    loss_sum = loss * float(len(active_indices))
    batch_mean_loss = loss_sum / float(max(1, effective_batch))
    return SemanticDecoderLossOutput(
        loss=loss,
        loss_sum=loss_sum,
        batch_mean_loss=batch_mean_loss,
        active_count=len(active_indices),
    )


def generate_decoder_outputs(
    *,
    runtime: Qwen3VLModelRuntime,
    semantic_prefix_tokens: torch.Tensor,
    prompt_texts: Sequence[str],
    subtemplates: Sequence[str],
    max_new_tokens: int = MAX_DECODER_GENERATION_TOKENS,
) -> list[str | None]:
    tokenizer = _tokenizer(runtime)
    embed_layer = runtime.model.get_input_embeddings()
    language_model = runtime.model.model.language_model
    lm_head = runtime.model.lm_head
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    outputs: list[str | None] = [None] * len(subtemplates)
    active_indices = [index for index, subtemplate in enumerate(subtemplates) if uses_decoder_training(str(subtemplate))]
    if not active_indices:
        return outputs

    prompt_text_batch = [build_decoder_prompt_text(str(prompt_texts[index])) for index in active_indices]
    prompt_encoding = tokenizer(prompt_text_batch, add_special_tokens=False, padding=True, return_tensors="pt")
    prompt_ids = prompt_encoding["input_ids"].to(runtime.device)
    prompt_attention_mask = prompt_encoding["attention_mask"].to(runtime.device)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids).to(dtype=runtime.torch_dtype)
        prefix_batch = semantic_prefix_tokens[active_indices].to(device=runtime.device, dtype=runtime.torch_dtype)
        inputs_embeds = torch.cat([prompt_embeds, prefix_batch], dim=1)
        attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(prefix_batch.shape[:2], dtype=torch.long, device=runtime.device),
            ],
            dim=1,
        )
        generated_tokens: list[torch.Tensor] = []
        past_key_values = None
        current_embeds = inputs_embeds
        current_attention_mask = attention_mask
        finished = torch.zeros((len(active_indices),), dtype=torch.bool, device=runtime.device)
        for _ in range(max_new_tokens):
            model_outputs = language_model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = lm_head(model_outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, int(eos_token_id)),
                    next_token,
                )
            generated_tokens.append(next_token)
            if eos_token_id is not None:
                finished = finished | next_token.eq(int(eos_token_id))
                if bool(finished.all()):
                    break
            past_key_values = model_outputs.past_key_values
            current_embeds = embed_layer(next_token.unsqueeze(1)).to(dtype=runtime.torch_dtype)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), dtype=torch.long, device=runtime.device),
                ],
                dim=1,
            )
        if generated_tokens:
            generated = torch.stack(generated_tokens, dim=1)
        else:
            generated = torch.full(
                (len(active_indices), 1),
                int(pad_token_id) if pad_token_id is not None else 0,
                dtype=torch.long,
                device=runtime.device,
            )
    for local_index, sample_index in enumerate(active_indices):
        decoded = tokenizer.decode(generated[local_index], skip_special_tokens=True)
        outputs[sample_index] = _clean_generated_text(decoded, prompt_text_batch[local_index])
    return outputs
