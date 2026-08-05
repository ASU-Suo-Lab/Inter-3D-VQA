# Copyright 2025 the LlamaFactory team.
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

import weakref
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.constants import IGNORE_INDEX
from ..extras.lidar_template import LIDAR_TEMPLATE_FAMILY_NAMES, LIDAR_TEMPLATE_ROUTE_INITIAL_SCALES
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.lidar import (
    LidarDecoderAdapterBlock,
    LidarNumericHead,
    LidarTemplateRouter,
    ObjectQueryConnector,
    ObjectGeometryEncoder,
    ObjectNumericEncoder,
    ObjectPrefixEncoder,
    SceneEncoder,
    SceneLatentConnector,
    TokenProjectionEncoder,
)
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_youtu_vl_model(model: "PreTrainedModel") -> None:
    original_forward = model.forward

    def forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if "loss" not in outputs and "labels" in kwargs:
            logits = outputs.get("logits")
            labels = kwargs.get("labels")
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                outputs["loss"] = loss

        return outputs

    model.forward = MethodType(forward, model)


def patch_qwen3_vl_lidar_modality_model(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    if getattr(model.config, "model_type", None) not in {"qwen3_vl", "qwen3_vl_moe"}:
        return

    if not model_args.use_lidar_modality:
        return

    def _freeze_module_params(module: nn.Module | None) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(False)

    text_hidden_size = model.config.text_config.hidden_size
    effective_use_lidar_object_geometry_features = (
        model_args.use_lidar_object_geometry_features and model_args.lidar_encoder_mode == "full"
    )
    effective_use_lidar_object_numeric_features = (
        model_args.use_lidar_object_numeric_features and model_args.lidar_encoder_mode == "full"
    )
    effective_use_lidar_scene_features = model_args.use_lidar_scene_features
    effective_use_lidar_template_router = model_args.use_lidar_template_router and model_args.use_lidar_decoder_adapter
    model.config.use_lidar_modality = True
    model.config.use_lidar_object_features = model_args.use_lidar_object_features
    model.config.use_lidar_object_geometry_features = effective_use_lidar_object_geometry_features
    model.config.use_lidar_object_numeric_features = effective_use_lidar_object_numeric_features
    model.config.use_lidar_scene_features = effective_use_lidar_scene_features
    model.config.use_lidar_object_memory = model_args.use_lidar_object_memory
    model.config.use_lidar_scene_memory = model_args.use_lidar_scene_memory
    model.config.use_lidar_template_router = effective_use_lidar_template_router
    model.config.lidar_encoder_mode = model_args.lidar_encoder_mode
    model.config.lidar_object_memory_mode = model_args.lidar_object_memory_mode
    model.config.lidar_scene_memory_mode = model_args.lidar_scene_memory_mode
    model.config.lidar_injection_mode = model_args.lidar_injection_mode
    model.config.use_lidar_decoder_adapter = model_args.use_lidar_decoder_adapter
    model.config.lidar_input_token = model_args.lidar_input_token
    model.config.lidar_input_token_id = None
    model.config.lidar_ablation_tag = model_args.lidar_ablation_tag
    model.config.lidar_template_family_names = list(LIDAR_TEMPLATE_FAMILY_NAMES)
    model.config.lidar_hidden_size = model_args.lidar_hidden_size
    model.config.lidar_object_prefix_length = model_args.lidar_object_prefix_length
    model.config.lidar_object_token_dim = model_args.lidar_object_token_dim
    model.config.lidar_object_geometry_token_dim = model_args.lidar_object_geometry_token_dim
    model.config.lidar_object_numeric_token_dim = model_args.lidar_object_numeric_token_dim
    model.config.lidar_object_numeric_label_vocab_size = model_args.lidar_object_numeric_label_vocab_size
    model.config.lidar_object_numeric_label_emb_dim = model_args.lidar_object_numeric_label_emb_dim
    model.config.lidar_scene_token_dim = model_args.lidar_scene_token_dim
    model.config.lidar_scene_token_length = model_args.lidar_scene_token_length
    model.config.lidar_scene_memory_length = model_args.lidar_scene_memory_length
    model.config.lidar_cross_attention_heads = model_args.lidar_cross_attention_heads
    model.config.lidar_decoder_adapter_layers = model_args.lidar_decoder_adapter_layers
    model.config.lidar_decoder_adapter_position = model_args.lidar_decoder_adapter_position
    model.config.lidar_decoder_adapter_heads = model_args.lidar_decoder_adapter_heads
    model.config.lidar_decoder_adapter_gate_bias = model_args.lidar_decoder_adapter_gate_bias
    model.config.lidar_template_router_output_dim = model_args.lidar_template_router_output_dim
    model.config.lidar_numeric_count_weight = model_args.lidar_numeric_count_weight
    model.config.lidar_numeric_motion_weight = model_args.lidar_numeric_motion_weight
    model.config.lidar_numeric_coord_weight = model_args.lidar_numeric_coord_weight
    model.config.lidar_gate_usage_weight = model_args.lidar_gate_usage_weight
    model.config.use_lidar_numeric_supervision = any(
        weight > 0.0
        for weight in (
            model_args.lidar_numeric_count_weight,
            model_args.lidar_numeric_motion_weight,
            model_args.lidar_numeric_coord_weight,
        )
    )
    lower_model = model.model
    lower_model.use_lidar_modality = True
    lower_model.use_lidar_object_features = model_args.use_lidar_object_features
    lower_model.use_lidar_object_geometry_features = effective_use_lidar_object_geometry_features
    lower_model.use_lidar_object_numeric_features = effective_use_lidar_object_numeric_features
    lower_model.use_lidar_scene_features = effective_use_lidar_scene_features
    lower_model.use_lidar_object_memory = model_args.use_lidar_object_memory
    lower_model.use_lidar_scene_memory = model_args.use_lidar_scene_memory
    lower_model.use_lidar_template_router = effective_use_lidar_template_router
    lower_model.lidar_encoder_mode = model_args.lidar_encoder_mode
    lower_model.lidar_object_memory_mode = model_args.lidar_object_memory_mode
    lower_model.lidar_scene_memory_mode = model_args.lidar_scene_memory_mode
    lower_model.lidar_injection_mode = model_args.lidar_injection_mode
    lower_model.use_lidar_decoder_adapter = model_args.use_lidar_decoder_adapter
    lower_model.lidar_decoder_adapter_position = model_args.lidar_decoder_adapter_position
    lower_model.lidar_input_token = model_args.lidar_input_token
    lower_model.lidar_input_token_id = None
    lower_model.lidar_ablation_tag = model_args.lidar_ablation_tag
    lower_model.lidar_object_prefix_length = model_args.lidar_object_prefix_length
    lower_model.lidar_scene_memory_length = model_args.lidar_scene_memory_length
    if model_args.lidar_encoder_mode == "projection_only":
        lower_model.lidar_object_encoder = TokenProjectionEncoder(
            input_dim=model_args.lidar_object_token_dim,
            hidden_dim=model_args.lidar_hidden_size,
        )
    else:
        lower_model.lidar_object_encoder = ObjectPrefixEncoder(
            input_dim=model_args.lidar_object_token_dim,
            hidden_dim=model_args.lidar_hidden_size,
        )
    lower_model.lidar_object_geometry_encoder = None
    if effective_use_lidar_object_geometry_features:
        lower_model.lidar_object_geometry_encoder = ObjectGeometryEncoder(
            input_dim=model_args.lidar_object_geometry_token_dim,
            hidden_dim=model_args.lidar_hidden_size,
        )
    lower_model.lidar_object_numeric_encoder = None
    if effective_use_lidar_object_numeric_features:
        lower_model.lidar_object_numeric_encoder = ObjectNumericEncoder(
            input_dim=model_args.lidar_object_numeric_token_dim,
            label_vocab_size=model_args.lidar_object_numeric_label_vocab_size,
            label_emb_dim=model_args.lidar_object_numeric_label_emb_dim,
            hidden_dim=model_args.lidar_hidden_size,
        )
    lower_model.lidar_scene_encoder = None
    lower_model.lidar_object_connector = None
    lower_model.lidar_scene_connector = None
    if effective_use_lidar_scene_features:
        if model_args.lidar_encoder_mode == "projection_only":
            lower_model.lidar_scene_encoder = TokenProjectionEncoder(
                input_dim=model_args.lidar_scene_token_dim,
                hidden_dim=model_args.lidar_hidden_size,
            )
        else:
            lower_model.lidar_scene_encoder = SceneEncoder(
                input_dim=model_args.lidar_scene_token_dim,
                hidden_dim=model_args.lidar_hidden_size,
            )
    lower_model.lidar_decoder_adapter_blocks = None
    lower_model.lidar_object_direct_proj = None
    lower_model.lidar_scene_direct_proj = None
    lower_model.lidar_object_injection_proj = None
    lower_model.lidar_scene_injection_proj = None
    lower_model.lidar_vision_fusion_proj = None
    lower_model.lidar_vision_fusion_gate = None
    if model_args.use_lidar_object_memory:
        if model_args.lidar_object_memory_mode == "connector":
            lower_model.lidar_object_connector = ObjectQueryConnector(
                input_dim=model_args.lidar_hidden_size,
                output_dim=text_hidden_size,
                query_length=model_args.lidar_object_prefix_length,
                num_heads=model_args.lidar_cross_attention_heads,
            )
        else:
            lower_model.lidar_object_direct_proj = nn.Linear(model_args.lidar_hidden_size, text_hidden_size)
    if effective_use_lidar_scene_features and model_args.use_lidar_scene_memory:
        if model_args.lidar_scene_memory_mode == "connector":
            lower_model.lidar_scene_connector = SceneLatentConnector(
                input_dim=model_args.lidar_hidden_size,
                output_dim=text_hidden_size,
                latent_length=model_args.lidar_scene_memory_length,
                num_heads=model_args.lidar_cross_attention_heads,
            )
        else:
            lower_model.lidar_scene_direct_proj = nn.Linear(model_args.lidar_hidden_size, text_hidden_size)
    if model_args.lidar_injection_mode in {"input_tokens", "vision_fusion"}:
        if model_args.use_lidar_object_features:
            lower_model.lidar_object_injection_proj = nn.Linear(model_args.lidar_hidden_size, text_hidden_size)
        if effective_use_lidar_scene_features:
            lower_model.lidar_scene_injection_proj = nn.Linear(model_args.lidar_hidden_size, text_hidden_size)
        if model_args.lidar_injection_mode == "vision_fusion":
            lower_model.lidar_vision_fusion_proj = nn.Linear(text_hidden_size, text_hidden_size)
            lower_model.lidar_vision_fusion_gate = nn.Linear(1, 1)
            nn.init.zeros_(lower_model.lidar_vision_fusion_gate.weight)
            nn.init.constant_(lower_model.lidar_vision_fusion_gate.bias, -6.0)
    if model_args.use_lidar_decoder_adapter:
        lower_model.lidar_decoder_adapter_blocks = nn.ModuleList(
            [
                LidarDecoderAdapterBlock(
                    hidden_dim=text_hidden_size,
                    num_heads=model_args.lidar_decoder_adapter_heads,
                    gate_bias=model_args.lidar_decoder_adapter_gate_bias,
                )
                for _ in range(model_args.lidar_decoder_adapter_layers)
            ]
        )
    lower_model.lidar_object_fusion_gate = None
    if effective_use_lidar_object_geometry_features:
        lower_model.lidar_object_fusion_gate = nn.Linear(model_args.lidar_hidden_size * 2, 1)
    lower_model.lidar_object_numeric_fusion_gate = None
    if effective_use_lidar_object_numeric_features:
        lower_model.lidar_object_numeric_fusion_gate = nn.Linear(model_args.lidar_hidden_size * 2, 1)
    lower_model.lidar_template_router = None
    if effective_use_lidar_template_router:
        lower_model.lidar_template_router = LidarTemplateRouter(
            LIDAR_TEMPLATE_ROUTE_INITIAL_SCALES,
            output_dim=model_args.lidar_template_router_output_dim,
        )
    if not model_args.use_lidar_decoder_adapter and model_args.lidar_injection_mode == "decoder_adapter":
        for module in (
            lower_model.lidar_object_encoder,
            lower_model.lidar_object_geometry_encoder,
            lower_model.lidar_object_numeric_encoder,
            lower_model.lidar_scene_encoder,
            lower_model.lidar_object_connector,
            lower_model.lidar_scene_connector,
            lower_model.lidar_object_direct_proj,
            lower_model.lidar_scene_direct_proj,
            lower_model.lidar_object_injection_proj,
            lower_model.lidar_scene_injection_proj,
            lower_model.lidar_vision_fusion_proj,
            lower_model.lidar_object_fusion_gate,
            lower_model.lidar_object_numeric_fusion_gate,
            lower_model.lidar_template_router,
        ):
            _freeze_module_params(module)
    if model.config.use_lidar_numeric_supervision:
        model.lidar_count_head = LidarNumericHead(text_hidden_size, 2)
        model.lidar_motion_head = LidarNumericHead(text_hidden_size, 3)
        model.lidar_coord_head = LidarNumericHead(text_hidden_size, 10)

    text_model = lower_model.language_model
    object.__setattr__(text_model, "_llamafactory_lidar_parent_ref", weakref.ref(lower_model))
    lower_model._cached_lidar_agent_memory = None
    lower_model._cached_lidar_agent_mask = None
    lower_model._cached_lidar_scene_memory = None
    lower_model._cached_lidar_scene_mask = None
    lower_model._cached_lidar_agent_route_scale = None
    lower_model._cached_lidar_scene_route_scale = None
    lower_model._cached_lidar_output_route_scale = None
    lower_model._cached_lidar_grad_anchor = None
    lower_model._cached_lidar_decoder_adapter_layer_indices = None

    def _clear_lidar_decoder_adapter_cache(self) -> None:
        self._cached_lidar_agent_memory = None
        self._cached_lidar_agent_mask = None
        self._cached_lidar_scene_memory = None
        self._cached_lidar_scene_mask = None
        self._cached_lidar_agent_route_scale = None
        self._cached_lidar_scene_route_scale = None
        self._cached_lidar_output_route_scale = None
        self._cached_lidar_grad_anchor = None
        self._cached_lidar_decoder_adapter_layer_indices = None

    lower_model.clear_lidar_decoder_adapter_cache = MethodType(_clear_lidar_decoder_adapter_cache, lower_model)

    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

    def _resolve_lidar_decoder_adapter_layer_indices(self, num_layers: int) -> list[int]:
        adapter_blocks = getattr(self, "lidar_decoder_adapter_blocks", None)
        if adapter_blocks is None or len(adapter_blocks) == 0:
            return []

        num_adapter_layers = min(len(adapter_blocks), num_layers)
        position = getattr(self, "lidar_decoder_adapter_position", "tail")
        if position == "front":
            start = 0
        elif position == "middle":
            start = max((num_layers - num_adapter_layers) // 2, 0)
        else:
            start = max(num_layers - num_adapter_layers, 0)
        return list(range(start, start + num_adapter_layers))

    lower_model.resolve_lidar_decoder_adapter_layer_indices = MethodType(
        _resolve_lidar_decoder_adapter_layer_indices,
        lower_model,
    )

    def text_forward_with_lidar_decoder_adapter(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        lidar_agent_memory = kwargs.pop("lidar_agent_memory", None)
        lidar_agent_mask = kwargs.pop("lidar_agent_mask", None)
        lidar_scene_memory = kwargs.pop("lidar_scene_memory", None)
        lidar_scene_mask = kwargs.pop("lidar_scene_mask", None)
        lidar_agent_route_scale = kwargs.pop("lidar_agent_route_scale", None)
        lidar_scene_route_scale = kwargs.pop("lidar_scene_route_scale", None)
        lidar_output_route_scale = kwargs.pop("lidar_output_route_scale", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        lidar_parent_ref = getattr(self, "_llamafactory_lidar_parent_ref", None)
        lidar_parent = lidar_parent_ref() if lidar_parent_ref is not None else None
        adapter_blocks = getattr(lidar_parent, "lidar_decoder_adapter_blocks", None) if lidar_parent is not None else None
        use_decoder_adapter = lidar_parent is not None and adapter_blocks is not None
        adapter_layer_indices = (
            lidar_parent.resolve_lidar_decoder_adapter_layer_indices(len(self.layers)) if use_decoder_adapter else []
        )
        if lidar_parent is not None:
            lidar_parent._cached_lidar_decoder_adapter_layer_indices = adapter_layer_indices
        adapter_layer_map = {layer_idx: block_idx for block_idx, layer_idx in enumerate(adapter_layer_indices)}

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            block_idx = adapter_layer_map.get(layer_idx)
            if use_decoder_adapter and block_idx is not None:
                adapter_block = adapter_blocks[block_idx]
                hidden_states = adapter_block(
                    hidden_states,
                    agent_memory=lidar_agent_memory,
                    agent_mask=lidar_agent_mask,
                    scene_memory=lidar_scene_memory,
                    scene_mask=lidar_scene_mask,
                    agent_route_scale=lidar_agent_route_scale,
                    scene_route_scale=lidar_scene_route_scale,
                    output_route_scale=lidar_output_route_scale,
                )

        hidden_states = self.norm(hidden_states)
        if getattr(model.config, "model_type", None) == "qwen3_vl_moe":
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                router_logits=None,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    text_model.forward = MethodType(text_forward_with_lidar_decoder_adapter, text_model)

    def _normalize_token_mask(self, hidden_states: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if token_mask is None:
            return torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=hidden_states.dtype)
        return token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _lidar_anchor_param(self) -> torch.nn.Parameter:
        for module_name in (
            "lidar_object_encoder",
            "lidar_scene_encoder",
            "lidar_object_direct_proj",
            "lidar_scene_direct_proj",
            "lidar_object_injection_proj",
            "lidar_scene_injection_proj",
            "lidar_vision_fusion_proj",
        ):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            try:
                return next(module.parameters())
            except StopIteration:
                continue
        raise ValueError("No LiDAR anchor parameter is available on the current model.")

    def _build_direct_memory(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
        projector: nn.Module,
        memory_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_states = projector(hidden_states)
        if token_mask is None:
            valid_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            valid_mask = token_mask.to(device=hidden_states.device, dtype=torch.bool)

        batch_size = projected_states.size(0)
        memory_states = projected_states.new_zeros((batch_size, memory_length, projected_states.size(-1)))
        memory_mask = torch.zeros((batch_size, memory_length), device=hidden_states.device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            current_mask = valid_mask[batch_idx]
            if current_mask.ndim == 0:
                current_mask = current_mask.view(1)
            selected = projected_states[batch_idx][current_mask]
            keep_count = min(int(selected.size(0)), memory_length)
            if keep_count <= 0:
                continue
            memory_states[batch_idx, :keep_count] = selected[:keep_count]
            memory_mask[batch_idx, :keep_count] = True
        return memory_states, memory_mask

    def _build_lidar_injection_tokens(
        self,
        lidar_object_features: torch.Tensor | None,
        lidar_object_feature_mask: torch.Tensor | None = None,
        lidar_scene_features: torch.Tensor | None = None,
        lidar_scene_feature_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        object_hidden_states, object_mask, scene_hidden_states, scene_mask = self._encode_lidar_hidden_states(
            lidar_object_features,
            lidar_object_feature_mask=lidar_object_feature_mask,
            lidar_scene_features=lidar_scene_features,
            lidar_scene_feature_mask=lidar_scene_feature_mask,
        )

        token_states: list[torch.Tensor] = []
        token_masks: list[torch.Tensor] = []
        if object_hidden_states is not None and self.lidar_object_injection_proj is not None:
            object_tokens, object_token_mask = self._build_direct_memory(
                object_hidden_states,
                object_mask > 0 if object_mask is not None else None,
                self.lidar_object_injection_proj,
                self.lidar_object_prefix_length,
            )
            token_states.append(object_tokens)
            token_masks.append(object_token_mask)
        if scene_hidden_states is not None and self.lidar_scene_injection_proj is not None:
            scene_tokens, scene_token_mask = self._build_direct_memory(
                scene_hidden_states,
                scene_mask > 0 if scene_mask is not None else None,
                self.lidar_scene_injection_proj,
                self.lidar_scene_memory_length,
            )
            token_states.append(scene_tokens)
            token_masks.append(scene_token_mask)

        if not token_states:
            return None, None
        return torch.cat(token_states, dim=1), torch.cat(token_masks, dim=1)

    def _build_lidar_grad_anchor(self) -> torch.Tensor:
        anchor = self._lidar_anchor_param().new_zeros(())
        module_names: list[str] = []
        if getattr(self, "lidar_injection_mode", "decoder_adapter") in {"input_tokens", "vision_fusion"}:
            module_names.extend(
                (
                    "lidar_object_encoder",
                    "lidar_scene_encoder",
                    "lidar_object_injection_proj",
                    "lidar_scene_injection_proj",
                    "lidar_vision_fusion_proj",
                )
            )
        if getattr(self, "use_lidar_object_memory", True):
            module_names.extend(
                (
                    "lidar_object_encoder",
                    "lidar_object_geometry_encoder",
                    "lidar_object_numeric_encoder",
                    "lidar_object_connector",
                    "lidar_object_direct_proj",
                    "lidar_object_fusion_gate",
                    "lidar_object_numeric_fusion_gate",
                )
            )
        if getattr(self, "use_lidar_scene_memory", True):
            module_names.extend(
                (
                    "lidar_scene_encoder",
                    "lidar_scene_connector",
                    "lidar_scene_direct_proj",
                )
            )
        if getattr(self, "use_lidar_decoder_adapter", True):
            module_names.extend(
                (
                    "lidar_decoder_adapter_blocks",
                    "lidar_template_router",
                )
            )
        for module_name in module_names:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad:
                    anchor = anchor + param.reshape(-1).sum() * 0.0
        return anchor

    def _encode_lidar_hidden_states(
        self,
        lidar_object_features: torch.Tensor | None,
        lidar_object_feature_mask: torch.Tensor | None = None,
        lidar_object_geometry_features: torch.Tensor | None = None,
        lidar_object_geometry_feature_mask: torch.Tensor | None = None,
        lidar_object_numeric_features: torch.Tensor | None = None,
        lidar_object_numeric_feature_mask: torch.Tensor | None = None,
        lidar_object_numeric_label_ids: torch.Tensor | None = None,
        lidar_scene_features: torch.Tensor | None = None,
        lidar_scene_feature_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        object_hidden_states = None
        object_mask = None
        scene_hidden_states = None
        scene_mask = None

        if lidar_object_features is not None:
            object_param = self._lidar_anchor_param()
            lidar_object_features = lidar_object_features.to(
                device=object_param.device,
                dtype=object_param.dtype,
            )
            object_hidden_states = self.lidar_object_encoder(lidar_object_features)
            object_mask = self._normalize_token_mask(object_hidden_states, lidar_object_feature_mask)

        if lidar_object_geometry_features is not None:
            if object_hidden_states is None or object_mask is None:
                raise ValueError("LiDAR object geometry features require LiDAR object features in the same batch.")
            if self.lidar_object_geometry_encoder is None:
                raise ValueError(
                    "LiDAR object geometry features were provided, but the LiDAR object geometry encoder is disabled."
                )
            lidar_object_geometry_features = lidar_object_geometry_features.to(
                device=object_hidden_states.device,
                dtype=object_hidden_states.dtype,
            )
            geometry_hidden_states = self.lidar_object_geometry_encoder(lidar_object_geometry_features)
            geometry_mask = self._normalize_token_mask(geometry_hidden_states, lidar_object_geometry_feature_mask)
            fusion_gate = torch.sigmoid(
                self.lidar_object_fusion_gate(torch.cat([object_hidden_states, geometry_hidden_states], dim=-1))
            )
            object_hidden_states = (
                object_hidden_states
                + fusion_gate * geometry_hidden_states * (object_mask * geometry_mask).unsqueeze(-1)
            )

        if lidar_object_numeric_features is not None:
            if object_hidden_states is None or object_mask is None:
                raise ValueError("LiDAR object numeric features require LiDAR object features in the same batch.")
            if self.lidar_object_numeric_encoder is None or self.lidar_object_numeric_fusion_gate is None:
                raise ValueError(
                    "LiDAR object numeric features were provided, but the LiDAR object numeric encoder is disabled."
                )
            if lidar_object_numeric_label_ids is None:
                raise ValueError("LiDAR object numeric features require numeric label ids in the same batch.")
            lidar_object_numeric_features = lidar_object_numeric_features.to(
                device=object_hidden_states.device,
                dtype=object_hidden_states.dtype,
            )
            lidar_object_numeric_label_ids = lidar_object_numeric_label_ids.to(
                device=object_hidden_states.device,
                dtype=torch.long,
            )
            numeric_hidden_states = self.lidar_object_numeric_encoder(
                lidar_object_numeric_features,
                lidar_object_numeric_label_ids,
            )
            numeric_mask = self._normalize_token_mask(numeric_hidden_states, lidar_object_numeric_feature_mask)
            numeric_fusion_gate = torch.sigmoid(
                self.lidar_object_numeric_fusion_gate(torch.cat([object_hidden_states, numeric_hidden_states], dim=-1))
            )
            object_hidden_states = (
                object_hidden_states
                + numeric_fusion_gate * numeric_hidden_states * (object_mask * numeric_mask).unsqueeze(-1)
            )

        if lidar_scene_features is not None:
            if self.lidar_scene_encoder is None:
                raise ValueError("LiDAR scene features were provided, but the LiDAR scene encoder is disabled.")
            if object_hidden_states is not None:
                scene_device = object_hidden_states.device
                scene_dtype = object_hidden_states.dtype
            else:
                scene_param = next(self.lidar_scene_encoder.parameters())
                scene_device = scene_param.device
                scene_dtype = scene_param.dtype
            lidar_scene_features = lidar_scene_features.to(
                device=scene_device,
                dtype=scene_dtype,
            )
            scene_hidden_states = self.lidar_scene_encoder(lidar_scene_features)
            scene_mask = self._normalize_token_mask(scene_hidden_states, lidar_scene_feature_mask)

        return object_hidden_states, object_mask, scene_hidden_states, scene_mask

    def _encode_lidar_external_memory(
        self,
        lidar_object_features: torch.Tensor | None,
        lidar_object_feature_mask: torch.Tensor | None = None,
        lidar_object_geometry_features: torch.Tensor | None = None,
        lidar_object_geometry_feature_mask: torch.Tensor | None = None,
        lidar_object_numeric_features: torch.Tensor | None = None,
        lidar_object_numeric_feature_mask: torch.Tensor | None = None,
        lidar_object_numeric_label_ids: torch.Tensor | None = None,
        lidar_scene_features: torch.Tensor | None = None,
        lidar_scene_feature_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        object_hidden_states, object_mask, scene_hidden_states, scene_mask = self._encode_lidar_hidden_states(
            lidar_object_features,
            lidar_object_feature_mask=lidar_object_feature_mask,
            lidar_object_geometry_features=lidar_object_geometry_features,
            lidar_object_geometry_feature_mask=lidar_object_geometry_feature_mask,
            lidar_object_numeric_features=lidar_object_numeric_features,
            lidar_object_numeric_feature_mask=lidar_object_numeric_feature_mask,
            lidar_object_numeric_label_ids=lidar_object_numeric_label_ids,
            lidar_scene_features=lidar_scene_features,
            lidar_scene_feature_mask=lidar_scene_feature_mask,
        )

        if self.use_lidar_object_memory and object_hidden_states is not None and object_mask is not None:
            agent_mask = object_mask > 0
            if self.lidar_object_memory_mode == "direct":
                if self.lidar_object_direct_proj is None:
                    raise ValueError("LiDAR object direct projector is required in direct object-memory mode.")
                agent_memory, agent_mask = self._build_direct_memory(
                    object_hidden_states,
                    agent_mask,
                    self.lidar_object_direct_proj,
                    self.lidar_object_prefix_length,
                )
            else:
                if self.lidar_object_connector is None:
                    raise ValueError("LiDAR object connector is required on the current main path.")
                agent_memory, agent_mask, _, _ = self.lidar_object_connector(object_hidden_states, agent_mask)
        else:
            agent_memory, agent_mask = None, None
        if self.use_lidar_scene_memory and scene_hidden_states is not None:
            if self.lidar_scene_memory_mode == "direct":
                if self.lidar_scene_direct_proj is None:
                    raise ValueError("LiDAR scene direct projector is required in direct scene-memory mode.")
                scene_memory, scene_memory_mask = self._build_direct_memory(
                    scene_hidden_states,
                    scene_mask,
                    self.lidar_scene_direct_proj,
                    self.lidar_scene_memory_length,
                )
            else:
                if self.lidar_scene_connector is None:
                    raise ValueError("LiDAR scene connector is required when LiDAR scene features are enabled.")
                scene_memory, scene_memory_mask = self.lidar_scene_connector(scene_hidden_states, scene_mask)
        else:
            scene_memory, scene_memory_mask = None, None

        if scene_memory is not None:
            scene_memory = scene_memory * scene_memory_mask.unsqueeze(-1).to(dtype=scene_memory.dtype)
        return agent_memory, agent_mask, scene_memory, scene_memory_mask

    def _pool_response_hidden_states(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            pooled = hidden_states.mean(dim=1)
            valid_mask = torch.ones(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)
            return pooled, valid_mask

        response_mask = labels.ne(IGNORE_INDEX)
        valid_mask = response_mask.any(dim=1)
        pooled = hidden_states.new_zeros((hidden_states.size(0), hidden_states.size(-1)))
        if bool(valid_mask.any()):
            weights = response_mask.to(dtype=hidden_states.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_values = (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / denom
            pooled[valid_mask] = pooled_values[valid_mask]
        return pooled, valid_mask

    def _masked_smooth_l1_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if targets is None or target_mask is None:
            return predictions.new_zeros(())
        targets = targets.to(device=predictions.device, dtype=predictions.dtype)
        target_mask = target_mask.to(device=predictions.device, dtype=predictions.dtype)
        if not bool(target_mask.gt(0.0).any()):
            return predictions.new_zeros(())
        raw_loss = torch.nn.functional.smooth_l1_loss(predictions, targets, reduction="none")
        return (raw_loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)

    def _collect_lidar_adapter_metrics(self) -> tuple[dict[str, float], torch.Tensor]:
        adapter_blocks = getattr(self, "lidar_decoder_adapter_blocks", None)
        if adapter_blocks is None or len(adapter_blocks) == 0:
            return {}, self._lidar_anchor_param().new_zeros(())

        metric_names = (
            "adapter_scene_gate_mean",
            "adapter_agent_gate_mean",
            "adapter_output_gate_mean",
            "adapter_scene_delta_norm",
            "adapter_agent_delta_norm",
            "adapter_scene_route_scale_mean",
            "adapter_agent_route_scale_mean",
            "adapter_output_route_scale_mean",
        )
        aggregated = {name: 0.0 for name in metric_names}
        regularizers = []
        for block in adapter_blocks:
            metrics = getattr(block, "_last_adapter_metrics", {})
            for name in metric_names:
                aggregated[name] += float(metrics.get(name, 0.0))
            regularizer = getattr(block, "_last_gate_usage_regularizer", None)
            if regularizer is not None:
                regularizers.append(regularizer)

        block_count = float(len(adapter_blocks))
        aggregated = {name: value / block_count for name, value in aggregated.items()}
        if regularizers:
            return aggregated, torch.stack(regularizers).mean()
        return aggregated, self._lidar_anchor_param().new_zeros(())

    def _collect_lidar_adapter_sample_metrics(self) -> dict[str, torch.Tensor]:
        adapter_blocks = getattr(self, "lidar_decoder_adapter_blocks", None)
        if adapter_blocks is None or len(adapter_blocks) == 0:
            return {}

        metric_names = (
            "adapter_scene_gate_mean",
            "adapter_agent_gate_mean",
            "adapter_output_gate_mean",
            "adapter_scene_delta_norm",
            "adapter_agent_delta_norm",
            "adapter_scene_route_scale_mean",
            "adapter_agent_route_scale_mean",
            "adapter_output_route_scale_mean",
        )
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}
        for block in adapter_blocks:
            metrics = getattr(block, "_last_adapter_sample_metrics", None)
            if not isinstance(metrics, dict):
                continue
            for name in metric_names:
                tensor = metrics.get(name)
                if tensor is not None:
                    collected[name].append(tensor)

        sample_metrics: dict[str, torch.Tensor] = {}
        for name, tensors in collected.items():
            if not tensors:
                continue
            stacked = torch.stack([tensor.to(dtype=torch.float32) for tensor in tensors], dim=0)
            sample_metrics[name] = stacked.mean(dim=0).detach().cpu()
        return sample_metrics

    lower_model._normalize_token_mask = MethodType(_normalize_token_mask, lower_model)
    lower_model._lidar_anchor_param = MethodType(_lidar_anchor_param, lower_model)
    lower_model._build_direct_memory = MethodType(_build_direct_memory, lower_model)
    lower_model._build_lidar_injection_tokens = MethodType(_build_lidar_injection_tokens, lower_model)
    lower_model._build_lidar_grad_anchor = MethodType(_build_lidar_grad_anchor, lower_model)
    lower_model._encode_lidar_hidden_states = MethodType(_encode_lidar_hidden_states, lower_model)
    lower_model.encode_lidar_external_memory = MethodType(_encode_lidar_external_memory, lower_model)
    lower_model._pool_response_hidden_states = MethodType(_pool_response_hidden_states, lower_model)
    lower_model._masked_smooth_l1_loss = MethodType(_masked_smooth_l1_loss, lower_model)
    lower_model._collect_lidar_adapter_metrics = MethodType(_collect_lidar_adapter_metrics, lower_model)
    lower_model._collect_lidar_adapter_sample_metrics = MethodType(_collect_lidar_adapter_sample_metrics, lower_model)

    original_lower_forward = lower_model.forward

    def lower_forward_with_lidar_modality(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        lidar_object_features=None,
        lidar_object_feature_mask=None,
        lidar_object_geometry_features=None,
        lidar_object_geometry_feature_mask=None,
        lidar_object_numeric_features=None,
        lidar_object_numeric_feature_mask=None,
        lidar_object_numeric_label_ids=None,
        lidar_scene_features=None,
        lidar_scene_feature_mask=None,
        lidar_template_family_ids=None,
        **kwargs,
    ):
        use_object_features = lidar_object_features is not None
        use_object_geometry_features = lidar_object_geometry_features is not None
        use_object_numeric_features = lidar_object_numeric_features is not None
        use_scene_features = lidar_scene_features is not None
        has_new_lidar_features = (
            use_object_features
            or use_object_geometry_features
            or use_object_numeric_features
            or use_scene_features
        )
        if past_key_values is None and not has_new_lidar_features:
            self.clear_lidar_decoder_adapter_cache()
        has_cached_lidar_memory = (
            self._cached_lidar_agent_memory is not None
            or self._cached_lidar_scene_memory is not None
        )
        if (
            not has_new_lidar_features
            and not has_cached_lidar_memory
        ):
            if past_key_values is None:
                self.clear_lidar_decoder_adapter_cache()
            return original_lower_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        if input_ids is None or inputs_embeds is not None:
            raise ValueError("LiDAR modality injection requires input_ids and does not support explicit inputs_embeds.")
        if not use_object_features and (
            use_object_geometry_features
            or use_object_numeric_features
        ):
            raise ValueError(
                "LiDAR object geometry or numeric features require LiDAR object features in the same batch."
            )

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(device=input_ids.device)
        attention_mask = attention_mask.clone()

        image_mask = None
        video_mask = None
        image_token_counts = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            image_token_counts = image_mask[..., 0].sum(dim=1).to(dtype=torch.long)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack_image_embeds = image_outputs.deepstack_features
        else:
            deepstack_image_embeds = None

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            deepstack_video_embeds = video_outputs.deepstack_features
        else:
            deepstack_video_embeds = None

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        lidar_agent_memory = None
        lidar_agent_mask = None
        lidar_scene_memory = None
        lidar_scene_memory_mask = None
        lidar_injection_tokens = None
        lidar_injection_mask = None
        lidar_agent_route_scale = None
        lidar_scene_route_scale = None
        lidar_output_route_scale = None
        lidar_injection_mode = getattr(self, "lidar_injection_mode", "decoder_adapter")
        self._cached_lidar_grad_anchor = self._build_lidar_grad_anchor()
        if has_new_lidar_features:
            if lidar_injection_mode == "decoder_adapter":
                lidar_agent_memory, lidar_agent_mask, lidar_scene_memory, lidar_scene_memory_mask = self.encode_lidar_external_memory(
                    lidar_object_features,
                    lidar_object_feature_mask,
                    lidar_object_geometry_features=lidar_object_geometry_features,
                    lidar_object_geometry_feature_mask=lidar_object_geometry_feature_mask,
                    lidar_object_numeric_features=lidar_object_numeric_features,
                    lidar_object_numeric_feature_mask=lidar_object_numeric_feature_mask,
                    lidar_object_numeric_label_ids=lidar_object_numeric_label_ids,
                    lidar_scene_features=lidar_scene_features,
                    lidar_scene_feature_mask=lidar_scene_feature_mask,
                )
                if self.use_lidar_template_router and self.lidar_template_router is not None:
                    if lidar_template_family_ids is None:
                        lidar_template_family_ids = torch.zeros(
                            (input_ids.size(0),),
                            device=inputs_embeds.device,
                            dtype=torch.long,
                        )
                    route_scales = self.lidar_template_router(lidar_template_family_ids)
                    route_scales = route_scales.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    lidar_agent_route_scale = route_scales[:, 0]
                    lidar_scene_route_scale = route_scales[:, 1]
                    if route_scales.size(-1) > 2:
                        lidar_output_route_scale = route_scales[:, 2]
                self._cached_lidar_agent_memory = lidar_agent_memory
                self._cached_lidar_agent_mask = lidar_agent_mask
                self._cached_lidar_scene_memory = lidar_scene_memory
                self._cached_lidar_scene_mask = lidar_scene_memory_mask
                self._cached_lidar_agent_route_scale = lidar_agent_route_scale
                self._cached_lidar_scene_route_scale = lidar_scene_route_scale
                self._cached_lidar_output_route_scale = lidar_output_route_scale
            else:
                self._cached_lidar_agent_memory = None
                self._cached_lidar_agent_mask = None
                self._cached_lidar_scene_memory = None
                self._cached_lidar_scene_mask = None
                self._cached_lidar_agent_route_scale = None
                self._cached_lidar_scene_route_scale = None
                self._cached_lidar_output_route_scale = None
                lidar_injection_tokens, lidar_injection_mask = self._build_lidar_injection_tokens(
                    lidar_object_features,
                    lidar_object_feature_mask=lidar_object_feature_mask,
                    lidar_scene_features=lidar_scene_features,
                    lidar_scene_feature_mask=lidar_scene_feature_mask,
                )

        if lidar_injection_tokens is not None and lidar_injection_mode == "input_tokens":
            lidar_input_token_id = getattr(self, "lidar_input_token_id", None)
            if lidar_input_token_id is None or lidar_input_token_id < 0:
                raise ValueError("LiDAR input-token injection requires a valid lidar_input_token_id.")
            placeholder_mask = input_ids.eq(lidar_input_token_id)
            expected_count = lidar_injection_tokens.size(1)
            actual_counts = placeholder_mask.sum(dim=1)
            if not bool(actual_counts.eq(expected_count).all()):
                raise ValueError(
                    "LiDAR input-token injection requires each sample to contain exactly "
                    f"{expected_count} LiDAR placeholder tokens."
                )
            for batch_idx in range(inputs_embeds.size(0)):
                current_positions = placeholder_mask[batch_idx].nonzero(as_tuple=False).view(-1)
                inputs_embeds[batch_idx, current_positions] = lidar_injection_tokens[batch_idx]

        if lidar_injection_tokens is not None and lidar_injection_mode == "vision_fusion":
            if image_embeds is None or image_token_counts is None:
                raise ValueError("LiDAR vision-fusion injection requires image inputs in the same batch.")
            if self.lidar_vision_fusion_proj is None or self.lidar_vision_fusion_gate is None:
                raise ValueError("LiDAR vision-fusion injection modules are missing on the current model.")
            lidar_mask = (
                lidar_injection_mask.to(device=lidar_injection_tokens.device, dtype=lidar_injection_tokens.dtype)
                if lidar_injection_mask is not None
                else torch.ones(lidar_injection_tokens.shape[:2], device=lidar_injection_tokens.device, dtype=lidar_injection_tokens.dtype)
            )
            pooled_lidar = (lidar_injection_tokens * lidar_mask.unsqueeze(-1)).sum(dim=1) / lidar_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            fused_lidar = self.lidar_vision_fusion_proj(pooled_lidar).to(dtype=image_embeds.dtype)
            gate_inputs = fused_lidar.new_ones((fused_lidar.size(0), 1))
            fusion_gate = torch.sigmoid(self.lidar_vision_fusion_gate(gate_inputs)).to(dtype=image_embeds.dtype)
            fused_lidar = torch.tanh(fused_lidar) * fusion_gate
            image_embeds = image_embeds.clone()
            image_cursor = 0
            for batch_idx, token_count in enumerate(image_token_counts.tolist()):
                if token_count <= 0:
                    continue
                image_embeds[image_cursor : image_cursor + token_count] = (
                    image_embeds[image_cursor : image_cursor + token_count]
                    + fused_lidar[batch_idx].unsqueeze(0)
                )
                image_cursor += token_count
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            lidar_agent_memory=(
                self._cached_lidar_agent_memory.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self._cached_lidar_agent_memory is not None
                else None
            ),
            lidar_agent_mask=(
                self._cached_lidar_agent_mask.to(device=inputs_embeds.device)
                if self._cached_lidar_agent_mask is not None
                else None
            ),
            lidar_scene_memory=(
                self._cached_lidar_scene_memory.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self._cached_lidar_scene_memory is not None
                else None
            ),
            lidar_scene_mask=(
                self._cached_lidar_scene_mask.to(device=inputs_embeds.device)
                if self._cached_lidar_scene_mask is not None
                else None
            ),
            lidar_agent_route_scale=(
                self._cached_lidar_agent_route_scale.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self._cached_lidar_agent_route_scale is not None
                else None
            ),
            lidar_scene_route_scale=(
                self._cached_lidar_scene_route_scale.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self._cached_lidar_scene_route_scale is not None
                else None
            ),
            lidar_output_route_scale=(
                self._cached_lidar_output_route_scale.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if self._cached_lidar_output_route_scale is not None
                else None
            ),
            **kwargs,
        )

        if getattr(self.config, "model_type", None) == "qwen3_vl_moe":
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModelOutputWithPast

            return Qwen3VLMoeModelOutputWithPast(
                **outputs,
                rope_deltas=self.rope_deltas,
                router_logits=getattr(outputs, "router_logits", None),
            )
        else:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

            return Qwen3VLModelOutputWithPast(
                **outputs,
                rope_deltas=self.rope_deltas,
            )

    lower_model.forward = MethodType(lower_forward_with_lidar_modality, lower_model)

    original_forward = model.forward

    def forward_with_lidar_modality(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        output_hidden_states=None,
        lidar_object_features=None,
        lidar_object_feature_mask=None,
        lidar_object_geometry_features=None,
        lidar_object_geometry_feature_mask=None,
        lidar_object_numeric_features=None,
        lidar_object_numeric_feature_mask=None,
        lidar_object_numeric_label_ids=None,
        lidar_scene_features=None,
        lidar_scene_feature_mask=None,
        lidar_template_family_ids=None,
        numeric_count_targets=None,
        numeric_count_mask=None,
        numeric_motion_targets=None,
        numeric_motion_mask=None,
        numeric_coord_targets=None,
        numeric_coord_mask=None,
        output_lidar_diagnostics: bool = False,
        **kwargs,
    ):
        if (
            lidar_object_features is not None
            or lidar_object_geometry_features is not None
            or lidar_object_numeric_features is not None
            or lidar_scene_features is not None
        ):
            if input_ids is None or inputs_embeds is not None:
                raise ValueError(
                    "LiDAR modality injection requires input_ids and does not support explicit inputs_embeds."
                )

        has_numeric_aux_targets = getattr(self.config, "use_lidar_numeric_supervision", False) and any(
            tensor is not None and bool(tensor.to(dtype=torch.float32).gt(0).any().item())
            for tensor in (numeric_count_mask, numeric_motion_mask, numeric_coord_mask)
        )
        need_hidden_states = bool(output_hidden_states) or has_numeric_aux_targets

        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            output_hidden_states=need_hidden_states,
            lidar_object_features=lidar_object_features,
            lidar_object_feature_mask=lidar_object_feature_mask,
            lidar_object_geometry_features=lidar_object_geometry_features,
            lidar_object_geometry_feature_mask=lidar_object_geometry_feature_mask,
            lidar_object_numeric_features=lidar_object_numeric_features,
            lidar_object_numeric_feature_mask=lidar_object_numeric_feature_mask,
            lidar_object_numeric_label_ids=lidar_object_numeric_label_ids,
            lidar_scene_features=lidar_scene_features,
            lidar_scene_feature_mask=lidar_scene_feature_mask,
            lidar_template_family_ids=lidar_template_family_ids,
            **kwargs,
        )
        lidar_grad_anchor = getattr(self.model, "_cached_lidar_grad_anchor", None)
        if outputs.loss is not None and lidar_grad_anchor is not None:
            outputs.loss = outputs.loss + lidar_grad_anchor
        adapter_metrics, gate_usage_regularizer = self.model._collect_lidar_adapter_metrics()
        adapter_sample_metrics = (
            self.model._collect_lidar_adapter_sample_metrics() if output_lidar_diagnostics else {}
        )

        if has_numeric_aux_targets and getattr(outputs, "hidden_states", None):
            final_hidden_states = outputs.hidden_states[-1]
            pooled_states, valid_response_mask = self.model._pool_response_hidden_states(final_hidden_states, labels)
            count_predictions = self.lidar_count_head(pooled_states)
            motion_predictions = self.lidar_motion_head(pooled_states)
            coord_predictions = self.lidar_coord_head(pooled_states)
            valid_mask = valid_response_mask.to(device=pooled_states.device, dtype=pooled_states.dtype).unsqueeze(-1)
            count_loss_mask = (
                numeric_count_mask.to(device=pooled_states.device, dtype=pooled_states.dtype) * valid_mask
                if numeric_count_mask is not None
                else None
            )
            motion_loss_mask = (
                numeric_motion_mask.to(device=pooled_states.device, dtype=pooled_states.dtype) * valid_mask
                if numeric_motion_mask is not None
                else None
            )
            coord_loss_mask = (
                numeric_coord_mask.to(device=pooled_states.device, dtype=pooled_states.dtype) * valid_mask
                if numeric_coord_mask is not None
                else None
            )
            count_loss = self.model._masked_smooth_l1_loss(
                count_predictions,
                numeric_count_targets,
                count_loss_mask,
            )
            motion_loss = self.model._masked_smooth_l1_loss(
                motion_predictions,
                numeric_motion_targets,
                motion_loss_mask,
            )
            coord_loss = self.model._masked_smooth_l1_loss(
                coord_predictions,
                numeric_coord_targets,
                coord_loss_mask,
            )
            total_aux_loss = (
                self.config.lidar_numeric_count_weight * count_loss
                + self.config.lidar_numeric_motion_weight * motion_loss
                + self.config.lidar_numeric_coord_weight * coord_loss
                + self.config.lidar_gate_usage_weight * gate_usage_regularizer
            )
            if outputs.loss is None:
                outputs.loss = total_aux_loss
            else:
                outputs.loss = outputs.loss + total_aux_loss
            adapter_metrics.update(
                {
                    "numeric_count_loss": float(count_loss.detach().cpu()),
                    "numeric_motion_loss": float(motion_loss.detach().cpu()),
                    "numeric_coord_loss": float(coord_loss.detach().cpu()),
                    "adapter_gate_usage_regularizer": float(gate_usage_regularizer.detach().cpu()),
                }
            )
        elif outputs.loss is not None and self.config.lidar_gate_usage_weight > 0 and lidar_object_features is not None:
            outputs.loss = outputs.loss + self.config.lidar_gate_usage_weight * gate_usage_regularizer
            adapter_metrics["adapter_gate_usage_regularizer"] = float(gate_usage_regularizer.detach().cpu())

        if adapter_metrics:
            outputs["lidar_aux_metrics"] = adapter_metrics
        if adapter_sample_metrics:
            outputs["lidar_aux_sample_metrics"] = adapter_sample_metrics
        return outputs

    model.forward = MethodType(forward_with_lidar_modality, model)

    original_prepare_inputs_for_generation = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_with_lidar_modality(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        rope_deltas=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        lidar_object_features=None,
        lidar_object_feature_mask=None,
        lidar_object_geometry_features=None,
        lidar_object_geometry_feature_mask=None,
        lidar_object_numeric_features=None,
        lidar_object_numeric_feature_mask=None,
        lidar_object_numeric_label_ids=None,
        lidar_scene_features=None,
        lidar_scene_feature_mask=None,
        lidar_template_family_ids=None,
        **kwargs,
    ):
        model_inputs = original_prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        keep_lidar_features = is_first_iteration
        if keep_lidar_features:
            model_inputs["lidar_object_features"] = lidar_object_features
            model_inputs["lidar_object_feature_mask"] = lidar_object_feature_mask
            model_inputs["lidar_object_geometry_features"] = lidar_object_geometry_features
            model_inputs["lidar_object_geometry_feature_mask"] = lidar_object_geometry_feature_mask
            model_inputs["lidar_object_numeric_features"] = lidar_object_numeric_features
            model_inputs["lidar_object_numeric_feature_mask"] = lidar_object_numeric_feature_mask
            model_inputs["lidar_object_numeric_label_ids"] = lidar_object_numeric_label_ids
            model_inputs["lidar_scene_features"] = lidar_scene_features
            model_inputs["lidar_scene_feature_mask"] = lidar_scene_feature_mask
            model_inputs["lidar_template_family_ids"] = lidar_template_family_ids
        else:
            model_inputs["lidar_object_features"] = None
            model_inputs["lidar_object_feature_mask"] = None
            model_inputs["lidar_object_geometry_features"] = None
            model_inputs["lidar_object_geometry_feature_mask"] = None
            model_inputs["lidar_object_numeric_features"] = None
            model_inputs["lidar_object_numeric_feature_mask"] = None
            model_inputs["lidar_object_numeric_label_ids"] = None
            model_inputs["lidar_scene_features"] = None
            model_inputs["lidar_scene_feature_mask"] = None
            model_inputs["lidar_template_family_ids"] = None

        return model_inputs

    model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_with_lidar_modality, model)


def reset_qwen3_vl_lidar_template_router_on_load(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    if not (
        model_args.use_lidar_modality
        and model_args.use_lidar_template_router
        and model_args.lidar_template_router_reset_on_load
    ):
        return

    base_model = model.get_base_model() if isinstance(model, PeftModel) else model
    lower_model = getattr(base_model, "model", None)
    router = getattr(lower_model, "lidar_template_router", None)
    if router is None:
        return

    router.reset_parameters(LIDAR_TEMPLATE_ROUTE_INITIAL_SCALES)
    logger.info_rank0("Reset LiDAR template-router scales after adapter loading.")


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")

    if model_args.use_lidar_modality and model_args.lidar_injection_mode == "input_tokens":
        lidar_input_token = model_args.lidar_input_token
        lidar_token_id = tokenizer.convert_tokens_to_ids(lidar_input_token)
        if lidar_token_id == tokenizer.unk_token_id:
            num_added_special_tokens = tokenizer.add_tokens(new_tokens=[lidar_input_token], special_tokens=True)
            logger.info_rank0(f"Add LiDAR input special token {lidar_input_token} to tokenizer's vocabulary.")
            if num_added_special_tokens > 0 and not model_args.resize_vocab:
                model_args.resize_vocab = True
                logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) != "minicpmo" and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    patch_qwen3_vl_lidar_modality_model(model, model_args)
    if model_args.use_lidar_modality and hasattr(model.config, "use_lidar_modality"):
        lidar_input_token_id = tokenizer.convert_tokens_to_ids(model_args.lidar_input_token)
        setattr(model.config, "lidar_input_token", model_args.lidar_input_token)
        setattr(model.config, "lidar_input_token_id", lidar_input_token_id)
        lower_model = getattr(model, "model", None)
        if lower_model is not None:
            setattr(lower_model, "lidar_input_token", model_args.lidar_input_token)
            setattr(lower_model, "lidar_input_token_id", lidar_input_token_id)

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        if getattr(model.config, "model_type", None) == "youtu_vl":
            patch_youtu_vl_model(model)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
