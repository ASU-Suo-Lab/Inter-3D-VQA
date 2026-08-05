#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.utils import rank_print
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        # breakpoint()

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # config.vocab_size 151936
        # Initialize weights and apply final processing

        self.post_init()

    def get_model(self):
        return self.model

    def _should_debug_forward(self) -> bool:
        if not getattr(self.config, "debug_forward", False):
            return False
        limit = int(getattr(self.config, "forward_debug_batch_limit", 2))
        current = int(getattr(self.config, "_forward_debug_generate_calls", 0))
        setattr(self.config, "_forward_debug_generate_calls", current + 1)
        return current < limit

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        graphs: Optional[List[dict]] = None,
        key_frames: Optional[List[List[float]]] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODOs[] : here go through GNN, go back to merge to batch 
        # TODOs[] : output temporal_edge_w too
        # breakpoint()
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, graphs, key_frames)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            # breakpoint()
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        key_frames: Optional[List[List[float]]] = None,
        graphs: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if inputs is None:
            inputs = input_ids
        elif input_ids is not None and not torch.equal(inputs, input_ids):
            raise ValueError("generate() received both inputs and input_ids with different values.")

        if inputs is None:
            raise ValueError("generate() requires inputs or input_ids.")

        debug_forward = self._should_debug_forward()
        if debug_forward:
            rank_print(
                "[debug-forward] generate start "
                f"inputs_shape={tuple(inputs.shape)} "
                f"attention_mask_shape={tuple(attention_mask.shape) if attention_mask is not None else None} "
                f"images_type={'list' if isinstance(images, list) else type(images).__name__ if images is not None else None} "
                f"modalities={modalities}"
            )

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, graphs=graphs, key_frames=key_frames)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if debug_forward:
            non_finite_inputs_embeds = (
                inputs_embeds is not None and inputs_embeds.is_floating_point() and not torch.isfinite(inputs_embeds).all().item()
            )
            rank_print(
                "[debug-forward] generate prepared "
                f"inputs_embeds_shape={tuple(inputs_embeds.shape) if inputs_embeds is not None else None} "
                f"inputs_embeds_dtype={inputs_embeds.dtype if inputs_embeds is not None else None} "
                f"inputs_embeds_non_finite={non_finite_inputs_embeds} "
                f"position_ids_shape={tuple(position_ids.shape) if position_ids is not None else None} "
                f"attention_mask_shape={tuple(attention_mask.shape) if attention_mask is not None else None}"
            )

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # breakpoint()
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
