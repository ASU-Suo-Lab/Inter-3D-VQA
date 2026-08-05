# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Self

import torch
from omegaconf import OmegaConf
from transformers.training_args import _convert_str_dict

from ..extras.constants import AttentionFunction, EngineName, QuantizationMethod, RopeScaling
from ..extras.logging import get_logger


logger = get_logger(__name__)


@dataclass
class BaseModelArguments:
    r"""Arguments pertaining to the model."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: str | None = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    add_tokens: str | None = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: str | None = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    new_special_tokens_config: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to YAML config with special token descriptions for semantic initialization. "
                "If set, this takes precedence over add_special_tokens. "
                "YAML format: {'<token>': 'description text', ...}"
            )
        },
    )
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = field(
        default="noise_init",
        metadata={
            "help": (
                "Initialization method for new special tokens: "
                "'noise_init' (default, random noise around mean), "
                "'desc_init' (semantic initialization from descriptions), "
                "'desc_init_w_noise' (semantic + random noise). "
                "Note: 'desc_init' methods require new_special_tokens_config."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: RopeScaling | None = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: AttentionFunction = field(
        default=AttentionFunction.AUTO,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Literal["convert", "load"] | None = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing (no need to install unsloth)."},
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: float | None = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(
        default=True,
        metadata={"help": "Whether or not to use reentrant gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: EngineName = field(
        default=EngineName.HF,
        metadata={"help": "Backend engine used at inference."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_kv_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    use_v1_kernels: bool | None = field(
        default=False,
        metadata={"help": "Whether or not to use high-performance kernels in training."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        if self.add_tokens is not None:  # support multiple tokens
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",")]

        # Process special tokens with priority: new_special_tokens_config > add_special_tokens
        if self.new_special_tokens_config is not None:
            # Priority 1: Load from YAML config (extracts both tokens and descriptions)
            try:
                cfg = OmegaConf.load(self.new_special_tokens_config)
                token_descriptions = OmegaConf.to_container(cfg)

                if not isinstance(token_descriptions, dict):
                    raise ValueError(
                        f"YAML config must be a dictionary mapping tokens to descriptions. "
                        f"Got: {type(token_descriptions)}"
                    )

                # Extract token list from config keys
                extracted_tokens = list(token_descriptions.keys())

                # Warn if both are set
                if self.add_special_tokens is not None:
                    logger.warning_rank0(
                        "Both 'new_special_tokens_config' and 'add_special_tokens' are set. "
                        f"Using tokens from config: {extracted_tokens}"
                    )

                # Override add_special_tokens with extracted tokens (as list)
                self.add_special_tokens = extracted_tokens

                # Store descriptions internally for later use (internal attribute)
                self._special_token_descriptions = token_descriptions

                logger.info_rank0(
                    f"Loaded {len(extracted_tokens)} special tokens with descriptions from: "
                    f"{self.new_special_tokens_config}"
                )

            except Exception as e:
                logger.error_rank0(
                    f"Failed to load special tokens config from '{self.new_special_tokens_config}': {e}"
                )
                raise

        elif self.add_special_tokens is not None:
            # Priority 2: Use simple comma-separated string (no descriptions)
            self.add_special_tokens = [token.strip() for token in self.add_special_tokens.split(",")]
            self._special_token_descriptions = None

        else:
            # No special tokens to add
            self._special_token_descriptions = None

        # Validate init method
        if self.init_special_tokens in ["desc_init", "desc_init_w_noise"]:
            if self._special_token_descriptions is None:
                logger.warning_rank0(
                    f"init_special_tokens='{self.init_special_tokens}' requires new_special_tokens_config. "
                    "Falling back to 'noise_init'"
                )
                self.init_special_tokens = "noise_init"


@dataclass
class QuantizationArguments:
    r"""Arguments pertaining to the quantization method."""

    quantization_method: QuantizationMethod = field(
        default=QuantizationMethod.BNB,
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Literal["auto"] | None = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )


@dataclass
class ProcessorArguments:
    r"""Arguments pertaining to the image processor."""

    image_max_pixels: int = field(
        default=768 * 768,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=32 * 32,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    image_do_pan_and_scan: bool = field(
        default=False,
        metadata={"help": "Use pan and scan to process image for gemma3."},
    )
    crop_to_patches: bool = field(
        default=False,
        metadata={"help": "Whether to crop the image to patches for internvl."},
    )
    video_max_pixels: int = field(
        default=256 * 256,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    video_min_pixels: int = field(
        default=16 * 16,
        metadata={"help": "The minimum number of pixels of video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={"help": "Whether or not to use audio in video inputs."},
    )
    audio_sampling_rate: int = field(
        default=16000,
        metadata={"help": "The sampling rate of audio inputs."},
    )
    use_lidar_modality: bool = field(
        default=False,
        metadata={"help": "Whether to encode LiDAR object features as modality tokens for Qwen3-VL models."},
    )
    use_lidar_object_features: bool = field(
        default=False,
        metadata={"help": "Whether to use offline LiDAR object tokens alongside LiDAR scene tokens."},
    )
    use_lidar_object_geometry_features: bool = field(
        default=False,
        metadata={"help": "Whether to use offline object-aligned geometry features to refine LiDAR object tokens."},
    )
    use_lidar_object_numeric_features: bool = field(
        default=False,
        metadata={"help": "Whether to use explicit object-box numeric features to refine LiDAR object tokens."},
    )
    use_lidar_scene_features: bool = field(
        default=False,
        metadata={"help": "Whether to use offline LiDAR scene tokens as scene memory for object refinement."},
    )
    use_lidar_object_memory: bool = field(
        default=True,
        metadata={"help": "Whether to construct LiDAR object memory for decoder-side injection."},
    )
    use_lidar_scene_memory: bool = field(
        default=True,
        metadata={"help": "Whether to construct LiDAR scene memory for decoder-side injection."},
    )
    use_lidar_template_router: bool = field(
        default=False,
        metadata={"help": "Whether to scale LiDAR object/scene memory by question template family."},
    )
    lidar_encoder_mode: str = field(
        default="full",
        metadata={"help": "LiDAR encoder ablation mode. Supported: `full`, `projection_only`."},
    )
    lidar_object_memory_mode: str = field(
        default="connector",
        metadata={"help": "LiDAR object memory construction mode. Supported: `connector`, `direct`."},
    )
    lidar_scene_memory_mode: str = field(
        default="connector",
        metadata={"help": "LiDAR scene memory construction mode. Supported: `connector`, `direct`."},
    )
    lidar_injection_mode: str = field(
        default="decoder_adapter",
        metadata={"help": "LiDAR injection mode. Supported: `decoder_adapter`, `input_tokens`, `vision_fusion`."},
    )
    use_lidar_decoder_adapter: bool = field(
        default=True,
        metadata={"help": "Whether to inject LiDAR memory into decoder layers through decoder adapters."},
    )
    lidar_input_token: str = field(
        default="<|lidar_pad|>",
        metadata={"help": "Special token used as LiDAR placeholder for input-token injection."},
    )
    lidar_ablation_tag: str | None = field(
        default=None,
        metadata={"help": "Optional ablation tag used for experiment bookkeeping."},
    )
    lidar_hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size of the lightweight LiDAR encoder before projection to text hidden size."},
    )
    lidar_object_prefix_length: int = field(
        default=16,
        metadata={"help": "Maximum number of LiDAR object tokens retained per frame before projection into the language model."},
    )
    lidar_object_token_dim: int = field(
        default=523,
        metadata={"help": "Feature dimension of each offline LiDAR object token before projection."},
    )
    lidar_object_geometry_token_dim: int = field(
        default=128,
        metadata={"help": "Feature dimension of each offline object-aligned LiDAR geometry feature before fusion."},
    )
    lidar_object_numeric_token_dim: int = field(
        default=10,
        metadata={"help": "Feature dimension of each explicit LiDAR object numeric feature vector before label embedding."},
    )
    lidar_object_numeric_label_vocab_size: int = field(
        default=16,
        metadata={"help": "Vocabulary size used for LiDAR object numeric label embeddings."},
    )
    lidar_object_numeric_label_emb_dim: int = field(
        default=32,
        metadata={"help": "Embedding dimension used for LiDAR object numeric label ids."},
    )
    lidar_scene_token_dim: int = field(
        default=386,
        metadata={"help": "Feature dimension of each offline LiDAR scene token before scene-memory encoding."},
    )
    lidar_scene_token_length: int = field(
        default=256,
        metadata={"help": "Number of LiDAR scene tokens retained per frame for scene-memory retrieval."},
    )
    lidar_scene_memory_length: int = field(
        default=64,
        metadata={"help": "Number of resampled LiDAR scene memory tokens exposed to the text decoder."},
    )
    lidar_cross_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads used in LiDAR cross attention blocks."},
    )
    lidar_decoder_adapter_layers: int = field(
        default=4,
        metadata={"help": "Number of top decoder layers augmented with LiDAR decoder adapters."},
    )
    lidar_decoder_adapter_position: str = field(
        default="tail",
        metadata={"help": "Where to place LiDAR decoder adapters. Supported: `tail`, `middle`, `front`."},
    )
    lidar_decoder_adapter_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads used in LiDAR decoder adapters."},
    )
    lidar_decoder_adapter_gate_bias: float = field(
        default=-6.0,
        metadata={"help": "Initial bias for LiDAR decoder adapter gates."},
    )
    lidar_template_router_output_dim: int = field(
        default=2,
        metadata={"help": "Number of route scales produced by the LiDAR template router."},
    )
    lidar_template_router_reset_on_load: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to reset loaded LiDAR template-router weights to the configured initial scales after "
                "adapter loading. Useful for inference-time router ablations without retraining."
            )
        },
    )
    lidar_numeric_count_weight: float = field(
        default=0.0,
        metadata={"help": "Auxiliary count regression weight for LiDAR-grounded modality training."},
    )
    lidar_numeric_motion_weight: float = field(
        default=0.0,
        metadata={"help": "Auxiliary motion regression weight for LiDAR-grounded modality training."},
    )
    lidar_numeric_coord_weight: float = field(
        default=0.0,
        metadata={"help": "Auxiliary coordinate regression weight for LiDAR-grounded modality training."},
    )
    lidar_gate_usage_weight: float = field(
        default=1.0e-3,
        metadata={"help": "Regularization weight encouraging decoder adapters to keep using LiDAR memory."},
    )

    def __post_init__(self):
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("`image_max_pixels` cannot be smaller than `image_min_pixels`.")

        if self.video_max_pixels < self.video_min_pixels:
            raise ValueError("`video_max_pixels` cannot be smaller than `video_min_pixels`.")

        if self.lidar_hidden_size <= 0:
            raise ValueError("`lidar_hidden_size` must be positive.")

        if self.lidar_object_prefix_length <= 0:
            raise ValueError("`lidar_object_prefix_length` must be positive.")

        if self.lidar_object_token_dim <= 0:
            raise ValueError("`lidar_object_token_dim` must be positive.")

        if self.lidar_object_geometry_token_dim <= 0:
            raise ValueError("`lidar_object_geometry_token_dim` must be positive.")

        if self.lidar_object_numeric_token_dim <= 0:
            raise ValueError("`lidar_object_numeric_token_dim` must be positive.")

        if self.lidar_object_numeric_label_vocab_size <= 0:
            raise ValueError("`lidar_object_numeric_label_vocab_size` must be positive.")

        if self.lidar_object_numeric_label_emb_dim <= 0:
            raise ValueError("`lidar_object_numeric_label_emb_dim` must be positive.")

        if self.lidar_scene_token_dim <= 0:
            raise ValueError("`lidar_scene_token_dim` must be positive.")

        if self.lidar_scene_token_length <= 0:
            raise ValueError("`lidar_scene_token_length` must be positive.")

        if self.lidar_scene_memory_length <= 0:
            raise ValueError("`lidar_scene_memory_length` must be positive.")

        if self.use_lidar_object_features and not self.use_lidar_modality:
            raise ValueError("`use_lidar_object_features` requires `use_lidar_modality=true`.")

        if self.use_lidar_object_geometry_features and not self.use_lidar_object_features:
            raise ValueError("`use_lidar_object_geometry_features` requires `use_lidar_object_features=true`.")

        if self.use_lidar_object_numeric_features and not self.use_lidar_object_features:
            raise ValueError("`use_lidar_object_numeric_features` requires `use_lidar_object_features=true`.")

        if self.use_lidar_scene_features and not self.use_lidar_modality:
            raise ValueError("`use_lidar_scene_features` requires `use_lidar_modality=true`.")

        if self.use_lidar_template_router and not self.use_lidar_modality:
            raise ValueError("`use_lidar_template_router` requires `use_lidar_modality=true`.")

        if self.use_lidar_modality and not self.use_lidar_object_features:
            raise ValueError("`use_lidar_modality` requires `use_lidar_object_features=true`.")

        if self.lidar_encoder_mode not in ("full", "projection_only"):
            raise ValueError("`lidar_encoder_mode` must be one of: `full`, `projection_only`.")

        if self.lidar_object_memory_mode not in ("connector", "direct"):
            raise ValueError("`lidar_object_memory_mode` must be one of: `connector`, `direct`.")

        if self.lidar_scene_memory_mode not in ("connector", "direct"):
            raise ValueError("`lidar_scene_memory_mode` must be one of: `connector`, `direct`.")

        if self.lidar_injection_mode not in ("decoder_adapter", "input_tokens", "vision_fusion"):
            raise ValueError(
                "`lidar_injection_mode` must be one of: `decoder_adapter`, `input_tokens`, `vision_fusion`."
            )

        if self.lidar_cross_attention_heads <= 0:
            raise ValueError("`lidar_cross_attention_heads` must be positive.")

        if self.lidar_decoder_adapter_layers <= 0:
            raise ValueError("`lidar_decoder_adapter_layers` must be positive.")

        if self.lidar_decoder_adapter_position not in ("tail", "middle", "front"):
            raise ValueError("`lidar_decoder_adapter_position` must be one of: `tail`, `middle`, `front`.")

        if self.lidar_decoder_adapter_heads <= 0:
            raise ValueError("`lidar_decoder_adapter_heads` must be positive.")

        if self.lidar_template_router_output_dim not in (2, 3):
            raise ValueError("`lidar_template_router_output_dim` must be 2 or 3.")

        if self.lidar_numeric_count_weight < 0:
            raise ValueError("`lidar_numeric_count_weight` must be non-negative.")

        if self.lidar_numeric_motion_weight < 0:
            raise ValueError("`lidar_numeric_motion_weight` must be non-negative.")

        if self.lidar_numeric_coord_weight < 0:
            raise ValueError("`lidar_numeric_coord_weight` must be non-negative.")

        if self.lidar_gate_usage_weight < 0:
            raise ValueError("`lidar_gate_usage_weight` must be non-negative.")


@dataclass
class ExportArguments:
    r"""Arguments pertaining to the model export."""

    export_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=5,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    export_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: str | None = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: str | None = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )

    def __post_init__(self):
        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")


@dataclass
class VllmArguments:
    r"""Arguments pertaining to the vLLM worker."""

    vllm_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.7,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    vllm_config: dict | str | None = field(
        default=None,
        metadata={"help": "Config to initialize the vllm engine. Please use JSON strings."},
    )

    def __post_init__(self):
        if isinstance(self.vllm_config, str) and self.vllm_config.startswith("{"):
            self.vllm_config = _convert_str_dict(json.loads(self.vllm_config))


@dataclass
class SGLangArguments:
    r"""Arguments pertaining to the SGLang worker."""

    sglang_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the SGLang engine."},
    )
    sglang_mem_fraction: float = field(
        default=0.7,
        metadata={"help": "The memory fraction (0-1) to be used for the SGLang engine."},
    )
    sglang_tp_size: int = field(
        default=-1,
        metadata={"help": "Tensor parallel size for the SGLang engine."},
    )
    sglang_config: dict | str | None = field(
        default=None,
        metadata={"help": "Config to initialize the SGLang engine. Please use JSON strings."},
    )
    sglang_lora_backend: Literal["triton", "flashinfer"] = field(
        default="triton",
        metadata={
            "help": "The backend of running GEMM kernels for Lora modules. Recommend using the Triton LoRA backend for better performance and stability."
        },
    )

    def __post_init__(self):
        if isinstance(self.sglang_config, str) and self.sglang_config.startswith("{"):
            self.sglang_config = _convert_str_dict(json.loads(self.sglang_config))


@dataclass
class KTransformersArguments:
    r"""Arguments pertaining to the KT training."""

    use_kt: bool = field(
        default=False,
        metadata={"help": "Whether To Use KTransformers Optimizations For LoRA Training."},
    )
    kt_optimize_rule: str | None = field(
        default=None,
        metadata={
            "help": "Path To The KTransformers Optimize Rule; See https://github.com/kvcache-ai/ktransformers/."
        },
    )
    cpu_infer: int | None = field(
        default=32,
        metadata={"help": "Number Of CPU Cores Used For Computation."},
    )
    chunk_size: int | None = field(
        default=8192,
        metadata={"help": "Chunk Size Used For CPU Compute In KTransformers."},
    )
    mode: str | None = field(
        default="normal",
        metadata={"help": "Normal Or Long_Context For Llama Models."},
    )

    kt_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum Sequence (Prompt + Response) Length Of The KT Engine."},
    )
    kt_use_cuda_graph: bool = field(
        default=True,
        metadata={"help": "Whether To Use CUDA Graphs For The KT Engine."},
    )
    kt_mode: str = field(
        default="normal",
        metadata={"help": "Normal Or Long_Context Mode For The KT Engine."},
    )
    kt_force_think: bool = field(
        default=False,
        metadata={"help": "Force-Think Toggle For The KT Engine."},
    )


@dataclass
class ModelArguments(
    SGLangArguments,
    VllmArguments,
    KTransformersArguments,
    ExportArguments,
    ProcessorArguments,
    QuantizationArguments,
    BaseModelArguments,
):
    r"""Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.

    The class on the most right will be displayed first.
    """

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)
        ProcessorArguments.__post_init__(self)
        ExportArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        SGLangArguments.__post_init__(self)

    compute_dtype: torch.dtype | None = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    device_map: str | dict[str, Any] | None = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: int | None = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)
        ProcessorArguments.__post_init__(self)
        ExportArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        SGLangArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()}
        return args
