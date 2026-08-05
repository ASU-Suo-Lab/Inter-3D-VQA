from typing import Tuple
import os

import torch
from transformers import AutoTokenizer, LlamaConfig

from .model.bevllm_llama import BevLLMLlamaForCausalLM
from .utils import count_parameters, get_model_size_in_gb

def build_model(config:dict) -> Tuple[BevLLMLlamaForCausalLM, AutoTokenizer]:

    access_token = config.get("access_token") or os.environ.get("HF_TOKEN")
    model_id = config["model_id"]

    model_config = LlamaConfig.from_pretrained(model_id, token=access_token, cache_dir=config["cache_dir"])
    model_config.cache_dir = config["cache_dir"]
    model_config.num_query_token = config.get("num_query_token", 32)
    model_config.bev_channels = config.get("bev_channels", 512)
    model_config.cross_attention_freq = config.get("cross_attention_freq", 2)
    model_config.pos_encoding_scale = config.get("pos_encoding_scale", 0.06)
    model_config.qformer_model_id = config.get("qformer_model_id", "bert-base-uncased")
    model_config.tokenizer_model_max_length = config.get("tokenizer_model_max_length", 2048)
    model_config.tokenizer_padding_side = config.get("tokenizer_padding_side", "right")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token, cache_dir=config["cache_dir"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = model_config.eos_token_id if model_config.eos_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(f"{model_id} did not resolve a valid pad_token_id for generation.")
    if eos_token_id is None:
        raise ValueError(f"{model_id} did not resolve a valid eos_token_id for generation.")

    model_config.pad_token_id = pad_token_id
    model_config.eos_token_id = eos_token_id

    model = BevLLMLlamaForCausalLM(model_config, freeze_qformer=False)
    model.config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    if config.get("use_lora", True):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required when model.use_lora=true.") from exc
        peft_config = LoraConfig(
            r=config["lora_config"]["r"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            bias=config["lora_config"]["bias"],
            target_modules=config["lora_config"]["target_modules"]
        )
        model.model = get_peft_model(model.model, peft_config)
    print(f"[INFO] Model size: {get_model_size_in_gb(model):.2f} GB")
    print(f"[INFO] Trainable params: {count_parameters(model):,}")

    return model, tokenizer


def load_checkpoint(model, checkpoint_path: str, map_location: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["MODEL_STATE"] if "MODEL_STATE" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "epochs_run": checkpoint.get("EPOCHS_RUN"),
    }

