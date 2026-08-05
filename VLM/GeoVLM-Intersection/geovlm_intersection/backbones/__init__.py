"""External backbone adapters for GeoVLM-Intersection."""

from geovlm_intersection.backbones.lion_adapter import LionRuntime, load_lion_runtime
from geovlm_intersection.backbones.lion_token_adapter import (
    LionModelRuntime,
    LionTokenOutputs,
    build_lion_model_runtime,
    extract_lion_tokens,
    require_cuda_runtime,
)
from geovlm_intersection.backbones.qwen3_vl_adapter import (
    Qwen3VLModelRuntime,
    Qwen3VLPreparedInputs,
    Qwen3VLQuestionEmbeddings,
    Qwen3VLRuntime,
    Qwen3VLTextInputs,
    Qwen3VLVisionFeatures,
    build_qwen3_vl_model_runtime,
    embed_qwen3_vl_question_ids,
    extract_qwen3_vl_vision_features,
    load_qwen3_vl_runtime,
    prepare_qwen3_vl_inputs,
    prepare_qwen3_vl_text_inputs,
)

__all__ = [
    "LionRuntime",
    "LionModelRuntime",
    "LionTokenOutputs",
    "Qwen3VLModelRuntime",
    "Qwen3VLPreparedInputs",
    "Qwen3VLQuestionEmbeddings",
    "Qwen3VLRuntime",
    "Qwen3VLTextInputs",
    "Qwen3VLVisionFeatures",
    "build_lion_model_runtime",
    "build_qwen3_vl_model_runtime",
    "embed_qwen3_vl_question_ids",
    "extract_lion_tokens",
    "extract_qwen3_vl_vision_features",
    "load_lion_runtime",
    "load_qwen3_vl_runtime",
    "prepare_qwen3_vl_inputs",
    "prepare_qwen3_vl_text_inputs",
    "require_cuda_runtime",
]
