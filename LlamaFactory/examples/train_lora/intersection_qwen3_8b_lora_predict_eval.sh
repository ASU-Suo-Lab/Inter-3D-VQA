#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${LLAMAFACTORY_DIR}/.." && pwd)"

CACHE_ROOT="${PROJECT_ROOT}/cache"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

EVAL_SCRIPT="${SCRIPT_DIR}/intersection_lora_eval.sh"
DATASET_DIR="${LLAMAFACTORY_DIR}/data/intersection_vqa_text_v5"
DATASET_PREFIX="intersection_vqa_text_v5"

resolve_lf_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${LLAMAFACTORY_DIR}/${input_path}"
  fi
}

ADAPTER_ARG="${1:-}"
PREDICT_OUTPUT_ARG="${2:-}"
EVAL_OUTPUT_ARG="${3:-}"
SPLIT="${4:-val}"
EXTRA_PREDICT_ARGS=("${@:5}")
NORMALIZED_PREDICT_ARGS=()

for arg in "${EXTRA_PREDICT_ARGS[@]}"; do
  if [[ "${arg}" == *=* && "${arg}" != --* ]]; then
    NORMALIZED_PREDICT_ARGS+=("--${arg%%=*}" "${arg#*=}")
  else
    NORMALIZED_PREDICT_ARGS+=("${arg}")
  fi
done

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "Missing text-only v5 dataset: ${DATASET_DIR}" >&2
  echo "Prepare it with:" >&2
  echo "  cd ${PROJECT_ROOT} && python utils/prepare_llamafactory_intersection_qwen3_text.py --source-dir LlamaFactory/data/intersection_vqa --output-dir LlamaFactory/data/intersection_vqa_text_v5 --dataset-name-prefix intersection_vqa_text_v5" >&2
  exit 1
fi

case "${SPLIT}" in
  train)
    DATASET_NAME="${DATASET_PREFIX}_train"
    ;;
  val)
    DATASET_NAME="${DATASET_PREFIX}_val"
    ;;
  *)
    echo "Unsupported split: ${SPLIT}. Use train or val." >&2
    exit 1
    ;;
esac

if [[ -n "${ADAPTER_ARG}" ]]; then
  ADAPTER_PATH="$(resolve_lf_path "${ADAPTER_ARG}")"
else
  ADAPTER_PATH="${LLAMAFACTORY_DIR}/saves/intersection_qwen3-8b/lora/sft_v5"
fi

if [[ -n "${PREDICT_OUTPUT_ARG}" ]]; then
  PREDICT_OUTPUT_DIR="$(resolve_lf_path "${PREDICT_OUTPUT_ARG}")"
else
  PREDICT_OUTPUT_DIR="${LLAMAFACTORY_DIR}/saves/intersection_qwen3-8b/lora/predict_v5"
fi

if [[ -n "${EVAL_OUTPUT_ARG}" ]]; then
  EVAL_OUTPUT_DIR="$(resolve_lf_path "${EVAL_OUTPUT_ARG}")"
else
  EVAL_OUTPUT_DIR="${LLAMAFACTORY_DIR}/eval/intersection_vqa_text_v5_qwen3_8b"
fi

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen3-8B \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --trust_remote_code \
  --stage sft \
  --do_predict \
  --finetuning_type lora \
  --dataset_dir "${DATASET_DIR}" \
  --eval_dataset "${DATASET_NAME}" \
  --template qwen3_nothink \
  --cutoff_len 2048 \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 4 \
  --output_dir "${PREDICT_OUTPUT_DIR}" \
  --overwrite_output_dir \
  --report_to none \
  --per_device_eval_batch_size 1 \
  --predict_with_generate \
  --bf16 \
  --ddp_timeout 180000000 \
  "${NORMALIZED_PREDICT_ARGS[@]}"

GENERATED_PREDICTIONS="${PREDICT_OUTPUT_DIR}/generated_predictions.jsonl"

bash "${EVAL_SCRIPT}" "${GENERATED_PREDICTIONS}" "${EVAL_OUTPUT_DIR}" "${SPLIT}" "${DATASET_DIR}" --model-tag qwen3_8b
