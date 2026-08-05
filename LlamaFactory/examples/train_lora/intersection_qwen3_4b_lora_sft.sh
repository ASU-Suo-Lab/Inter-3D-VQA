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

DATASET_DIR="${LLAMAFACTORY_DIR}/data/intersection_vqa_text_v5"
DATASET_PREFIX="intersection_vqa_text_v5"
OUTPUT_DIR="${LLAMAFACTORY_DIR}/saves/intersection_qwen3-4b/lora/sft_v5"
EXTRA_ARGS=("$@")

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "Missing text-only v5 dataset: ${DATASET_DIR}" >&2
  echo "Prepare it with:" >&2
  echo "  cd ${PROJECT_ROOT} && python utils/prepare_llamafactory_intersection_qwen3_text.py --source-dir LlamaFactory/data/intersection_vqa --output-dir LlamaFactory/data/intersection_vqa_text_v5 --dataset-name-prefix intersection_vqa_text_v5" >&2
  exit 1
fi

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train examples/train_lora/intersection_qwen3_4b_lora_sft.yaml \
  dataset_dir="${DATASET_DIR#${LLAMAFACTORY_DIR}/}" \
  dataset="${DATASET_PREFIX}_train" \
  eval_dataset="${DATASET_PREFIX}_val" \
  output_dir="${OUTPUT_DIR#${LLAMAFACTORY_DIR}/}" \
  "${EXTRA_ARGS[@]}"
