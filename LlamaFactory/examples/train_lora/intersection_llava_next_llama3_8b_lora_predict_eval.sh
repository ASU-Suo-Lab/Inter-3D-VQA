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

resolve_lf_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${LLAMAFACTORY_DIR}/${input_path}"
  fi
}

default_dataset_dir_for_version() {
  local dataset_version="$1"
  case "${dataset_version}" in
    v5)
      printf '%s\n' "${LLAMAFACTORY_DIR}/data/intersection_vqa"
      ;;
    v6)
      printf '%s\n' "${LLAMAFACTORY_DIR}/data/intersection_vqa_v6"
      ;;
    *)
      echo "Unsupported dataset_version: ${dataset_version}. Use v5 or v6." >&2
      exit 1
      ;;
  esac
}

default_adapter_path_for_version() {
  local dataset_version="$1"
  case "${dataset_version}" in
    v5)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_llava_next_llama3_8b/lora/sft_v5"
      ;;
    v6)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_llava_next_llama3_8b/lora/sft_v6"
      ;;
    *)
      echo "Unsupported dataset_version: ${dataset_version}. Use v5 or v6." >&2
      exit 1
      ;;
  esac
}

default_predict_output_dir_for_version() {
  local dataset_version="$1"
  case "${dataset_version}" in
    v5)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_llava_next_llama3_8b/lora/predict_v5"
      ;;
    v6)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_llava_next_llama3_8b/lora/predict_v6"
      ;;
    *)
      echo "Unsupported dataset_version: ${dataset_version}. Use v5 or v6." >&2
      exit 1
      ;;
  esac
}

infer_dataset_version_from_dir() {
  local dataset_dir="$1"
  case "${dataset_dir}" in
    *intersection_vqa_v6*)
      printf '%s\n' "v6"
      ;;
    *)
      printf '%s\n' "v5"
      ;;
  esac
}

validate_dataset_info() {
  python3 - "$1" "$2" <<'PY2'
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
expected_prefix = sys.argv[2]
info_path = dataset_dir / "dataset_info.json"
if not info_path.is_file():
    raise SystemExit(f"dataset_info.json not found in {dataset_dir}")
info = json.loads(info_path.read_text(encoding="utf-8"))
expected_keys = {f"{expected_prefix}_train", f"{expected_prefix}_val"}
missing = sorted(expected_keys - set(info.keys()))
if missing:
    raise SystemExit(
        f"dataset_info.json in {dataset_dir} is missing expected entries for {expected_prefix}: {missing}. "
        f"Regenerate the dataset with --dataset-version {expected_prefix.removeprefix('intersection_vqa_') if expected_prefix != 'intersection_vqa' else 'v5'}."
    )
PY2
}

ADAPTER_ARG="${1:-}"
PREDICT_OUTPUT_ARG="${2:-}"
SPLIT="${4:-val}"
DATASET_VERSION="v5"
DATASET_VERSION_EXPLICIT=0
DATASET_DIR_ARG=""
if [[ $# -ge 5 && "${5}" != --* ]]; then
  if [[ "${5}" == "v5" || "${5}" == "v6" ]]; then
    DATASET_VERSION="${5}"
    DATASET_VERSION_EXPLICIT=1
    if [[ $# -ge 6 && "${6}" != --* ]]; then
      DATASET_DIR_ARG="$6"
      EXTRA_EVAL_ARGS=("${@:7}")
    else
      EXTRA_EVAL_ARGS=("${@:6}")
    fi
  else
    DATASET_DIR_ARG="$5"
    EXTRA_EVAL_ARGS=("${@:6}")
  fi
else
  EXTRA_EVAL_ARGS=("${@:5}")
fi

if [[ -n "${DATASET_DIR_ARG}" ]]; then
  DATASET_DIR="$(resolve_lf_path "${DATASET_DIR_ARG}")"
  if [[ ${DATASET_VERSION_EXPLICIT} -eq 0 ]]; then
    DATASET_VERSION="$(infer_dataset_version_from_dir "${DATASET_DIR}")"
  fi
else
  DATASET_DIR="$(default_dataset_dir_for_version "${DATASET_VERSION}")"
fi

if [[ "${DATASET_VERSION}" == "v5" ]]; then
  DATASET_PREFIX="intersection_vqa"
else
  DATASET_PREFIX="intersection_vqa_v6"
fi
validate_dataset_info "${DATASET_DIR}" "${DATASET_PREFIX}"

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
  ADAPTER_PATH="$(default_adapter_path_for_version "${DATASET_VERSION}")"
fi

if [[ -n "${PREDICT_OUTPUT_ARG}" ]]; then
  PREDICT_OUTPUT_DIR="$(resolve_lf_path "${PREDICT_OUTPUT_ARG}")"
else
  PREDICT_OUTPUT_DIR="$(default_predict_output_dir_for_version "${DATASET_VERSION}")"
fi
if [[ $# -ge 3 && -n "${3}" ]]; then
  EVAL_OUTPUT_DIR="$(resolve_lf_path "${3}")"
else
  EVAL_OUTPUT_DIR="${LLAMAFACTORY_DIR}/eval/${DATASET_PREFIX}_llava_next_llama3_8b"
fi

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train \
  --model_name_or_path llava-hf/llama3-llava-next-8b-hf \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --image_max_pixels 196608 \
  --stage sft \
  --do_predict \
  --finetuning_type lora \
  --dataset_dir "${DATASET_DIR}" \
  --eval_dataset "${DATASET_NAME}" \
  --template llava_next_llama3 \
  --cutoff_len 8192 \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 4 \
  --output_dir "${PREDICT_OUTPUT_DIR}" \
  --overwrite_output_dir \
  --report_to none \
  --per_device_eval_batch_size 1 \
  --predict_with_generate \
  --bf16 \
  --ddp_timeout 180000000

GENERATED_PREDICTIONS="${PREDICT_OUTPUT_DIR}/generated_predictions.jsonl"

bash "${EVAL_SCRIPT}" "${GENERATED_PREDICTIONS}" "${EVAL_OUTPUT_DIR}" "${SPLIT}" "${DATASET_VERSION}" "${DATASET_DIR}" --model-tag llava_next_llama3_8b "${EXTRA_EVAL_ARGS[@]}"
