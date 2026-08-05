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

find_sidecar() {
  local dataset_dir="$1"
  shopt -s nullglob
  local matches=("${dataset_dir}"/*_eval_sidecar.jsonl)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected exactly one *_eval_sidecar.jsonl in ${dataset_dir}, found ${#matches[@]}." >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
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

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <predictions.jsonl|predictions.json> [output_dir] [split] [dataset_version|dataset_dir] [dataset_dir] [extra eval args...] --model-tag <model_tag>"
  echo "Example V5: $0 /path/to/generated_predictions.jsonl "" val v5 "" --model-tag qwen3vl_8b --skip-semantic-metrics"
  echo "Example V6: $0 /path/to/generated_predictions.jsonl "" val v6 "" --model-tag llava_next_llama3_8b --skip-semantic-metrics"
  exit 1
fi

PREDICTIONS_ARG="$1"
OUTPUT_DIR_ARG="${2:-}"
SPLIT="${3:-val}"
DATASET_VERSION="v5"
DATASET_VERSION_EXPLICIT=0
DATASET_DIR_ARG=""

if [[ $# -ge 4 && "${4}" != --* ]]; then
  if [[ "${4}" == "v5" || "${4}" == "v6" ]]; then
    DATASET_VERSION="${4}"
    DATASET_VERSION_EXPLICIT=1
    if [[ $# -ge 5 && "${5}" != --* ]]; then
      DATASET_DIR_ARG="$5"
      EXTRA_ARGS=("${@:6}")
    else
      EXTRA_ARGS=("${@:5}")
    fi
  else
    DATASET_DIR_ARG="$4"
    EXTRA_ARGS=("${@:5}")
  fi
else
  EXTRA_ARGS=("${@:4}")
fi

MODEL_TAG=""
PASSTHROUGH_ARGS=()
index=0
while [[ ${index} -lt ${#EXTRA_ARGS[@]} ]]; do
  arg="${EXTRA_ARGS[${index}]}"
  if [[ "${arg}" == "--model-tag" ]]; then
    if [[ $((index + 1)) -ge ${#EXTRA_ARGS[@]} ]]; then
      echo "Missing value for --model-tag" >&2
      exit 1
    fi
    MODEL_TAG="${EXTRA_ARGS[$((index + 1))]}"
    index=$((index + 2))
    continue
  fi
  PASSTHROUGH_ARGS+=("${arg}")
  index=$((index + 1))
done
EXTRA_ARGS=("${PASSTHROUGH_ARGS[@]}")

if [[ -n "${DATASET_DIR_ARG}" ]]; then
  DATASET_DIR="$(resolve_lf_path "${DATASET_DIR_ARG}")"
  if [[ ${DATASET_VERSION_EXPLICIT} -eq 0 ]]; then
    DATASET_VERSION="$(infer_dataset_version_from_dir "${DATASET_DIR}")"
  fi
else
  DATASET_DIR="$(default_dataset_dir_for_version "${DATASET_VERSION}")"
fi

PREDICTIONS_PATH="$(resolve_lf_path "${PREDICTIONS_ARG}")"

if [[ -n "${OUTPUT_DIR_ARG}" ]]; then
  OUTPUT_DIR="$(resolve_lf_path "${OUTPUT_DIR_ARG}")"
else
  if [[ -z "${MODEL_TAG}" ]]; then
    echo "--model-tag is required when output_dir is omitted." >&2
    exit 1
  fi
  OUTPUT_DIR="${LLAMAFACTORY_DIR}/eval/$(basename "${DATASET_DIR}")_${MODEL_TAG}"
fi

SIDECAR_PATH="$(find_sidecar "${DATASET_DIR}")"

cd "${PROJECT_ROOT}"
python3 utils/evaluate_intersection_vqa_auto.py \
  --predictions "${PREDICTIONS_PATH}" \
  --sidecar-jsonl "${SIDECAR_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}"
