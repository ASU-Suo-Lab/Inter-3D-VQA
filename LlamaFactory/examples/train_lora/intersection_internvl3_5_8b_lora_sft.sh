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

default_output_dir_for_version() {
  local dataset_version="$1"
  case "${dataset_version}" in
    v5)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_internvl3_5-8b/lora/sft_v5"
      ;;
    v6)
      printf '%s\n' "${LLAMAFACTORY_DIR}/saves/intersection_internvl3_5-8b/lora/sft_v6"
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
    version = "v5" if expected_prefix == "intersection_vqa" else expected_prefix.removeprefix("intersection_vqa_")
    raise SystemExit(
        f"dataset_info.json in {dataset_dir} is missing expected entries for {expected_prefix}: {missing}. "
        f"Regenerate the dataset with --dataset-version {version}."
    )
PY2
}

if [[ $# -ge 1 ]]; then
  DATASET_VERSION_OR_DIR="$1"
else
  DATASET_VERSION_OR_DIR="v5"
fi
DATASET_DIR_ARG="${2:-}"
if [[ "${DATASET_VERSION_OR_DIR}" == "v5" || "${DATASET_VERSION_OR_DIR}" == "v6" ]]; then
  DATASET_VERSION="${DATASET_VERSION_OR_DIR}"
  EXTRA_ARGS=("${@:3}")
else
  DATASET_DIR_ARG="${DATASET_VERSION_OR_DIR}"
  DATASET_VERSION="$(infer_dataset_version_from_dir "$(resolve_lf_path "${DATASET_DIR_ARG}")")"
  EXTRA_ARGS=("${@:2}")
fi

if [[ -n "${DATASET_DIR_ARG}" ]]; then
  DATASET_DIR="$(resolve_lf_path "${DATASET_DIR_ARG}")"
else
  DATASET_DIR="$(default_dataset_dir_for_version "${DATASET_VERSION}")"
fi

if [[ "${DATASET_VERSION}" == "v5" ]]; then
  DATASET_PREFIX="intersection_vqa"
else
  DATASET_PREFIX="intersection_vqa_v6"
fi
validate_dataset_info "${DATASET_DIR}" "${DATASET_PREFIX}"
OUTPUT_DIR="$(default_output_dir_for_version "${DATASET_VERSION}")"

RELATIVE_DATASET_DIR="${DATASET_DIR}"
if [[ "${RELATIVE_DATASET_DIR}" == "${LLAMAFACTORY_DIR}"/* ]]; then
  RELATIVE_DATASET_DIR="${RELATIVE_DATASET_DIR#${LLAMAFACTORY_DIR}/}"
fi

RELATIVE_OUTPUT_DIR="${OUTPUT_DIR}"
if [[ "${RELATIVE_OUTPUT_DIR}" == "${LLAMAFACTORY_DIR}"/* ]]; then
  RELATIVE_OUTPUT_DIR="${RELATIVE_OUTPUT_DIR#${LLAMAFACTORY_DIR}/}"
fi

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train examples/train_lora/intersection_internvl3_5_8b_lora_sft.yaml \
  dataset_dir="${RELATIVE_DATASET_DIR}" \
  dataset="${DATASET_PREFIX}_train" \
  eval_dataset="${DATASET_PREFIX}_val" \
  output_dir="${RELATIVE_OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
