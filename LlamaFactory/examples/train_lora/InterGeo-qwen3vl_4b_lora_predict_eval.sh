#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_VERSION="${1:-v5}"
if [[ "${DATASET_VERSION}" != "v5" && "${DATASET_VERSION}" != "v6" ]]; then
  echo "Usage: $0 [v5|v6] [eval_output_dir] [extra predict args...]" >&2
  exit 1
fi
shift || true

if [[ "${DATASET_VERSION}" == "v6" ]]; then
  DATASET_PREFIX="intersection_vqa_v6_lidar_rawscene_objroute_v2"
else
  DATASET_PREFIX="intersection_vqa_lidar_rawscene_objroute_v2"
fi

EVAL_OUTPUT_DIR="${1:-${LLAMAFACTORY_DIR}/eval/InterGeo-qwen3vl_4b/${DATASET_VERSION}}"
if [[ $# -gt 0 ]]; then
  shift
fi

PREDICT_DIR="${LLAMAFACTORY_DIR}/saves/InterGeo-qwen3vl_4b/${DATASET_VERSION}/predict"

cd "${LLAMAFACTORY_DIR}"
bash "${SCRIPT_DIR}/InterGeo-qwen3vl_4b_lora_predict_val.sh" "${DATASET_VERSION}" "$@"

bash "${SCRIPT_DIR}/intersection_lora_eval.sh" \
  "${PREDICT_DIR}/generated_predictions.jsonl" \
  "${EVAL_OUTPUT_DIR}" \
  val \
  "${DATASET_VERSION}" \
  "${LLAMAFACTORY_DIR}/data/${DATASET_PREFIX}" \
  --model-tag "InterGeo-qwen3vl_4b_${DATASET_VERSION}"
