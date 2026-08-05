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

DATASET_VERSION="${1:-v5}"
if [[ "${DATASET_VERSION}" != "v5" && "${DATASET_VERSION}" != "v6" ]]; then
  echo "Unsupported dataset version: ${DATASET_VERSION}. Use v5 or v6." >&2
  exit 1
fi
shift || true

if [[ "${DATASET_VERSION}" == "v6" ]]; then
  DATASET_PREFIX="intersection_vqa_v6_lidar_rawscene_objroute_v2"
else
  DATASET_PREFIX="intersection_vqa_lidar_rawscene_objroute_v2"
fi

DATASET_DIR="${LLAMAFACTORY_DIR}/data/${DATASET_PREFIX}"
OUTPUT_DIR="${LLAMAFACTORY_DIR}/saves/InterGeo-qwen3vl_4b/${DATASET_VERSION}/train"

python3 - "${DATASET_DIR}" "${DATASET_PREFIX}" <<'PY'
import json
from pathlib import Path
import sys

dataset_dir = Path(sys.argv[1])
dataset_prefix = sys.argv[2]
info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
required = {f"{dataset_prefix}_train", f"{dataset_prefix}_val"}
missing = sorted(required - set(info))
if missing:
    raise SystemExit(f"Missing InterGeo-qwen3vl_4b dataset entries: {missing}")

summary = json.loads((dataset_dir / "split_summary.json").read_text(encoding="utf-8"))
if summary.get("lidar_object_evidence_max_objects") != 8:
    raise SystemExit("InterGeo-qwen3vl_4b requires lidar_object_evidence_max_objects=8.")
if summary.get("lidar_object_evidence_selection") != "template_aware":
    raise SystemExit("InterGeo-qwen3vl_4b requires template_aware LiDAR object evidence selection.")
PY

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train examples/train_lora/InterGeo-qwen3vl_4b_lora_sft.yaml \
  dataset_dir="${DATASET_DIR#${LLAMAFACTORY_DIR}/}" \
  dataset="${DATASET_PREFIX}_train" \
  eval_dataset="${DATASET_PREFIX}_val" \
  output_dir="${OUTPUT_DIR#${LLAMAFACTORY_DIR}/}" \
  lidar_ablation_tag="InterGeo-qwen3vl_4b_${DATASET_VERSION}" \
  "$@"
