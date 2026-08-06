#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${LLAMAFACTORY_DIR}/.." && pwd)"

DATASET_VERSION="${1:-v5}"
LION_QUALITY="${2:-high}"
FEATURE_OUTPUT_DIR="${3:-${PROJECT_ROOT}/cache/interGeo_lion_tokens/${DATASET_VERSION}}"

if [[ "${DATASET_VERSION}" != "v5" && "${DATASET_VERSION}" != "v6" ]]; then
  echo "Unsupported dataset version: ${DATASET_VERSION}. Use v5 or v6." >&2
  exit 1
fi

if [[ "${DATASET_VERSION}" == "v6" ]]; then
  DATASET_PREFIX="interGeo_vqa_v6_lidar"
else
  DATASET_PREFIX="interGeo_vqa_v5_lidar"
fi
DATASET_OUTPUT_DIR="${LLAMAFACTORY_DIR}/data/${DATASET_PREFIX}"

python3 "${PROJECT_ROOT}/utils/prepare_intersection_lidar_frames.py" \
  --dataset-version "${DATASET_VERSION}"

python3 "${PROJECT_ROOT}/utils/extract_llamafactory_lidar_bev_tokens.py" \
  --dataset-version "${DATASET_VERSION}" \
  --lion-quality "${LION_QUALITY}" \
  --output-dir "${FEATURE_OUTPUT_DIR}" \
  --scene-token-budget 256 \
  --object-candidate-limit 32 \
  --object-score-threshold 0.4 \
  --object-geometry-points 32 \
  --overwrite

python3 "${PROJECT_ROOT}/utils/prepare_llamafactory_intersection_vqa.py" \
  --qa-json "${PROJECT_ROOT}/intersection_qa_pairs_${DATASET_VERSION}.json" \
  --dataset-version "${DATASET_VERSION}" \
  --prompt-mode pointcloud_plus_image_detailed_rawscene \
  --dataset-name-prefix "${DATASET_PREFIX}" \
  --output-dir "${DATASET_OUTPUT_DIR}" \
  --lidar-object-dir "${FEATURE_OUTPUT_DIR}" \
  --lidar-object-geometry-dir "${FEATURE_OUTPUT_DIR}" \
  --lidar-scene-dir "${FEATURE_OUTPUT_DIR}" \
  --lidar-object-evidence-max-objects 8 \
  --lidar-object-evidence-selection template_aware
