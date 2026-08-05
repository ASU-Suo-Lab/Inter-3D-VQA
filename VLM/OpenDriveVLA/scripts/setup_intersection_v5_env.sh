#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-opendrivevla-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "[setup_intersection_v5_env] repo_root=${REPO_ROOT}"
echo "[setup_intersection_v5_env] env_name=${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip git ninja cmake pkg-config make
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip "setuptools<82" wheel
conda run -n "${ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
conda run -n "${ENV_NAME}" python -m pip install \
  accelerate \
  av \
  datasets \
  decord \
  einops \
  ftfy \
  matplotlib \
  mmengine==0.9.0 \
  motmetrics==1.4.0 \
  numpy==1.26.1 \
  open_clip_torch \
  opencv-python \
  peft==0.4.0 \
  requests \
  scikit-learn==1.2.2 \
  scipy \
  sentencepiece~=0.1.99 \
  timm \
  tokenizers~=0.15.2 \
  "urllib3<=2.0.0"
conda run -n "${ENV_NAME}" python -m pip install transformers==4.40.1
conda run -n "${ENV_NAME}" python -m pip install --no-deps sentence-transformers==2.7.0 bert-score==0.3.13
conda run -n "${ENV_NAME}" python -m pip install "mmdet==2.26.0" "mmsegmentation==0.29.1"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"
conda run -n "${ENV_NAME}" python -m pip install "setuptools<82"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-openDriveVLA}"
conda run -n "${ENV_NAME}" bash -lc "cd '${REPO_ROOT}/third_party/mmcv_1_7_2' && MMCV_WITH_OPS=1 python -m pip install -e . --no-build-isolation"
conda run -n "${ENV_NAME}" bash -lc "cd '${REPO_ROOT}/third_party/mmdetection3d_1_0_0rc6' && python -m pip install -e . --no-build-isolation"
conda run -n "${ENV_NAME}" python -m opendrivevla_v5.cli.check_env --model-path "${REPO_ROOT}/weights"

echo "[setup_intersection_v5_env] completed"
