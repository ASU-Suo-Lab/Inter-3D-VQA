#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-drivelm-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
LLAMA_DIR="${DRIVELM_LLAMA_DIR:-${REPO_ROOT}/weights/llama}"
ADAPTER_CHECKPOINT="${REPO_ROOT}/challenge/llama_adapter_v2_multimodal7b/weights/checkpoint_BIAS-7B.pth"

echo "[setup_intersection_v5_env] repo_root=${REPO_ROOT}"
echo "[setup_intersection_v5_env] env_name=${ENV_NAME}"
echo "[setup_intersection_v5_env] llama_dir=${LLAMA_DIR}"

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
  fairscale \
  matplotlib \
  numpy==1.26.1 \
  opencv-python \
  pandas \
  Pillow \
  PyYAML \
  scikit-learn==1.2.2 \
  scipy \
  sentencepiece~=0.1.99 \
  tensorboard \
  timm==0.4.12 \
  tqdm
conda run -n "${ENV_NAME}" python -m pip install "clip @ git+https://github.com/openai/CLIP.git"
conda run -n "${ENV_NAME}" python -m pip install --no-deps sentence-transformers==2.7.0 bert-score==0.3.13 transformers==4.40.1

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-driveLM}"
conda run -n "${ENV_NAME}" python -m drivelm_v5.cli.check_env --llama-dir "${LLAMA_DIR}" --adapter-checkpoint "${ADAPTER_CHECKPOINT}"

echo "[setup_intersection_v5_env] completed"
