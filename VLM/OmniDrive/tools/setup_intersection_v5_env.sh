#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-omnidrive-cu128}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLM_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MMCV_SRC="${VLM_ROOT}/OpenDriveVLA/third_party/mmcv_1_7_2"
MMDET3D_SRC="${REPO_ROOT}/mmdetection3d"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment already exists: ${ENV_NAME}"
else
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/env/environment_intersection_v5.yml"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
conda run -n "${ENV_NAME}" python -m pip install accelerate==1.13.0 timm
conda run -n "${ENV_NAME}" python -m pip install mmdet==2.28.2 mmsegmentation==0.30.0 numba==0.59.1
conda run -n "${ENV_NAME}" python -m pip uninstall -y mmcv mmcv-full || true
conda run -n "${ENV_NAME}" bash -lc "cd \"${MMCV_SRC}\" && TORCH_CUDA_ARCH_LIST=\"${CUDA_ARCH_LIST}\" MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation -e ."
conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation --no-deps -e "${MMDET3D_SRC}"
conda run -n "${ENV_NAME}" bash -lc "cd \"${REPO_ROOT}\" && python -m omnidrive_v5.cli.check_env --require-cuda"
