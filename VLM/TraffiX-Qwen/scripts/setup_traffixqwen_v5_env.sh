#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/env/environment_intersection_v5.yml}"
ENV_NAME="${ENV_NAME:-traffixqwen-cu128}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Environment file not found: ${ENV_FILE}" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment already exists: ${ENV_NAME}" >&2
    echo "Remove it manually or choose a different ENV_NAME before rerunning." >&2
    exit 1
fi

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
PY
conda run -n "${ENV_NAME}" python -m traffixqwen_v5.cli.pipeline --help >/dev/null
