#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-geovlm-cu128}"
BASE_ENV="${BASE_ENV:-openpcdet}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_REQS="${REPO_ROOT}/env/requirements_geovlm_overlay.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if [[ ! -f "${OVERLAY_REQS}" ]]; then
  echo "Missing overlay requirements: ${OVERLAY_REQS}" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${BASE_ENV}"; then
  echo "Base environment '${BASE_ENV}' does not exist." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' already exists. Reusing it."
else
  echo "Cloning '${BASE_ENV}' -> '${ENV_NAME}'"
  conda create -y -n "${ENV_NAME}" --clone "${BASE_ENV}"
fi

echo "Installing GeoVLM overlay requirements into '${ENV_NAME}'"
conda run -n "${ENV_NAME}" pip install --upgrade -r "${OVERLAY_REQS}"

echo "Validating key packages in '${ENV_NAME}'"
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
mods = ["torch", "transformers", "tokenizers", "accelerate", "peft", "spconv", "SharedArray"]
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, exc)
        sys.exit(1)
PY

echo "GeoVLM environment is ready."
echo "Activate with: conda activate ${ENV_NAME}"
