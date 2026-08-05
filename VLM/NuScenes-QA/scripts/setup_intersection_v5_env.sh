#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/env/environment_intersection_v5.yml"

conda env create -f "${ENV_FILE}" || conda env update -f "${ENV_FILE}" --prune

echo "Environment ready. Activate with: conda activate nuscenesqa-cu128"

