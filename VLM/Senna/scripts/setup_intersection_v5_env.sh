#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/env/environment_intersection_v5.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to create senna-cu128" >&2
  exit 1
fi

conda env create -f "${ENV_FILE}" || conda env update -f "${ENV_FILE}" --prune
echo "Created/updated conda environment: senna-cu128"
