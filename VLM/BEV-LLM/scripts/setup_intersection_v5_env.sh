#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YAML="${REPO_ROOT}/env/environment_intersection_v5.yml"

conda env update --file "${ENV_YAML}" --prune
