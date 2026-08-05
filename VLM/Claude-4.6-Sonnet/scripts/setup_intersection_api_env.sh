#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! conda info --envs | awk '{print $1}' | grep -qx "mla"; then
  echo "Missing conda environment: mla" >&2
  echo "Use the existing mla environment for this API repo." >&2
  echo "If mla is unavailable, the fallback template is env/environment_intersection_api.yml." >&2
  exit 1
fi

conda run -n mla python -c "import importlib.util as u, sys; mods=['transformers','bert_score','sentence_transformers','pandas','sklearn','matplotlib']; missing=[m for m in mods if u.find_spec(m) is None]; sys.stderr.write('Missing modules in mla: ' + ', '.join(missing) + '\n') if missing else None; sys.exit(1 if missing else 0)"
echo "mla is ready for Claude 4.6 Sonnet intersection inference."
echo "Activate with: conda activate mla"
