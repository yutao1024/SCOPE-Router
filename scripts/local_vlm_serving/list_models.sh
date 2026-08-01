#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/local_models/vlm}"

cd "${REPO_ROOT}"

python tools/local_vlm_serving.py \
  --config config/local_vlm_serving.yaml \
  --model-root "${MODEL_ROOT}" \
  list "$@"
