#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/local_vlm_serving/serve_model.sh MODEL [serve args...]

Examples:
  bash scripts/local_vlm_serving/serve_model.sh Qwen2.5-VL-32B-Instruct
  bash scripts/local_vlm_serving/serve_model.sh InternVL2_5-78B --port 8001
  bash scripts/local_vlm_serving/serve_model.sh Pixtral-12B --gpus 2 --port 8002
  bash scripts/local_vlm_serving/serve_model.sh Janus-Pro-7B --backend vllm --dry-run

Environment:
  MODEL_ROOT   Local snapshot root. Default: ./local_models/vlm
  HF_HOME      Hugging Face cache root. Default: ./.cache/huggingface
EOF
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/local_models/vlm}"

cd "${REPO_ROOT}"

python tools/local_vlm_serving.py \
  --config config/local_vlm_serving.yaml \
  --model-root "${MODEL_ROOT}" \
  serve "$@"
