#!/usr/bin/env bash
set -euo pipefail

# Install serving dependencies for local VLM inference.
# Run this inside the conda env used for VL-RouterBench.

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

MAX_JOBS="${MAX_JOBS:-4}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

python -m pip install -U pip
python -m pip install -U \
  "huggingface_hub[cli]" \
  "vllm>=0.9.1" \
  "sglang" \
  "qwen-vl-utils" \
  "accelerate" \
  "bitsandbytes" \
  "blobfile" \
  "fastapi" \
  "uvicorn"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS}" python -m pip install -U flash-attn --no-build-isolation
else
  echo "Skipping flash-attn. Set INSTALL_FLASH_ATTN=1 if Kimi/long-context serving needs it."
fi
