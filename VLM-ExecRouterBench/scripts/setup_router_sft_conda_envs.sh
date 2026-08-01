#!/usr/bin/env bash
set -euo pipefail

# Install the auxiliary conda environments used by RouterSFT generation.
#
# Paths you may need to change on a new machine:
#   EXTERNAL_ROOT: parent directory containing LiveCodeBench, SWE-bench, mini-swe-agent
#   LIVECODEBENCH_DIR: official LiveCodeBench checkout
#   SWEBENCH_DIR: official SWE-bench checkout
#   MINI_SWE_AGENT_DIR: mini-swe-agent checkout
#   PADDLE_CUDA_INDEX: PaddlePaddle GPU wheel index matching the machine CUDA runtime
#
# Examples:
#   bash scripts/setup_router_sft_conda_envs.sh all
#   bash scripts/setup_router_sft_conda_envs.sh code ocr
#   EXTERNAL_ROOT=<EXTERNAL_ROOT> bash scripts/setup_router_sft_conda_envs.sh swebench swe-agent

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "${ROOT}/.." && pwd)/external}"
LIVECODEBENCH_DIR="${LIVECODEBENCH_DIR:-${EXTERNAL_ROOT}/LiveCodeBench}"
SWEBENCH_DIR="${SWEBENCH_DIR:-${EXTERNAL_ROOT}/SWE-bench}"
MINI_SWE_AGENT_DIR="${MINI_SWE_AGENT_DIR:-${EXTERNAL_ROOT}/mini-swe-agent}"
PADDLE_CUDA_INDEX="${PADDLE_CUDA_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

CODE_ENV="${CODE_ENV:-router-code-eval}"
OCR_ENV="${OCR_ENV:-ocr}"
SWEBENCH_ENV="${SWEBENCH_ENV:-swebench}"
BROWSECOMP_ENV="${BROWSECOMP_ENV:-browsecomp-plus}"
SWE_AGENT_ENV="${SWE_AGENT_ENV:-swe-agent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

usage() {
  cat <<EOF
Usage: bash scripts/setup_router_sft_conda_envs.sh [all|code|ocr|swebench|browsecomp|swe-agent]...

Default paths:
  EXTERNAL_ROOT=${EXTERNAL_ROOT}
  LIVECODEBENCH_DIR=${LIVECODEBENCH_DIR}
  SWEBENCH_DIR=${SWEBENCH_DIR}
  MINI_SWE_AGENT_DIR=${MINI_SWE_AGENT_DIR}
  PADDLE_CUDA_INDEX=${PADDLE_CUDA_INDEX}

Examples:
  bash scripts/setup_router_sft_conda_envs.sh all
  bash scripts/setup_router_sft_conda_envs.sh code ocr
  EXTERNAL_ROOT=<EXTERNAL_ROOT> bash scripts/setup_router_sft_conda_envs.sh swebench swe-agent
EOF
}

need_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup-envs] conda is required on PATH" >&2
    exit 1
  fi
}

ensure_env() {
  local env_name="$1"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[setup-envs] reusing conda env: ${env_name}"
  else
    echo "[setup-envs] creating conda env: ${env_name}"
    conda create -y -n "${env_name}" "python=${PYTHON_VERSION}" pip
  fi
  conda run -n "${env_name}" python -m pip install -U pip setuptools wheel
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[setup-envs] missing ${label}: ${path}" >&2
    echo "[setup-envs] set ${label}_DIR or EXTERNAL_ROOT before running this target." >&2
    exit 1
  fi
}

install_code_env() {
  ensure_env "${CODE_ENV}"
  conda run -n "${CODE_ENV}" python -m pip install \
    "numpy==1.24.4" \
    "pandas==1.5.3" \
    "matplotlib==3.8.2" \
    pytest \
    requests \
    tqdm
  conda run -n "${CODE_ENV}" python -m pip install bigcodebench
  require_dir "${LIVECODEBENCH_DIR}" "LIVECODEBENCH"
  conda run -n "${CODE_ENV}" python -m pip install -e "${LIVECODEBENCH_DIR}"
}

install_ocr_env() {
  ensure_env "${OCR_ENV}"
  conda run -n "${OCR_ENV}" python -m pip install "paddlepaddle-gpu==3.3.1" \
    -i "${PADDLE_CUDA_INDEX}" \
    --trusted-host www.paddlepaddle.org.cn
  conda run -n "${OCR_ENV}" python -m pip install \
    "paddleocr==3.7.0" \
    "paddlex==3.7.2" \
    "opencv-contrib-python==4.10.0.84" \
    "pillow==12.3.0" \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "requests==2.34.2" \
    "pydantic==2.13.4" \
    "modelscope==1.38.1" \
    "huggingface_hub==1.22.0"
}

install_swebench_env() {
  ensure_env "${SWEBENCH_ENV}"
  conda run -n "${SWEBENCH_ENV}" python -m pip install \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "datasets==5.0.0" \
    "pyarrow==24.0.0" \
    "docker==7.1.0" \
    "GitPython==3.1.50" \
    "unidiff==0.7.5" \
    "modal==1.5.1" \
    "requests==2.34.2" \
    "tqdm==4.68.4" \
    "beautifulsoup4==4.15.0" \
    "python-dotenv==1.2.2" \
    "rich==15.0.0" \
    "ghapi==2.0.0" \
    "pre-commit==4.6.0"
  require_dir "${SWEBENCH_DIR}" "SWEBENCH"
  conda run -n "${SWEBENCH_ENV}" python -m pip install -e "${SWEBENCH_DIR}"
}

install_browsecomp_env() {
  ensure_env "${BROWSECOMP_ENV}"
  conda run -n "${BROWSECOMP_ENV}" python -m pip install \
    "torch==2.13.0" \
    "transformers==5.13.0" \
    "sentence-transformers==5.6.0" \
    "tokenizers==0.22.2" \
    "safetensors==0.8.0" \
    "sentencepiece==0.2.1"
  conda run -n "${BROWSECOMP_ENV}" python -m pip install \
    "pyserini==1.2.0" \
    "onnxruntime==1.23.2" \
    "openai==2.44.0" \
    "tiktoken==0.13.0" \
    "fastapi==0.139.0" \
    "uvicorn==0.51.0" \
    "Flask==3.1.3" \
    "mcp==1.28.1" \
    "pydantic==2.13.4" \
    "pydantic-settings==2.14.2" \
    "pandas==2.3.3" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "scikit-learn==1.7.2" \
    "requests==2.34.2" \
    "tqdm==4.68.4" \
    "huggingface_hub==1.22.0"
  conda install -n "${BROWSECOMP_ENV}" -c conda-forge openjdk=21 -y
}

install_swe_agent_env() {
  ensure_env "${SWE_AGENT_ENV}"
  conda run -n "${SWE_AGENT_ENV}" python -m pip install \
    "datasets==5.0.0" \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "pyarrow==24.0.0" \
    "requests==2.34.2" \
    "openai==2.44.0" \
    "litellm==1.91.1" \
    "tiktoken==0.13.0" \
    "pillow==12.3.0" \
    "python-dotenv==1.2.2" \
    "rich==15.0.0" \
    "textual==8.2.8" \
    "typer==0.26.8" \
    "prompt_toolkit==3.0.52" \
    "pydantic==2.13.4" \
    "jsonschema==4.26.0" \
    "tqdm==4.68.4" \
    "PyYAML==6.0.3"
  require_dir "${MINI_SWE_AGENT_DIR}" "MINI_SWE_AGENT"
  conda run -n "${SWE_AGENT_ENV}" python -m pip install -e "${MINI_SWE_AGENT_DIR}"
}

verify_envs() {
  cat <<EOF

[setup-envs] install commands completed.

Suggested runtime exports:
  export BIGCODEBENCH_EVAL_COMMAND='conda run -n ${CODE_ENV} python scripts/run_bigcodebench_eval.py --predictions {predictions} --result {result}'
  export LIVECODEBENCH_EVAL_COMMAND='conda run -n ${CODE_ENV} python scripts/run_livecodebench_eval.py --predictions {predictions} --result {result}'
  export MINI_AGENT_OCR_BACKEND=paddle_http
  export MINI_AGENT_PADDLE_OCR_URL=http://127.0.0.1:8766/ocr
  export SWEBENCH_CONDA_ENV=${SWEBENCH_ENV}
  export MINI_AGENT_CONDA_ENV=${SWE_AGENT_ENV}
  export BROWSECOMP_PLUS_RETRIEVER_SERVER_URL=http://127.0.0.1:8765

Start services:
  conda run -n ${OCR_ENV} python scripts/ocr_server.py --host 127.0.0.1 --port 8766 --lang ch --device gpu:0
  conda run -n ${BROWSECOMP_ENV} python scripts/browsecomp_plus_retriever.py --serve --warmup --host 127.0.0.1 --port 8765 --retriever qwen3-embedding-8b
EOF
}

main() {
  need_conda
  if [[ "$#" -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local targets=("$@")
  if [[ "$#" -eq 1 && "$1" == "all" ]]; then
    targets=(code ocr swebench browsecomp swe-agent)
  fi

  for target in "${targets[@]}"; do
    case "${target}" in
      code|router-code-eval) install_code_env ;;
      ocr) install_ocr_env ;;
      swebench) install_swebench_env ;;
      browsecomp|browsecomp-plus|search) install_browsecomp_env ;;
      swe-agent|mini-swe-agent) install_swe_agent_env ;;
      all)
        install_code_env
        install_ocr_env
        install_swebench_env
        install_browsecomp_env
        install_swe_agent_env
        ;;
      *)
        echo "[setup-envs] unknown target: ${target}" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
  verify_envs
}

main "$@"
