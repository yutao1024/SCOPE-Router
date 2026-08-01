#!/usr/bin/env bash
set -euo pipefail

# Build a clean RouterSFT bundle for sharing/uploading.
#
# Default server layout:
#   <PROJECT_ROOT>
#   <EXTERNAL_ROOT>
#
# Output:
#   <BUNDLE_ROOT>/
#     vlm-exec-routerbench/
#     external/

SOURCE_ROOT="${SOURCE_ROOT:-<WORK_ROOT>}"
ROUTER_SRC="${ROUTER_SRC:-${SOURCE_ROOT}/vlm-exec-routerbench}"
EXTERNAL_SRC="${EXTERNAL_SRC:-${SOURCE_ROOT}/external}"
BUNDLE_DIR="${BUNDLE_DIR:-${SOURCE_ROOT}/router_sft_hf_bundle}"

need_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "[bundle] missing required command: ${name}" >&2
    exit 1
  fi
}

require_inputs() {
  if [[ ! -d "${ROUTER_SRC}" ]]; then
    echo "[bundle] missing vlm-exec-routerbench source: ${ROUTER_SRC}" >&2
    exit 1
  fi
  if [[ ! -d "${EXTERNAL_SRC}" ]]; then
    echo "[bundle] missing external source: ${EXTERNAL_SRC}" >&2
    exit 1
  fi
}

write_sanitized_env() {
  local env_file="${BUNDLE_DIR}/vlm-exec-routerbench/configs/router_sft_env.example.sh"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi

  cat > "${env_file}" <<'EOF'
#!/usr/bin/env bash
# RouterSFT runtime environment template for shared bundles.
# Fill keys in your shell or copy this file to configs/router_sft_env.local.sh.

ROUTER_ROOT="${ROUTER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "${ROUTER_ROOT}/.." && pwd)/external}"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
export BAILIAN_BASE_URL="${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

export DUBRIFY_API_KEY="${DUBRIFY_API_KEY:-}"
export DUBRIFY_BASE_URL="${DUBRIFY_BASE_URL:-https://api.dubrify.com/v1}"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

export HF_TOKEN="${HF_TOKEN:-}"
export SWEBENCH_API_KEY="${SWEBENCH_API_KEY:-}"

export ROUTER_SFT_JUDGE_MODEL="${ROUTER_SFT_JUDGE_MODEL:-gpt-5.5}"

export MINI_AGENT_EXECUTOR_COMMAND="${MINI_AGENT_EXECUTOR_COMMAND:-python3 scripts/run_mini_agent_executor.py --input {input} --output {output} --model {executor_model_ref} --temperature {temperature} --max-tokens {max_tokens} --timeout {timeout}}"
export SWEBENCH_MINI_AGENT_COMMAND="${SWEBENCH_MINI_AGENT_COMMAND:-python3 scripts/run_swebench_mini_agent_executor.py --input {input} --output {output} --model {executor_model_ref} --model-ref {executor_model_ref}}"
export MINI_AGENT_HTTP_TRANSPORT="${MINI_AGENT_HTTP_TRANSPORT:-curl}"
export MINI_AGENT_DEEPSEEK_THINKING="${MINI_AGENT_DEEPSEEK_THINKING:-disabled}"
export MINI_AGENT_TRAJECTORY_DIR="${MINI_AGENT_TRAJECTORY_DIR:-${ROUTER_ROOT}/runs/mini_agent_trajectories}"

export BIGCODEBENCH_EVAL_COMMAND="${BIGCODEBENCH_EVAL_COMMAND:-conda run -n router-code-eval python scripts/run_bigcodebench_eval.py --predictions {predictions} --result {result}}"
export LIVECODEBENCH_EVAL_COMMAND="${LIVECODEBENCH_EVAL_COMMAND:-conda run -n router-code-eval python scripts/run_livecodebench_eval.py --predictions {predictions} --result {result}}"

export SWEBENCH_KEEP_WORKSPACE="${SWEBENCH_KEEP_WORKSPACE:-0}"
export SWEBENCH_WORK_ROOT="${SWEBENCH_WORK_ROOT:-${ROUTER_ROOT}/swebench_workspaces}"
export MINI_SWE_AGENT_ROOT="${MINI_SWE_AGENT_ROOT:-${EXTERNAL_ROOT}/mini-swe-agent}"

export BROWSECOMP_PLUS_RETRIEVER="${BROWSECOMP_PLUS_RETRIEVER:-qwen3-embedding-8b}"
export BROWSECOMP_PLUS_BM25_INDEX_PATH="${BROWSECOMP_PLUS_BM25_INDEX_PATH:-${ROUTER_ROOT}/indexes/bm25}"
export BROWSECOMP_PLUS_QWEN3_EMBEDDING_8B_INDEX_PATH="${BROWSECOMP_PLUS_QWEN3_EMBEDDING_8B_INDEX_PATH:-${ROUTER_ROOT}/indexes/qwen3-embedding-8b}"
export BROWSECOMP_PLUS_EMBEDDING_MODEL="${BROWSECOMP_PLUS_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
export BROWSECOMP_PLUS_RETRIEVER_SERVER_URL="${BROWSECOMP_PLUS_RETRIEVER_SERVER_URL:-http://127.0.0.1:8765}"
export BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS="${BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS:-600000}"
export BROWSECOMP_PLUS_TOP_K="${BROWSECOMP_PLUS_TOP_K:-5}"
export BROWSECOMP_PLUS_MAX_DOC_CHARS="${BROWSECOMP_PLUS_MAX_DOC_CHARS:-2400}"
export BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS="${BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS:-20}"
export BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND="${BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND:-3}"
export BROWSECOMP_PLUS_FORCE_REFINE_SEARCH="${BROWSECOMP_PLUS_FORCE_REFINE_SEARCH:-0}"
export ROUTER_SFT_BROWSECOMP_PLUS_LLM_JUDGE="${ROUTER_SFT_BROWSECOMP_PLUS_LLM_JUDGE:-1}"

export MINI_AGENT_OCR_BACKEND="${MINI_AGENT_OCR_BACKEND:-paddle_http}"
export MINI_AGENT_MM_MAX_MODEL_CALLS="${MINI_AGENT_MM_MAX_MODEL_CALLS:-8}"
export MINI_AGENT_MM_MAX_TOOL_CALLS="${MINI_AGENT_MM_MAX_TOOL_CALLS:-15}"
export MINI_AGENT_PADDLE_OCR_TIMEOUT="${MINI_AGENT_PADDLE_OCR_TIMEOUT:-300}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/jvm" ]]; then
  export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX}/lib/jvm}"
  export JVM_PATH="${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}"
  export JDK_HOME="${JDK_HOME:-${JAVA_HOME}}"
  export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH:-}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_OPENAI_KEY}"
EOF
}

main() {
  need_cmd rsync
  require_inputs

  mkdir -p "${BUNDLE_DIR}"
  echo "[bundle] output: ${BUNDLE_DIR}"

  rsync -a --delete \
    --exclude ".git/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".pytest_cache/" \
    --exclude ".mypy_cache/" \
    --exclude "nohup.out" \
    --exclude "*.openclaw_swebench_*.json" \
    --exclude "ClawBenches/" \
    --exclude "logs/" \
    --exclude "manifests/" \
    --exclude "outputs/" \
    --exclude "reports/" \
    --exclude "runs/" \
    --exclude "sft/" \
    --exclude "sb-cli-reports/" \
    --exclude "swebench_workspaces/" \
    --exclude "openclaw_tasks_backup/" \
    --exclude "raw_hf_mixed_backup_20260626-0614/" \
    "${ROUTER_SRC}/" "${BUNDLE_DIR}/vlm-exec-routerbench/"

  rsync -a --delete "${EXTERNAL_SRC}/" "${BUNDLE_DIR}/external/"

  mkdir -p "${BUNDLE_DIR}/vlm-exec-routerbench/logs"
  write_sanitized_env

  echo "[bundle] done"
  echo "[bundle] inspect:"
  echo "  du -sh ${BUNDLE_DIR}"
  echo "  find ${BUNDLE_DIR}/vlm-exec-routerbench -maxdepth 1 -mindepth 1 -printf '%f\n' | sort"
}

main "$@"
