#!/usr/bin/env bash
# RouterSFT runtime environment template.
#
# Usage:
#   cp configs/router_sft_env.example.sh configs/router_sft_env.local.sh
#   chmod 600 configs/router_sft_env.local.sh
#   vim configs/router_sft_env.local.sh
#   source configs/router_sft_env.local.sh
#
# Keep real keys only in configs/router_sft_env.local.sh or your shell profile.
# Do not put real keys in this example file.

ROUTER_ROOT="${ROUTER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Default external repo root is next to vlm-exec-routerbench, for example:
#   <EXTERNAL_ROOT>
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "${ROUTER_ROOT}/.." && pwd)/external}"

# DeepSeek judge / text model provider.
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"

# Bailian / DashScope provider. Kept only as a fallback; default candidates
# currently route through OpenRouter, including Qwen.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
export BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Dubrify provider. Pricing/docs entry point:
#   https://dubrify.com/pricing
export DUBRIFY_API_KEY="${DUBRIFY_API_KEY:-}"
export DUBRIFY_BASE_URL="https://api.dubrify.com/v1"

# OpenRouter provider for all default multimodal candidates.
#   https://openrouter.ai/
# Bare GPT model names like gpt-5.4-mini, gpt-5.4, and gpt-5.5 are
# sent as openai/<model> by default. Bare Qwen names like
# qwen3-vl-8b-instruct are sent as qwen/<model> by default.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# Optional: override OpenRouter model ids if their catalog names differ.
# export OPENROUTER_GPT_5_4_MINI_MODEL="openai/gpt-5.4-mini"
# export OPENROUTER_GPT_5_4_MODEL="openai/gpt-5.4"
# export OPENROUTER_GPT_5_5_MODEL="openai/gpt-5.5"

# Default LLM judge. Use a multimodal judge so visual QA fallback grading can
# inspect the attached image(s).
export ROUTER_SFT_JUDGE_MODEL="${ROUTER_SFT_JUDGE_MODEL:-gpt-5.5}"

# Default private executor adapter. This is the default path on the current
# server; OpenClaw is intentionally not configured there for mini-agent debug.
export MINI_AGENT_EXECUTOR_COMMAND='python3 scripts/run_mini_agent_executor.py --input {input} --output {output} --model {executor_model_ref} --temperature {temperature} --max-tokens {max_tokens} --timeout {timeout}'
export SWEBENCH_MINI_AGENT_COMMAND='python3 scripts/run_swebench_mini_agent_executor.py --input {input} --output {output} --model {executor_model_ref} --model-ref {executor_model_ref}'
export MINI_AGENT_HTTP_TRANSPORT="${MINI_AGENT_HTTP_TRANSPORT:-curl}"
export MINI_AGENT_DEEPSEEK_THINKING="${MINI_AGENT_DEEPSEEK_THINKING:-disabled}"
export MINI_AGENT_TRAJECTORY_DIR="${MINI_AGENT_TRAJECTORY_DIR:-${ROUTER_ROOT}/runs/mini_agent_trajectories}"

# Optional OpenClaw adapter defaults. Keep these commented on the current
# server unless explicitly reproducing the public OpenClaw-backed pipeline.
# export OPENCLAW_BIN="openclaw"
# export OPENCLAW_AGENT="main"
# export OPENCLAW_TIMEOUT_UNIT="seconds"
# export OPENCLAW_TEXT_MODE="cli"
# Tool workflow mode:
#   infer            stable default; avoids agent executing benchmark tools, but may lose token usage
#   agent            original agent path; useful for token/accounting debugging
#   agent-then-infer capture an agent attempt, then fall back to infer if no tool_calls appear
# export OPENCLAW_TOOL_MODE="${OPENCLAW_TOOL_MODE:-infer}"
# export OPENCLAW_TEXT_THINKING="${OPENCLAW_TEXT_THINKING:-off}"
# export OPENCLAW_VISION_MODE="auto"
# export OPENCLAW_VISION_RETRIES="2"
# export OPENCLAW_VISION_RETRY_SLEEP="2.0"
# export OPENCLAW_GATEWAY_MEDIA_TIMEOUT="90"
# Optional: source workspace for bootstrap files copied into the small
# Gateway runtime workspace before each OpenClaw executor call. On the server:
# export OPENCLAW_BOOTSTRAP_WORKSPACE="<OPENCLAW_WORKSPACE>"
# export OPENCLAW_BOOTSTRAP_WORKSPACE="${OPENCLAW_BOOTSTRAP_WORKSPACE:-}"
# Text/tool/code execution defaults to the stable `openclaw agent --message`
# path. Set OPENCLAW_TEXT_MODE=gateway only when explicitly debugging the
# Gateway agent path and its operator scopes are approved.
# Multimodal OpenClaw execution now defaults to the Gateway attachment pipeline.
# Large attachment params are written to a temp file and sent through
# scripts/openclaw_gateway_call.mjs to avoid argv size limits.
# Keep OPENCLAW_VISION_COMMAND unset unless you explicitly want the legacy
# describe-then-answer fallback for local smoke tests.
# export OPENCLAW_VISION_COMMAND='conda run -n openclaw python scripts/run_openclaw_vision_executor.py --input {input} --output {output} --model-ref {openclaw_model_ref} --timeout {timeout} --retries 2 --retry-sleep 2.0'

# Optional: override the OpenClaw config path if needed.
# export OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
# Optional: override OpenClaw package root if the executor cannot infer it from OPENCLAW_BIN.
# export OPENCLAW_PACKAGE_DIR="/path/to/lib/node_modules/openclaw"

# Optional: model ref formatting used by generate_router_sft.py command templates.
# Usually keep provider/model as-is:
#   openrouter/qwen/qwen3-vl-8b-instruct
#   openrouter/openai/gpt-5.4
# export OPENCLAW_MODEL_REF_TEMPLATE="{provider}/{model}"
#export HTTP_PROXY=http://127.0.0.1:17897
#export HTTPS_PROXY=http://127.0.0.1:17897
#export NO_PROXY=localhost,127.0.0.1,::1,dashscope.aliyuncs.com,api.deepseek.com,openrouter.ai

# export OPENCLAW_EXECUTOR_COMMAND='python3 scripts/run_openclaw_executor.py --input {input} --output {output} --model-ref-template {executor_model_ref}'
# unset OPENCLAW_GATEWAY_URL OPENCLAW_GATEWAY_TOKEN OPENCLAW_GATEWAY_PASSWORD

export HF_TOKEN="${HF_TOKEN:-}"

export BIGCODEBENCH_EVAL_COMMAND='conda run -n router-code-eval python scripts/run_bigcodebench_eval.py --predictions {predictions} --result {result}'
export LIVECODEBENCH_EVAL_COMMAND='conda run -n router-code-eval python scripts/run_livecodebench_eval.py --predictions {predictions} --result {result}'

export SWEBENCH_KEEP_WORKSPACE=0
export SWEBENCH_WORK_ROOT="${SWEBENCH_WORK_ROOT:-${ROUTER_ROOT}/swebench_workspaces}"

# BrowseComp-Plus mini-agent retrieval. This does not require OpenClaw plugins.
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

export PYTHONNOUSERSITE=1
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/jvm" ]]; then
  export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX}/lib/jvm}"
  export JVM_PATH="${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}"
  export JDK_HOME="${JDK_HOME:-${JAVA_HOME}}"
  export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH:-}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_OPENAI_KEY}"
export CUDA_VISIBLE_DEVICES=""

export SWEBENCH_API_KEY="${SWEBENCH_API_KEY:-}"

export MINI_SWE_AGENT_ROOT="${MINI_SWE_AGENT_ROOT:-${EXTERNAL_ROOT}/mini-swe-agent}"

export MINI_AGENT_OCR_BACKEND="${MINI_AGENT_OCR_BACKEND:-paddle_http}"
export MINI_AGENT_MM_MAX_MODEL_CALLS="${MINI_AGENT_MM_MAX_MODEL_CALLS:-8}"
export MINI_AGENT_MM_MAX_TOOL_CALLS="${MINI_AGENT_MM_MAX_TOOL_CALLS:-15}"
export MINI_AGENT_PADDLE_OCR_TIMEOUT="${MINI_AGENT_PADDLE_OCR_TIMEOUT:-300}"
