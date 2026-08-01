#!/usr/bin/env bash
set -euo pipefail

# Start/stop the local RouterSFT auxiliary services:
#   - BrowseComp-Plus retrievers on ports 8765-8768
#   - PaddleOCR servers on ports 8775-8778
#
# Defaults match the multi-shard RouterSFT generation setup. Override with env:
#   BROWSECOMP_ENV=browsecomp-plus
#   OCR_ENV=ocr
#   RETRIEVER_PORTS="8765 8766 8767 8768"
#   RETRIEVER_GPUS="0 1 2 3"
#   OCR_PORTS="8775 8776 8777 8778"
#   OCR_GPUS="4 5 6 7"
#   RETRIEVER_MIN_FREE_MB=20000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

BROWSECOMP_ENV="${BROWSECOMP_ENV:-browsecomp-plus}"
OCR_ENV="${OCR_ENV:-ocr}"
BROWSECOMP_RETRIEVER="${BROWSECOMP_RETRIEVER:-qwen3-embedding-8b}"

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
OCR_HOST="${OCR_HOST:-127.0.0.1}"
RETRIEVER_PORTS="${RETRIEVER_PORTS:-8765 8766 8767 8768}"
RETRIEVER_GPUS="${RETRIEVER_GPUS:-0 1 2 3}"
OCR_PORTS="${OCR_PORTS:-8775 8776 8777 8778}"
OCR_GPUS="${OCR_GPUS:-4 5 6 7}"

RETRIEVER_HEALTH_TIMEOUT="${RETRIEVER_HEALTH_TIMEOUT:-5}"
RETRIEVER_SEARCH_TIMEOUT="${RETRIEVER_SEARCH_TIMEOUT:-180}"
OCR_HEALTH_TIMEOUT="${OCR_HEALTH_TIMEOUT:-5}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-60}"
WAIT_SLEEP="${WAIT_SLEEP:-2}"
RETRIEVER_MIN_FREE_MB="${RETRIEVER_MIN_FREE_MB:-20000}"
OCR_MIN_FREE_MB="${OCR_MIN_FREE_MB:-2000}"

SEARCH_PROBE_QUERY="${SEARCH_PROBE_QUERY:-ReFrame Stamp movie festival 2018-2023}"

usage() {
  cat <<EOF
Usage: bash scripts/start_router_sft_services.sh [start|stop|restart|status|check]

Defaults:
  ROOT=${ROOT}
  LOG_DIR=${LOG_DIR}
  BROWSECOMP_ENV=${BROWSECOMP_ENV}
  OCR_ENV=${OCR_ENV}
  RETRIEVER_PORTS="${RETRIEVER_PORTS}"
  RETRIEVER_GPUS="${RETRIEVER_GPUS}"
  OCR_PORTS="${OCR_PORTS}"
  OCR_GPUS="${OCR_GPUS}"
  RETRIEVER_MIN_FREE_MB=${RETRIEVER_MIN_FREE_MB}
  OCR_MIN_FREE_MB=${OCR_MIN_FREE_MB}

Examples:
  bash scripts/start_router_sft_services.sh start
  bash scripts/start_router_sft_services.sh restart
  RETRIEVER_GPUS="2 3" RETRIEVER_PORTS="8765 8766" bash scripts/start_router_sft_services.sh start
  RETRIEVER_MIN_FREE_MB=0 bash scripts/start_router_sft_services.sh start
EOF
}

need_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "[services] missing required command: ${name}" >&2
    exit 1
  fi
}

require_tools() {
  need_cmd conda
  need_cmd curl
  need_cmd lsof
  need_cmd python3
}

env_prefix() {
  local env_name="$1"
  conda run -n "${env_name}" python -c 'import sys; print(sys.prefix)'
}

gpu_free_mb() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v target="${gpu}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1)
          gsub(/^[ \t]+|[ \t]+$/, "", $2)
          if ($1 == target) {
            print $2
            found=1
          }
        }
        END { if (!found) exit 1 }
      '
}

check_gpu_free_mb() {
  local label="$1"
  local min_free_mb="$2"
  shift 2

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[services] nvidia-smi not found; skipping ${label} GPU memory preflight"
    return 0
  fi

  local failed=0
  local gpu free_mb
  for gpu in "$@"; do
    if [[ "${gpu}" == "cpu" || "${gpu}" == "none" || "${gpu}" == "-" ]]; then
      continue
    fi
    if ! free_mb="$(gpu_free_mb "${gpu}")"; then
      echo "[services] could not read free memory for GPU ${gpu}" >&2
      failed=1
      continue
    fi
    echo "[services] ${label} GPU ${gpu}: ${free_mb} MiB free"
    if [[ "${free_mb}" -lt "${min_free_mb}" ]]; then
      echo "[services] ${label} GPU ${gpu} has ${free_mb} MiB free, below ${min_free_mb} MiB" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    cat >&2 <<EOF
[services] GPU memory preflight failed.
[services] Choose less busy GPUs, for example:
  RETRIEVER_GPUS="4 5" RETRIEVER_PORTS="8765 8766" bash scripts/start_router_sft_services.sh start
[services] Or disable this check if you know what you are doing:
  RETRIEVER_MIN_FREE_MB=0 bash scripts/start_router_sft_services.sh start
EOF
    exit 1
  fi
}

read_words() {
  local -n out_ref="$1"
  local value="$2"
  # shellcheck disable=SC2206
  out_ref=(${value})
}

assert_same_length() {
  local label_a="$1"
  local len_a="$2"
  local label_b="$3"
  local len_b="$4"
  if [[ "${len_a}" -ne "${len_b}" ]]; then
    echo "[services] ${label_a} count (${len_a}) must match ${label_b} count (${len_b})" >&2
    exit 1
  fi
}

pid_for_port() {
  local port="$1"
  lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
}

stop_ports() {
  local ports=("$@")
  local port pid

  for port in "${ports[@]}"; do
    pid="$(pid_for_port "${port}")"
    if [[ -n "${pid}" ]]; then
      echo "[services] TERM port ${port} pid ${pid}"
      kill ${pid} 2>/dev/null || true
    fi
  done

  sleep 2

  for port in "${ports[@]}"; do
    pid="$(pid_for_port "${port}")"
    if [[ -n "${pid}" ]]; then
      echo "[services] KILL port ${port} pid ${pid}"
      kill -9 ${pid} 2>/dev/null || true
    fi
  done

  sleep 1
}

print_status() {
  local ports=("$@")
  local port pid
  for port in "${ports[@]}"; do
    pid="$(pid_for_port "${port}")"
    if [[ -n "${pid}" ]]; then
      echo "[services] port ${port}: LISTEN pid ${pid}"
    else
      echo "[services] port ${port}: free"
    fi
  done
}

start_retrievers() {
  local browsecomp_prefix
  browsecomp_prefix="$(env_prefix "${BROWSECOMP_ENV}")"
  local default_java_home="${browsecomp_prefix}/lib/jvm"
  local java_home="${JAVA_HOME:-${default_java_home}}"
  local jvm_path="${JVM_PATH:-${java_home}/lib/server/libjvm.so}"

  if [[ ! -f "${jvm_path}" && -f "${default_java_home}/lib/server/libjvm.so" ]]; then
    echo "[services] ignoring invalid JAVA_HOME/JVM_PATH from environment; using ${default_java_home}"
    java_home="${default_java_home}"
    jvm_path="${default_java_home}/lib/server/libjvm.so"
  fi

  local browsecomp_python="${browsecomp_prefix}/bin/python"
  local retriever_ld_library_path="${java_home}/lib/server:${java_home}/lib:${browsecomp_prefix}/lib:${LD_LIBRARY_PATH:-}"
  local openai_api_key="${OPENAI_API_KEY:-dummy}"
  local openai_admin_key="${OPENAI_ADMIN_KEY:-dummy}"

  if [[ ! -f "${jvm_path}" ]]; then
    echo "[services] libjvm.so not found: ${jvm_path}" >&2
    echo "[services] install OpenJDK in ${BROWSECOMP_ENV} or set JVM_PATH explicitly." >&2
    exit 1
  fi
  if [[ ! -x "${browsecomp_python}" ]]; then
    echo "[services] python not found in ${BROWSECOMP_ENV}: ${browsecomp_python}" >&2
    exit 1
  fi

  local ports gpus
  read_words ports "${RETRIEVER_PORTS}"
  read_words gpus "${RETRIEVER_GPUS}"
  assert_same_length RETRIEVER_PORTS "${#ports[@]}" RETRIEVER_GPUS "${#gpus[@]}"
  check_gpu_free_mb retriever "${RETRIEVER_MIN_FREE_MB}" "${gpus[@]}"

  echo "[services] checking BrowseComp BM25/JVM fallback"
  if ! (
    cd "${ROOT}"
    JAVA_HOME="${java_home}" \
    JVM_PATH="${jvm_path}" \
    LD_LIBRARY_PATH="${retriever_ld_library_path}" \
    OPENAI_API_KEY="${openai_api_key}" \
    OPENAI_ADMIN_KEY="${openai_admin_key}" \
    "${browsecomp_python}" -c 'import sys
sys.path.insert(0, "scripts")
import browsecomp_plus_retriever as r
docs = r.official_bm25_search("ReFrame Stamp movie festival 2018-2023", 1, 120)
text = (docs[0].get("text") or "") if docs else ""
print("bm25_probe", docs[0].get("docid") if docs else None, len(text), repr(text[:80]))
raise SystemExit(0 if text.strip() else 1)'
  ); then
    echo "[services] BrowseComp BM25/JVM preflight failed; check JAVA_HOME/JVM_PATH and indexes/bm25" >&2
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  local index port gpu pid_file log_file
  for index in "${!ports[@]}"; do
    port="${ports[${index}]}"
    gpu="${gpus[${index}]}"
    pid_file="${LOG_DIR}/browsecomp_plus_retriever_${port}.pid"
    log_file="${LOG_DIR}/browsecomp_plus_retriever_${port}.log"

    echo "[services] start retriever port ${port} gpu ${gpu}"
    (
      cd "${ROOT}"
      JAVA_HOME="${java_home}" \
      JVM_PATH="${jvm_path}" \
      LD_LIBRARY_PATH="${retriever_ld_library_path}" \
      OPENAI_API_KEY="${openai_api_key}" \
      OPENAI_ADMIN_KEY="${openai_admin_key}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      nohup "${browsecomp_python}" -u scripts/browsecomp_plus_retriever.py \
          --serve \
          --host "${RETRIEVER_HOST}" \
          --port "${port}" \
          --retriever "${BROWSECOMP_RETRIEVER}" \
        > "${log_file}" 2>&1 &
      echo $! > "${pid_file}"
    )
  done
}

start_ocr() {
  local ocr_prefix
  ocr_prefix="$(env_prefix "${OCR_ENV}")"
  local ocr_python="${ocr_prefix}/bin/python"
  if [[ ! -x "${ocr_python}" ]]; then
    echo "[services] python not found in ${OCR_ENV}: ${ocr_python}" >&2
    exit 1
  fi

  local ports gpus
  read_words ports "${OCR_PORTS}"
  read_words gpus "${OCR_GPUS}"
  assert_same_length OCR_PORTS "${#ports[@]}" OCR_GPUS "${#gpus[@]}"
  check_gpu_free_mb OCR "${OCR_MIN_FREE_MB}" "${gpus[@]}"

  mkdir -p "${LOG_DIR}"
  local index port gpu pid_file log_file
  for index in "${!ports[@]}"; do
    port="${ports[${index}]}"
    gpu="${gpus[${index}]}"
    pid_file="${LOG_DIR}/ocr_server_${port}.pid"
    log_file="${LOG_DIR}/ocr_server_${port}.log"

    echo "[services] start OCR port ${port} gpu ${gpu}"
    (
      cd "${ROOT}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      nohup "${ocr_python}" -u scripts/ocr_server.py \
          --host "${OCR_HOST}" \
          --port "${port}" \
          --lang ch \
        > "${log_file}" 2>&1 &
      echo $! > "${pid_file}"
    )
  done
}

wait_retriever_health() {
  local port="$1"
  local log_file="${LOG_DIR}/browsecomp_plus_retriever_${port}.log"
  local attempt
  for attempt in $(seq 1 "${WAIT_ATTEMPTS}"); do
    if curl --max-time "${RETRIEVER_HEALTH_TIMEOUT}" -fs \
      "http://${RETRIEVER_HOST}:${port}/health" >/tmp/router_sft_retriever_health_"${port}".json 2>/dev/null; then
      echo "[services] retriever ${port} health OK"
      cat /tmp/router_sft_retriever_health_"${port}".json
      echo
      return 0
    fi
    sleep "${WAIT_SLEEP}"
  done

  echo "[services] retriever ${port} health FAILED" >&2
  tail -120 "${log_file}" >&2 || true
  return 1
}

check_retriever_search() {
  local port="$1"
  local log_file="${LOG_DIR}/browsecomp_plus_retriever_${port}.log"
  local body_file="/tmp/router_sft_retriever_search_${port}.json"
  local body
  body="$(
    SEARCH_PROBE_QUERY="${SEARCH_PROBE_QUERY}" \
    BROWSECOMP_RETRIEVER="${BROWSECOMP_RETRIEVER}" \
    python3 -c 'import json, os; print(json.dumps({"query": os.environ["SEARCH_PROBE_QUERY"], "top_k": 1, "max_doc_chars": 120, "retriever": os.environ["BROWSECOMP_RETRIEVER"]}))'
  )"

  local attempt
  for attempt in $(seq 1 5); do
    if curl --max-time "${RETRIEVER_SEARCH_TIMEOUT}" -fs \
      "http://${RETRIEVER_HOST}:${port}/search" \
      -H "Content-Type: application/json" \
      -d "${body}" > "${body_file}" 2>/dev/null; then
      if python3 -c 'import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
r=d["results"][0]
text=r.get("text") or ""
print(sys.argv[2], r.get("docid"), len(text), repr(text[:80]))
raise SystemExit(0 if text.strip() else 1)' "${body_file}" "${port}"; then
        return 0
      fi
    fi
    sleep 3
  done

  echo "[services] retriever ${port} search FAILED" >&2
  cat "${body_file}" >&2 2>/dev/null || true
  tail -160 "${log_file}" >&2 || true
  return 1
}

wait_ocr_health() {
  local port="$1"
  local log_file="${LOG_DIR}/ocr_server_${port}.log"
  local attempt
  for attempt in $(seq 1 "${WAIT_ATTEMPTS}"); do
    if curl --max-time "${OCR_HEALTH_TIMEOUT}" -fs \
      "http://${OCR_HOST}:${port}/health" >/tmp/router_sft_ocr_health_"${port}".json 2>/dev/null; then
      echo "[services] OCR ${port} health OK"
      cat /tmp/router_sft_ocr_health_"${port}".json
      echo
      return 0
    fi
    sleep "${WAIT_SLEEP}"
  done

  echo "[services] OCR ${port} health FAILED" >&2
  tail -160 "${log_file}" >&2 || true
  return 1
}

check_all() {
  local retriever_ports ocr_ports port
  read_words retriever_ports "${RETRIEVER_PORTS}"
  read_words ocr_ports "${OCR_PORTS}"

  local failed=0
  for port in "${retriever_ports[@]}"; do
    wait_retriever_health "${port}" || failed=1
  done
  for port in "${retriever_ports[@]}"; do
    check_retriever_search "${port}" || failed=1
  done
  for port in "${ocr_ports[@]}"; do
    wait_ocr_health "${port}" || failed=1
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "[services] one or more service checks failed" >&2
    exit 1
  fi
}

start_all() {
  local retriever_ports ocr_ports
  read_words retriever_ports "${RETRIEVER_PORTS}"
  read_words ocr_ports "${OCR_PORTS}"

  stop_ports "${retriever_ports[@]}" "${ocr_ports[@]}"
  start_retrievers
  start_ocr
  check_all

  cat <<EOF

[services] all services are ready.

Runtime exports for generation:
  export MINI_AGENT_OCR_BACKEND=paddle_http
  export MINI_AGENT_PADDLE_OCR_TIMEOUT=300
  export BROWSECOMP_PLUS_RETRIEVER=${BROWSECOMP_RETRIEVER}
  export BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS=600000

Shard mapping:
  BrowseComp retrievers: ${RETRIEVER_PORTS}
  OCR servers: ${OCR_PORTS}
EOF
}

stop_all() {
  local retriever_ports ocr_ports
  read_words retriever_ports "${RETRIEVER_PORTS}"
  read_words ocr_ports "${OCR_PORTS}"
  stop_ports "${retriever_ports[@]}" "${ocr_ports[@]}"
  print_status "${retriever_ports[@]}" "${ocr_ports[@]}"
}

status_all() {
  local retriever_ports ocr_ports
  read_words retriever_ports "${RETRIEVER_PORTS}"
  read_words ocr_ports "${OCR_PORTS}"
  print_status "${retriever_ports[@]}" "${ocr_ports[@]}"
}

main() {
  require_tools
  local command="${1:-start}"
  case "${command}" in
    start) start_all ;;
    restart) stop_all; start_all ;;
    stop) stop_all ;;
    status) status_all ;;
    check) check_all ;;
    -h|--help|help) usage ;;
    *)
      echo "[services] unknown command: ${command}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
