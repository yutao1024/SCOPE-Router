#!/usr/bin/env bash
set -euo pipefail

# Run RouterSFT generation in split shards:
#   - code_no_swebench: 8 shards by default
#   - vqa/mm: 8 shards by default, OCR servers round-robin on 8775-8778
#   - browsecomp_plus: 4 shards by default, retriever servers on 8765-8768
#
# Start services first:
#   bash scripts/start_router_sft_services.sh start
#
# Run detached:
#   nohup bash scripts/run_router_sft_full_shards.sh > logs/router_sft_full_shards.log 2>&1 &

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

init_conda_shell() {
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
  fi
}

source_runtime_env() {
  local env_file="${ROUTER_SFT_ENV_FILE:-}"
  if [[ -z "${env_file}" ]]; then
    if [[ -f configs/router_sft_env.local.sh ]]; then
      env_file="configs/router_sft_env.local.sh"
    else
      env_file="configs/router_sft_env.example.sh"
    fi
  fi
  if [[ "${env_file}" == "none" ]]; then
    echo "[run-shards] skipping runtime env source"
    return 0
  fi
  if [[ ! -f "${env_file}" ]]; then
    echo "[run-shards] runtime env file not found: ${env_file}" >&2
    exit 1
  fi

  echo "[run-shards] source runtime env: ${env_file}"
  set +u
  # shellcheck disable=SC1090
  source "${env_file}"
  set -u
}

init_conda_shell
source_runtime_env

N_CODE="${N_CODE:-8}"
N_VQA="${N_VQA:-8}"
N_BROWSE="${N_BROWSE:-4}"

CODE_WORKERS="${CODE_WORKERS:-1}"
VQA_WORKERS="${VQA_WORKERS:-1}"
BROWSE_WORKERS="${BROWSE_WORKERS:-1}"

CODE_TASK_FILE="${CODE_TASK_FILE:-openclaw_tasks/code_debug_edit/tasks_code_no_swebench.jsonl}"
VQA_TASK_FILE="${VQA_TASK_FILE:-openclaw_tasks/multimodal_doc_visual/tasks_all_vqa.jsonl}"
BROWSE_TASK_FILE="${BROWSE_TASK_FILE:-openclaw_tasks/tool_workflow/tasks_browsecomp_plus.jsonl}"

OCR_BASE_PORT="${OCR_BASE_PORT:-8775}"
OCR_PORT_COUNT="${OCR_PORT_COUNT:-4}"
BROWSE_RETRIEVER_BASE_PORT="${BROWSE_RETRIEVER_BASE_PORT:-8765}"

OUT="${OUT:-outputs/tencent_batch_full_split_shards_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT}" logs

export MINI_AGENT_OCR_BACKEND="${MINI_AGENT_OCR_BACKEND:-paddle_http}"
export MINI_AGENT_MM_MAX_MODEL_CALLS="${MINI_AGENT_MM_MAX_MODEL_CALLS:-8}"
export MINI_AGENT_MM_MAX_TOOL_CALLS="${MINI_AGENT_MM_MAX_TOOL_CALLS:-15}"
export MINI_AGENT_PADDLE_OCR_TIMEOUT="${MINI_AGENT_PADDLE_OCR_TIMEOUT:-300}"

export BROWSECOMP_PLUS_RETRIEVER="${BROWSECOMP_PLUS_RETRIEVER:-qwen3-embedding-8b}"
export BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS="${BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS:-600000}"
export BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS="${BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS:-20}"
export BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND="${BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND:-3}"

split_shards() {
  local n="$1"
  local tasks="$2"
  local shard_dir="$3"

  if [[ ! -f "${tasks}" ]]; then
    echo "[run-shards] missing task file: ${tasks}" >&2
    exit 1
  fi

  mkdir -p "${shard_dir}"

  python3 - "${n}" "${tasks}" "${shard_dir}" <<'PY'
import sys
from pathlib import Path

n = int(sys.argv[1])
src = Path(sys.argv[2])
out = Path(sys.argv[3])

handles = [(out / f"tasks_{i:02d}.jsonl").open("w", encoding="utf-8") for i in range(n)]
try:
    for idx, line in enumerate(src.open(encoding="utf-8")):
        if line.strip():
            handles[idx % n].write(line)
finally:
    for h in handles:
        h.close()

print(f"wrote {n} shards to {out}")
PY
}

merge_outputs() {
  local shard_dir="$1"
  local out_dir="$2"

  find "${shard_dir}" -maxdepth 1 -name "results_*.jsonl" -print0 \
    | sort -z \
    | xargs -0 cat > "${out_dir}/executor_results.jsonl"

  find "${shard_dir}" -maxdepth 1 -name "sft_*.jsonl" -print0 \
    | sort -z \
    | xargs -0 cat > "${out_dir}/router_sft.jsonl"
}

run_code_dataset() {
  local name="code_no_swebench"
  local shard_dir="${OUT}/${name}/shards"

  split_shards "${N_CODE}" "${CODE_TASK_FILE}" "${shard_dir}"

  for idx in $(seq 0 $((N_CODE - 1))); do
    local i
    i="$(printf "%02d" "${idx}")"

    python3 scripts/generate_router_sft.py \
      --tasks "${shard_dir}/tasks_${i}.jsonl" \
      --results-out "${shard_dir}/results_${i}.jsonl" \
      --sft-out "${shard_dir}/sft_${i}.jsonl" \
      --summary-out "${shard_dir}/summary_${i}.json" \
      --per-dataset-output-dir "${shard_dir}/per_dataset_${i}" \
      --category code_debug_edit \
      --executor-backend raw_api \
      --run-all \
      --budget-policy full \
      --max-workers "${CODE_WORKERS}" \
      --target-sft-rows 1000000 \
      --skip-errors \
      > "${shard_dir}/run_${i}.log" 2>&1 &
  done

  wait
  merge_outputs "${shard_dir}" "${OUT}/${name}"
  echo "done: ${OUT}/${name}"
}

run_vqa_dataset() {
  local name="vqa"
  local shard_dir="${OUT}/${name}/shards"

  split_shards "${N_VQA}" "${VQA_TASK_FILE}" "${shard_dir}"

  for idx in $(seq 0 $((N_VQA - 1))); do
    local i ocr_port
    i="$(printf "%02d" "${idx}")"
    ocr_port=$((OCR_BASE_PORT + (idx % OCR_PORT_COUNT)))

    MINI_AGENT_PADDLE_OCR_URL="http://127.0.0.1:${ocr_port}/ocr" \
    python3 scripts/generate_router_sft.py \
      --tasks "${shard_dir}/tasks_${i}.jsonl" \
      --results-out "${shard_dir}/results_${i}.jsonl" \
      --sft-out "${shard_dir}/sft_${i}.jsonl" \
      --summary-out "${shard_dir}/summary_${i}.json" \
      --per-dataset-output-dir "${shard_dir}/per_dataset_${i}" \
      --category multimodal_doc_visual \
      --executor-backend mini_agent \
      --run-all \
      --budget-policy full \
      --max-workers "${VQA_WORKERS}" \
      --target-sft-rows 1000000 \
      --skip-errors \
      > "${shard_dir}/run_${i}.log" 2>&1 &
  done

  wait
  merge_outputs "${shard_dir}" "${OUT}/${name}"
  echo "done: ${OUT}/${name}"
}

run_browsecomp_plus_dataset() {
  local name="browsecomp_plus"
  local shard_dir="${OUT}/${name}/shards"

  split_shards "${N_BROWSE}" "${BROWSE_TASK_FILE}" "${shard_dir}"

  for idx in $(seq 0 $((N_BROWSE - 1))); do
    local i retriever_port
    i="$(printf "%02d" "${idx}")"
    retriever_port=$((BROWSE_RETRIEVER_BASE_PORT + idx))

    BROWSECOMP_PLUS_RETRIEVER_SERVER_URL="http://127.0.0.1:${retriever_port}" \
    python3 scripts/generate_router_sft.py \
      --tasks "${shard_dir}/tasks_${i}.jsonl" \
      --results-out "${shard_dir}/results_${i}.jsonl" \
      --sft-out "${shard_dir}/sft_${i}.jsonl" \
      --summary-out "${shard_dir}/summary_${i}.json" \
      --per-dataset-output-dir "${shard_dir}/per_dataset_${i}" \
      --category tool_workflow \
      --executor-backend raw_api \
      --run-all \
      --budget-policy full \
      --max-workers "${BROWSE_WORKERS}" \
      --target-sft-rows 1000000 \
      --skip-errors \
      > "${shard_dir}/run_${i}.log" 2>&1 &
  done

  wait
  merge_outputs "${shard_dir}" "${OUT}/${name}"
  echo "done: ${OUT}/${name}"
}

echo "[run-shards] OUT=${OUT}"
echo "[run-shards] code shards=${N_CODE}, vqa shards=${N_VQA}, browse shards=${N_BROWSE}"

run_code_dataset &
run_vqa_dataset &
run_browsecomp_plus_dataset &

wait

echo "all done: ${OUT}"
