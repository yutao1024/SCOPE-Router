#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="."
OUTPUT_DIR="outputs/router_throughput/all_methods"
SPLIT="test"
LIMIT_SAMPLES="1024"
BATCH_SIZES="1,8,16,32,64"
WARMUP_RUNS="5"
TEST_RUNS="50"
TEXT_ENCODER="BAAI/bge-m3"
VISION_ENCODER="facebook/dinov2-large"
DEVICE="cuda"
MODE="online"
ONLINE_FEATURE_STRATEGY="composed"
INCLUDE_BASELINES="1"
SCAN_DIRS=()
ROUTERS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_router_throughput_benchmark.sh [options]

Options:
  --dataset-dir DIR        Dataset dir (default: .)
  --output-dir DIR         Output dir (default: outputs/router_throughput/all_methods)
  --split SPLIT            train/dev/test (default: test)
  --limit-samples N        Samples available for profiling (default: 1024)
  --batch-sizes LIST       Example: 1,8,16,32,64
  --warmup-runs N          Warmup repeats (default: 5)
  --test-runs N            Timed repeats (default: 50)
  --text-encoder NAME      Text embedding encoder (default: BAAI/bge-m3)
  --vision-encoder NAME    Vision embedding encoder (default: facebook/dinov2-large)
  --device cuda|cpu        Device for loading GPU routers (default: cuda)
  --mode offline|online    offline=decision only, online=include frozen feature encoders (default: online)
  --online-feature-strategy composed|inline
                           composed=measure shared BGE/DINO once then add router latency (default)
  --scan-dir DIR           Add directory to recursively scan for .pkl routers
  --router SPEC            Add explicit router spec: name=method:path or method:path
  --no-baselines           Do not include simple baselines

If no --scan-dir is supplied, outputs/ is scanned recursively.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --limit-samples) LIMIT_SAMPLES="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --warmup-runs) WARMUP_RUNS="$2"; shift 2 ;;
    --test-runs) TEST_RUNS="$2"; shift 2 ;;
    --text-encoder) TEXT_ENCODER="$2"; shift 2 ;;
    --vision-encoder) VISION_ENCODER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --online-feature-strategy) ONLINE_FEATURE_STRATEGY="$2"; shift 2 ;;
    --scan-dir) SCAN_DIRS+=("$2"); shift 2 ;;
    --router) ROUTERS+=("$2"); shift 2 ;;
    --no-baselines) INCLUDE_BASELINES="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ${#SCAN_DIRS[@]} -eq 0 ]]; then
  [[ -d outputs ]] && SCAN_DIRS+=(outputs)
fi

cmd=(
  python scripts/benchmark_router_throughput.py
  --dataset_dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --limit-samples "$LIMIT_SAMPLES"
  --batch-sizes "$BATCH_SIZES"
  --warmup-runs "$WARMUP_RUNS"
  --test-runs "$TEST_RUNS"
  --text-encoder "$TEXT_ENCODER"
  --vision-encoder "$VISION_ENCODER"
  --device "$DEVICE"
  --mode "$MODE"
  --online-feature-strategy "$ONLINE_FEATURE_STRATEGY"
)

if [[ "$INCLUDE_BASELINES" == "1" ]]; then
  cmd+=(--include-baselines)
fi

for d in "${SCAN_DIRS[@]}"; do
  cmd+=(--scan-dir "$d")
done

for r in "${ROUTERS[@]}"; do
  cmd+=(--router "$r")
done

echo "[run] ${cmd[*]}"
"${cmd[@]}"
