#!/bin/bash
# VLM Router Benchmark - Run Feature-level Routers (no soft-label config needed here)
#
# Routers covered:
# - knn
# - prknn
# - ovr
# - kmeans
#
# Example:
#   bash scripts/run_feature_routers.sh --dataset-dir . --output-base outputs

set -euo pipefail

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# Make script runnable from any working directory (assume repo root is parent of scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-.}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"

TEXT_ENCODER="${TEXT_ENCODER:-BAAI/bge-m3}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
FUSION_METHOD="${FUSION_METHOD:-normalize_concat}"

# KNN / PRKNN
K="${K:-5}"
PRKNN_METRIC="${PRKNN_METRIC:-cosine}"
PRKNN_WEIGHTS="${PRKNN_WEIGHTS:-distance}" # uniform|distance

# KMeans
KMEANS_NORMALIZE="${KMEANS_NORMALIZE:-true}"  # true|false
KMEANS_LAMBDA_COST="${KMEANS_LAMBDA_COST:-0.0}"

# Behavior
STOP_ON_FAILURE=false

show_usage() {
  cat << EOF
Usage: bash scripts/run_feature_routers.sh [OPTIONS]

Options:
  --dataset-dir DIR           Dataset root (default: .)
  --output-base DIR           Output base directory (default: outputs_feature)

  --text-encoder NAME         Text encoder (default: BAAI/bge-m3)
  --vision-encoder NAME       Vision encoder (default: facebook/dinov2-base)
  --fusion-method NAME        Fusion method (default: normalize_concat)

  --k N                       K for knn/prknn (default: 5)
  --prknn-metric NAME         Metric for prknn (default: cosine)
  --prknn-weights NAME        Weights for prknn: uniform|distance (default: distance)

  --kmeans-normalize true|false   Normalize embeddings for kmeans (default: true)
  --kmeans-lambda-cost VAL        Kmeans lambda_cost (default: 0.0)

  --stop-on-failure           Stop immediately if any router fails (default: continue)
  --help, -h                  Show help
EOF
}

need_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "❌ Missing required file: $path"
    return 1
  fi
}

encoder_to_filename() {
  local name="$1"
  if [[ "$name" == */* ]]; then
    echo "${name##*/}"
  else
    echo "$name"
  fi
}

check_embeddings() {
  local text_fn vision_fn
  text_fn="$(encoder_to_filename "$TEXT_ENCODER")"
  vision_fn="$(encoder_to_filename "$VISION_ENCODER")"
  need_file "$DATASET_DIR/EMBEDDINGS/text/${text_fn}.parquet"
  need_file "$DATASET_DIR/EMBEDDINGS/vision/${vision_fn}.parquet"
}

run_one() {
  local name="$1"; shift
  echo ""
  echo "================================================================================"
  echo "🚀 Running: $name"
  echo "Command: $*"
  echo "================================================================================"
  set +e
  "$@"
  local code=$?
  set -e
  if [ $code -ne 0 ]; then
    echo "❌ FAILED: $name (exit=$code)"
    if [ "$STOP_ON_FAILURE" = true ]; then
      exit $code
    fi
  else
    echo "✅ DONE: $name"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --output-base) OUTPUT_BASE="$2"; shift 2;;
    --text-encoder) TEXT_ENCODER="$2"; shift 2;;
    --vision-encoder) VISION_ENCODER="$2"; shift 2;;
    --fusion-method) FUSION_METHOD="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --prknn-metric) PRKNN_METRIC="$2"; shift 2;;
    --prknn-weights) PRKNN_WEIGHTS="$2"; shift 2;;
    --kmeans-normalize) KMEANS_NORMALIZE="$2"; shift 2;;
    --kmeans-lambda-cost) KMEANS_LAMBDA_COST="$2"; shift 2;;
    --stop-on-failure) STOP_ON_FAILURE=true; shift;;
    --help|-h) show_usage; exit 0;;
    *) echo "Unknown option: $1"; echo "Use --help"; exit 1;;
  esac
done

echo "================================================================================"
echo "VLM Router - Feature-level Routers"
echo "================================================================================"
echo "dataset_dir: $DATASET_DIR"
echo "output_base: $OUTPUT_BASE"
echo "text_encoder: $TEXT_ENCODER"
echo "vision_encoder: $VISION_ENCODER"
echo "fusion_method: $FUSION_METHOD"
echo "knn/prknn k: $K"
echo "prknn metric: $PRKNN_METRIC"
echo "prknn weights: $PRKNN_WEIGHTS"
echo "kmeans normalize: $KMEANS_NORMALIZE"
echo "kmeans lambda_cost: $KMEANS_LAMBDA_COST"
echo "================================================================================"

# Common dataset prep checks
need_file "$DATASET_DIR/data/matrices/Y.npz"
if [ ! -f "$DATASET_DIR/data/matrices/C.npy" ] && [ ! -f "$DATASET_DIR/data/matrices/C.npz" ]; then
  echo "❌ Missing cost matrix: $DATASET_DIR/data/matrices/C.npy (or C.npz)"
  exit 1
fi
need_file "$DATASET_DIR/data/registry/meta.parquet"

# Embeddings required by feature-level routers
check_embeddings

mkdir -p "$OUTPUT_BASE"

run_one "knn" python routers/knn/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --k "$K" \
  --fusion_method "$FUSION_METHOD" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --output_dir "$OUTPUT_BASE/knn/k_${K}"

run_one "prknn" python routers/prknn/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --k "$K" \
  --fusion_method "$FUSION_METHOD" \
  --metric "$PRKNN_METRIC" \
  --weights "$PRKNN_WEIGHTS" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --output_dir "$OUTPUT_BASE/prknn/k_${K}_${PRKNN_METRIC}_${PRKNN_WEIGHTS}"

run_one "ovr" python routers/ovr/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --output_dir "$OUTPUT_BASE/ovr"

if [ "$KMEANS_NORMALIZE" = "true" ]; then
  KMEANS_NORMALIZE_ARGS=(--normalize)
else
  KMEANS_NORMALIZE_ARGS=(--no-normalize)
fi

run_one "kmeans" python routers/kmeans/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --fusion_method "$FUSION_METHOD" \
  --lambda_cost "$KMEANS_LAMBDA_COST" \
  "${KMEANS_NORMALIZE_ARGS[@]}" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --output_dir "$OUTPUT_BASE/kmeans/norm_${KMEANS_NORMALIZE}_lambda_${KMEANS_LAMBDA_COST}"

echo ""
echo "================================================================================"
echo "✅ Finished feature-level routers (see logs above for failures, if any)"
echo "Outputs: $OUTPUT_BASE"
echo "================================================================================"
