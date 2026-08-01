#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-.}"
SEED="${SEED:-45}"
SIZE="${SIZE:-1024}"
TEXT_ENCODER="${TEXT_ENCODER:-BAAI/bge-m3}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-large}"
FUSION_METHOD="${FUSION_METHOD:-normalize_concat}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scope_router}"

cd "$(dirname "$0")/.."

mkdir -p CALIBRATION "$OUTPUT_DIR"

TEXT_TAG="${TEXT_ENCODER//\//_}"
VISION_TAG="${VISION_ENCODER//\//_}"
CALIB_NAME="calib_${SIZE}_hybrid_r0p5_d0p3_v0p2_dw0p7_cw0p3_m0p6_h0p2_e0p2_temp0p6_${TEXT_TAG}_${VISION_TAG}"
CALIB_PATH="CALIBRATION/${CALIB_NAME}.jsonl"
PROFILE_PATH="CALIBRATION/${CALIB_NAME}_query_aware_profile.npz"

python tools/select_calibration_set.py \
  --dataset-dir "$DATASET_DIR" \
  --size "$SIZE" \
  --strategy hybrid \
  --hybrid-random-ratio 0.5 \
  --hybrid-diagnostic-ratio 0.3 \
  --hybrid-diversity-ratio 0.2 \
  --diagnostic-disagreement-weight 0.7 \
  --diagnostic-cost-weight 0.3 \
  --diagnostic-medium-ratio 0.6 \
  --diagnostic-hard-ratio 0.2 \
  --diagnostic-easy-ratio 0.2 \
  --temperature 0.6 \
  --text-encoder "$TEXT_ENCODER" \
  --vision-encoder "$VISION_ENCODER" \
  --fusion-method "$FUSION_METHOD" \
  --seed "$SEED" \
  --output-dir CALIBRATION \
  --name "$CALIB_NAME"

python tools/build_calibration_profile.py \
  --dataset-dir "$DATASET_DIR" \
  --calibration-file "$CALIB_PATH" \
  --output-dir CALIBRATION \
  --include-query-embeddings \
  --text-encoder "$TEXT_ENCODER" \
  --vision-encoder "$VISION_ENCODER" \
  --fusion-method "$FUSION_METHOD" \
  --name "${CALIB_NAME}_query_aware_profile"

python routers/scope_router/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --profile_path "$PROFILE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --fusion_method "$FUSION_METHOD" \
  --embedding_dim 64 \
  --query_hidden_dim 128 \
  --profile_hidden_dim 128 \
  --dropout 0.5 \
  --learning_rate 1e-3 \
  --weight_decay 3e-3 \
  --optimizer_type adamw \
  --lr_scheduler none \
  --batch_size 512 \
  --max_iter 100 \
  --temperature 0.07 \
  --score_type dot \
  --loss_type crm \
  --crm_target relevance \
  --crm_bias none \
  --rccr_weight 1.0 \
  --rccr_temperature 0.1 \
  --learn_rccr_temperature \
  --train_lambda 10 \
  --cost_scale 100 \
  --patience 20 \
  --monitor_metric rank_score \
  --seed "$SEED" \
  --skip_latency
