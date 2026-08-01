#!/bin/bash
# VLM Router Benchmark - Run All Steps (1-6)
# Run all data preparation steps at once

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default start/end steps
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-6}"

# Paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
# Step 1/4 use evaluation (includes fields like is_correct, etc.)
VLMEVALKIT_EVAL_DIR="${VLMEVALKIT_EVAL_DIR:-vlm_router_data/VLMEvalKit_evaluation}"
# Step 2 uses inference (raw outputs, for output token statistics)
VLMEVALKIT_INFER_DIR="${VLMEVALKIT_INFER_DIR:-vlm_router_data/VLMEvalKit_inference}"

# Options
FORCE=false

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Functions
# ============================================================================

show_usage() {
    cat << EOF
Usage: bash run_all.sh [OPTIONS]

Description:
  Run all data preparation steps of VLM Router Benchmark at once (Steps 1-6).
  
  Step 1: Build benchmark (from VLMEvalKit evaluation)
  Step 2: Calculate token statistics
  Step 3: Build quality/cost matrices
  Step 4: Validate data integrity
  Step 5: Extract features (text + vision)
  Step 6: Evaluate baseline routers

Options:
  --start-from STEP       Start from the specified step (default: 1)
  --end-at STEP           End at the specified step (default: 6)
  --output-dir DIR        Output root directory (default: .)
  --vlmevalkit-eval-dir DIR     VLMEvalKit evaluation directory (default: vlm_router_data/VLMEvalKit_evaluation)
  --vlmevalkit-infer-dir DIR    VLMEvalKit inference directory (default: vlm_router_data/VLMEvalKit_inference)
  --vlmevalkit-dir DIR          Backward-compatible alias of --vlmevalkit-eval-dir
  --force                 Force re-run all steps
  --help, -h              Show this help message

Environment Variables:
  START_STEP              Start step
  END_STEP                End step
  OUTPUT_DIR              Output root directory
  VLMEVALKIT_EVAL_DIR     VLMEvalKit evaluation directory
  VLMEVALKIT_INFER_DIR    VLMEvalKit inference directory

Examples:
  # Run all steps
  bash run_all.sh

  # Start from Step 2
  bash run_all.sh --start-from 2

  # Run only Step 1-3
  bash run_all.sh --end-at 3

  # Force re-run all steps
  bash run_all.sh --force

  # Use environment variables
  START_STEP=2 END_STEP=3 bash run_all.sh

EOF
}

print_step_header() {
    local step_num=$1
    local step_title=$2
    echo ""
    echo "================================================================================"
    echo -e "${CYAN}Step $step_num: $step_title${NC}"
    echo "================================================================================"
}

print_success() {
    echo ""
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo ""
    echo -e "${RED}❌ $1${NC}"
}

print_skip() {
    echo ""
    echo -e "${YELLOW}⏭️  $1${NC}"
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            START_STEP="$2"
            shift 2
            ;;
        --end-at)
            END_STEP="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vlmevalkit-eval-dir)
            VLMEVALKIT_EVAL_DIR="$2"
            shift 2
            ;;
        --vlmevalkit-infer-dir)
            VLMEVALKIT_INFER_DIR="$2"
            shift 2
            ;;
        --vlmevalkit-dir)
            # backward-compatible alias
            VLMEVALKIT_EVAL_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to view help"
            exit 1
            ;;
    esac
done

# Validate step numbers
if [ "$START_STEP" -lt 1 ] || [ "$START_STEP" -gt 6 ]; then
    echo -e "${RED}Error: START_STEP must be between 1 and 6${NC}"
    exit 1
fi

if [ "$END_STEP" -lt 1 ] || [ "$END_STEP" -gt 6 ]; then
    echo -e "${RED}Error: END_STEP must be between 1 and 6${NC}"
    exit 1
fi

if [ "$START_STEP" -gt "$END_STEP" ]; then
    echo -e "${RED}Error: START_STEP cannot be greater than END_STEP${NC}"
    exit 1
fi

# ============================================================================
# Main
# ============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print header
echo "================================================================================"
echo -e "${BLUE}VLM Router Benchmark - Run All Steps${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Step range: $START_STEP - $END_STEP"
echo "  Output directory: $OUTPUT_DIR"
echo "  VLMEvalKit evaluation directory: $VLMEVALKIT_EVAL_DIR"
echo "  VLMEvalKit inference directory: $VLMEVALKIT_INFER_DIR"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force mode: Yes${NC}"
fi
echo ""

# Track start time
start_time=$(date +%s)

# Build force flag
FORCE_FLAG=""
if [ "$FORCE" = true ]; then
    FORCE_FLAG="--force"
fi

# ============================================================================
# Step 1: Build Benchmark
# ============================================================================
if [ "$START_STEP" -le 1 ] && [ "$END_STEP" -ge 1 ]; then
    print_step_header 1 "Build benchmark (from VLMEvalKit evaluation)"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step1_build_benchmark.sh \
        --output-dir "$OUTPUT_DIR" \
        --vlmevalkit-dir "$VLMEVALKIT_EVAL_DIR" \
        $FORCE_FLAG || {
        print_error "Step 1 failed"
        exit 1
    }
    
    print_success "Step 1 completed"
fi

# ============================================================================
# Step 2: Calculate Token Statistics
# ============================================================================
if [ "$START_STEP" -le 2 ] && [ "$END_STEP" -ge 2 ]; then
    print_step_header 2 "Calculate token statistics"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step2_calculate_tokens.sh \
        --output-dir "$OUTPUT_DIR" \
        --vlmevalkit-dir "$VLMEVALKIT_INFER_DIR" \
        $FORCE_FLAG || {
        print_error "Step 2 failed"
        exit 1
    }
    
    print_success "Step 2 completed"
fi

# ============================================================================
# Step 3: Build Quality/Cost Matrices
# ============================================================================
if [ "$START_STEP" -le 3 ] && [ "$END_STEP" -ge 3 ]; then
    print_step_header 3 "Build quality/cost matrices"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step3_build_matrices.sh \
        --output-dir "$OUTPUT_DIR" \
        $FORCE_FLAG || {
        print_error "Step 3 failed"
        exit 1
    }
    
    print_success "Step 3 completed"
fi

# ============================================================================
# Step 4: Validate Data Integrity
# ============================================================================
if [ "$START_STEP" -le 4 ] && [ "$END_STEP" -ge 4 ]; then
    print_step_header 4 "Validate data integrity"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step4_validate_data.sh \
        --output-dir "$OUTPUT_DIR" \
        --vlmevalkit-dir "$VLMEVALKIT_EVAL_DIR" \
        $FORCE_FLAG || {
        print_error "Step 4 failed"
        exit 1
    }
    
    print_success "Step 4 completed"
fi

# ============================================================================
# Step 5: Extract Features
# ============================================================================
if [ "$START_STEP" -le 5 ] && [ "$END_STEP" -ge 5 ]; then
    print_step_header 5 "Extract features (text + vision)"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step5_extract_features.sh \
        --output-dir "$OUTPUT_DIR" \
        $FORCE_FLAG || {
        print_error "Step 5 failed"
        exit 1
    }
    
    print_success "Step 5 completed"
fi

# ============================================================================
# Step 6: Evaluate Baseline Routers
# ============================================================================
if [ "$START_STEP" -le 6 ] && [ "$END_STEP" -ge 6 ]; then
    print_step_header 6 "Evaluate baseline routers"
    
    cd "$SCRIPT_DIR/.."
    bash scripts/run_step6_evaluate_baselines.sh \
        --output-dir "$OUTPUT_DIR" \
        $FORCE_FLAG || {
        print_error "Step 6 failed"
        exit 1
    }
    
    print_success "Step 6 completed"
fi

# ============================================================================
# Summary
# ============================================================================

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ All steps completed!${NC}"
echo "================================================================================"
echo ""
echo "Completed steps: $START_STEP - $END_STEP"
echo "Total time: ${minutes}m${seconds}s"
echo ""
echo "Generated files:"
if [ "$END_STEP" -ge 1 ]; then
    echo "  📁 BENCHMARKS/          - Sample data"
    echo "  📁 ORACLE/score/        - Scoring data"
    echo "  📁 SPLITS/              - Data splits"
fi
if [ "$END_STEP" -ge 2 ]; then
    echo "  📄 reports/token_statistics/  - Token statistics"
fi
if [ "$END_STEP" -ge 3 ]; then
    echo "  📊 data/matrices/       - Quality/cost matrices"
    echo "  📋 data/registry/       - Metadata"
fi
if [ "$END_STEP" -ge 4 ]; then
    echo "  📝 reports/data_integrity/  - Validation reports"
fi
if [ "$END_STEP" -ge 5 ]; then
    echo "  🔢 EMBEDDINGS/text/         - Text features"
    echo "  🖼️  EMBEDDINGS/vision/       - Vision features"
fi
if [ "$END_STEP" -ge 6 ]; then
    echo "  📄 reports/baselines_evaluation/  - Baselines evaluation summary"
fi
echo ""

if [ "$END_STEP" -ge 4 ]; then
    echo "Data preparation is complete! Now you can:"
    echo "  1. View validation report: cat reports/data_integrity/validation_report.txt"
    if [ "$END_STEP" -ge 6 ]; then
        echo "  2. View baselines summary: cat reports/baselines_evaluation/summary.csv"
        echo "  3. Train router models with extracted features (if needed)"
    elif [ "$END_STEP" -ge 5 ]; then
        echo "  2. Inspect extracted features: ls -lh EMBEDDINGS/"
        echo "  3. Continue to Step 6 to evaluate baselines: bash scripts/run_step6_evaluate_baselines.sh"
    else
        echo "  2. Continue to Step 5 to extract features: bash scripts/run_step5_extract_features.sh"
    fi
    echo ""
fi

echo "Script usage:"
echo "  View help: bash scripts/run_all.sh --help"
echo "  Run specific steps: bash scripts/run_all.sh --start-from 2 --end-at 3"
echo ""
