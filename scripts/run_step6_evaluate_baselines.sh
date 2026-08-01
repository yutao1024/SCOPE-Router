#!/bin/bash
# VLM Router Benchmark - Step 6: Evaluate Baseline Routers
# Evaluate baseline routers (Strongest/Cheapest/Oracle/Random) and write summary to reports/baselines_evaluation

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
BASELINES_OUTPUT_DIR="${BASELINES_OUTPUT_DIR:-outputs/baselines_evaluation}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# Options
FORCE=false

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# Functions
# ============================================================================

show_usage() {
    cat << EOF
Usage: bash run_step6_evaluate_baselines.sh [OPTIONS]

Description:
  Evaluate baseline routers (StrongestGlobal/StrongestPerDataset/CheapestGlobal/Oracle/Random).
  Outputs are written by default to: \$OUTPUT_DIR/outputs/baselines_evaluation/

Options:
  --output-dir DIR              Dataset root / output root directory (default: .)
  --baselines-output-dir DIR    Baselines output directory (relative to output-dir or absolute path)
                                (default: outputs/baselines_evaluation)
  --random-seed N               Random router seed (default: 42)
  --force                       Force re-evaluation
  --help, -h                    Show this help message

Environment Variables:
  OUTPUT_DIR
  BASELINES_OUTPUT_DIR
  RANDOM_SEED

Prerequisites:
  - Step 3 must be completed (data/matrices/Y.npz and data/matrices/C.npy exist)
  - Step 1 must be completed (data/registry/model_index.pkl/meta.parquet exist)

Outputs:
  - outputs/baselines_evaluation/summary.csv
  - outputs/baselines_evaluation/detailed_results.json
  - outputs/baselines_evaluation/*_samples.csv
  - outputs/baselines_evaluation/*_test_by_dataset.csv

Examples:
  bash scripts/run_step6_evaluate_baselines.sh
  bash scripts/run_step6_evaluate_baselines.sh --output-dir .
  bash scripts/run_step6_evaluate_baselines.sh --force
  OUTPUT_DIR=. RANDOM_SEED=123 bash scripts/run_step6_evaluate_baselines.sh

EOF
}

check_dependencies() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python is not installed${NC}"
        exit 1
    fi

    # eval_baselines reads parquet and uses numpy/pandas
    python -c "import numpy, pandas, pyarrow" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install numpy pandas pyarrow"
        exit 1
    }
}

check_prerequisites() {
    # Matrices
    if [ ! -f "$OUTPUT_DIR/data/matrices/Y.npz" ]; then
        echo -e "${RED}❌ Missing quality matrix: $OUTPUT_DIR/data/matrices/Y.npz${NC}"
        echo "Please run Step 3 first: bash scripts/run_step3_build_matrices.sh"
        exit 1
    fi

    if [ ! -f "$OUTPUT_DIR/data/matrices/C.npy" ] && [ ! -f "$OUTPUT_DIR/data/matrices/C.npz" ]; then
        echo -e "${RED}❌ Missing cost matrix: $OUTPUT_DIR/data/matrices/C.npy (or C.npz)${NC}"
        echo "Please run Step 3 first: bash scripts/run_step3_build_matrices.sh"
        exit 1
    fi

    # Registry
    if [ ! -f "$OUTPUT_DIR/data/registry/meta.parquet" ]; then
        echo -e "${RED}❌ Missing metadata: $OUTPUT_DIR/data/registry/meta.parquet${NC}"
        echo "Please run Step 3 first: bash scripts/run_step3_build_matrices.sh"
        exit 1
    fi
    if [ ! -f "$OUTPUT_DIR/data/registry/model_index.pkl" ]; then
        echo -e "${RED}❌ Missing model index: $OUTPUT_DIR/data/registry/model_index.pkl${NC}"
        echo "Please run Step 3 first: bash scripts/run_step3_build_matrices.sh"
        exit 1
    fi
}

resolve_baselines_output_dir() {
    # If BASELINES_OUTPUT_DIR is absolute, keep it; otherwise, join with OUTPUT_DIR.
    if [[ "$BASELINES_OUTPUT_DIR" = /* ]]; then
        echo "$BASELINES_OUTPUT_DIR"
    else
        echo "$OUTPUT_DIR/$BASELINES_OUTPUT_DIR"
    fi
}

check_output_exists() {
    local out_dir
    out_dir="$(resolve_baselines_output_dir)"
    if [ -f "$out_dir/summary.csv" ] && [ -f "$out_dir/detailed_results.json" ]; then
        return 0
    fi
    return 1
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --baselines-output-dir)
            BASELINES_OUTPUT_DIR="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
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

# ============================================================================
# Main
# ============================================================================

echo "================================================================================"
echo -e "${BLUE}VLM Router Benchmark - Step 6: Evaluate Baselines${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset directory: $OUTPUT_DIR"
echo "  Output directory: $(resolve_baselines_output_dir)"
echo "  Random seed: $RANDOM_SEED"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force re-evaluation: Yes${NC}"
fi
echo ""

check_dependencies
check_prerequisites

if check_output_exists && [ "$FORCE" = false ]; then
    echo -e "${YELLOW}⏭️  Baselines evaluation already exists; skipping${NC}"
    echo "  Existing files:"
    echo "    - $(resolve_baselines_output_dir)/summary.csv"
    echo "    - $(resolve_baselines_output_dir)/detailed_results.json"
    echo "  To re-evaluate, use the --force option"
    echo ""
    echo -e "${GREEN}✅ Step 6 completed (using existing outputs)${NC}"
    exit 0
fi

# Run evaluation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTERS_DIR="$(cd "$SCRIPT_DIR/../routers" && pwd)"
BASELINES_OUT="$(resolve_baselines_output_dir)"

mkdir -p "$BASELINES_OUT"

PYTHONPATH="$SCRIPT_DIR/.." python "$ROUTERS_DIR/utils/eval_baselines.py" \
    --dataset_dir "$OUTPUT_DIR" \
    --output_dir "$BASELINES_OUT" \
    --random_seed "$RANDOM_SEED"

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 6: Evaluate Baselines completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📄 $BASELINES_OUT/summary.csv"
echo "  📄 $BASELINES_OUT/detailed_results.json"
echo "  📄 $BASELINES_OUT/*_samples.csv"
echo "  📄 $BASELINES_OUT/*_test_by_dataset.csv"
echo ""

