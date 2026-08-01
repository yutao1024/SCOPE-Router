#!/bin/bash
# VLM Router Benchmark - Step 3: Build Quality/Cost Matrices
# Build quality matrix (Y) and cost matrix (C) for router training

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TOKEN_STATS_DIR="${TOKEN_STATS_DIR:-reports/token_statistics}"
PRICING_CONFIG="${PRICING_CONFIG:-config/pricing.yaml}"
MATRICES_DIR="${MATRICES_DIR:-data/matrices}"

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
Usage: bash run_step3_build_matrices.sh [OPTIONS]

Description:
  Build quality matrix (Y) and cost matrix (C) for router training.
  - Quality matrix: model correctness on each sample (0 or 1)
  - Cost matrix: token-based true cost

Options:
  --output-dir DIR        Output root directory (default: .)
  --token-stats-dir DIR   Token statistics directory (default: reports/token_statistics)
  --pricing-config FILE   Pricing config file (default: config/pricing.yaml)
  --matrices-dir DIR      Matrices output directory (default: data/matrices)
  --force                 Force rebuild
  --help, -h              Show this help message

Environment Variables:
  OUTPUT_DIR              Output root directory
  TOKEN_STATS_DIR         Token statistics directory
  PRICING_CONFIG          Pricing config file
  MATRICES_DIR            Matrices output directory

Prerequisites:
  - Step 1 must be completed (BENCHMARKS/ and ORACLE/score/ must exist)
  - Step 2 must be completed (token_based_costs.csv must exist)

Output:
  - data/matrices/Y.npz               Quality matrix (N×K, key is Y in the npz)
  - data/matrices/C.npy               Cost matrix
  - data/matrices/cost_bounds.json    Cost bounds for Arena Score
  - data/registry/meta.parquet        Metadata

Examples:
  # Use default paths
  bash run_step3_build_matrices.sh

  # Use custom paths
  bash run_step3_build_matrices.sh \\
      --token-stats-dir custom/token_stats \\
      --matrices-dir custom/matrices

  # Force rebuild
  bash run_step3_build_matrices.sh --force

EOF
}

check_dependencies() {
    echo "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python is not installed${NC}"
        exit 1
    fi
    
    # Check required Python packages
    python -c "import pandas, numpy, scipy, yaml" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install pandas numpy scipy pyyaml"
        exit 1
    }
    
    echo -e "${GREEN}✓ Dependency check passed${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Step 1 outputs
    if [ ! -d "$OUTPUT_DIR/BENCHMARKS" ]; then
        echo -e "${RED}❌ BENCHMARKS directory does not exist${NC}"
        echo "Please run Step 1 first: bash scripts/run_step1_build_benchmark.sh"
        exit 1
    fi
    
    if [ ! -d "$OUTPUT_DIR/ORACLE/score" ]; then
        echo -e "${RED}❌ ORACLE/score directory does not exist${NC}"
        echo "Please run Step 1 first: bash scripts/run_step1_build_benchmark.sh"
        exit 1
    fi
    
    # Check Step 2 outputs
    if [ ! -f "$TOKEN_STATS_DIR/token_based_costs.csv" ]; then
        echo -e "${RED}❌ Token stats file not found: $TOKEN_STATS_DIR/token_based_costs.csv${NC}"
        echo "Please run Step 2 first: bash scripts/run_step2_calculate_tokens.sh"
        exit 1
    fi
    
    # Check pricing config
    if [ ! -f "$PRICING_CONFIG" ]; then
        echo -e "${RED}❌ Pricing config file not found: $PRICING_CONFIG${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisite check passed${NC}"
}

check_output_exists() {
    if [ -f "$MATRICES_DIR/Y.npz" ] && \
       [ -f "$MATRICES_DIR/C.npy" ] && \
       [ -f "$MATRICES_DIR/cost_bounds.json" ] && \
       [ -f "$OUTPUT_DIR/data/registry/meta.parquet" ]; then
        return 0  # Exists
    else
        return 1  # Not exists
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --token-stats-dir)
            TOKEN_STATS_DIR="$2"
            shift 2
            ;;
        --pricing-config)
            PRICING_CONFIG="$2"
            shift 2
            ;;
        --matrices-dir)
            MATRICES_DIR="$2"
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
echo -e "${BLUE}VLM Router Benchmark - Step 3: Build Quality/Cost Matrices${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Token stats directory: $TOKEN_STATS_DIR"
echo "  Pricing config: $PRICING_CONFIG"
echo "  Matrices output directory: $MATRICES_DIR"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force rebuild: Yes${NC}"
fi
echo ""

# Check dependencies
check_dependencies

# Check prerequisites
check_prerequisites

# Check if output exists
if check_output_exists; then
    if [ "$FORCE" = false ]; then
        echo -e "${YELLOW}⏭️  Matrix files already exist; skipping build${NC}"
        echo "  Existing files:"
        echo "    - $MATRICES_DIR/Y.npz (quality matrix)"
        echo "    - $MATRICES_DIR/C.npy (cost matrix)"
        echo "    - $MATRICES_DIR/cost_bounds.json (Arena Score bounds)"
        echo "    - data/registry/meta.parquet (metadata)"
        echo "  To rebuild, use the --force option"
        echo ""
        echo -e "${GREEN}✅ Step 3 completed (using existing outputs)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Existing matrix files will be overwritten${NC}"
    fi
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

# Part 1: Build Y matrix and metadata
echo "="*80
echo -e "${GREEN}Part 1: Build quality matrix (Y) and metadata...${NC}"
echo "="*80
echo ""
echo "Notes:"
echo "  - Merge scoring data from all datasets"
echo "  - Build sparse quality matrix Y[sample_idx, model_idx]"
echo "  - Generate metadata (sample_id/model_id mapping)"
echo ""

python "$TOOLS_DIR/build_matrices.py" --dataset_dir "$OUTPUT_DIR"

# Check intermediate outputs
if [ -f "$MATRICES_DIR/Y.npz" ]; then
    echo -e "${GREEN}✓ Y.npz created successfully${NC}"
else
    echo -e "${RED}❌ Failed to create Y.npz${NC}"
    exit 1
fi

if [ -f "$OUTPUT_DIR/data/registry/meta.parquet" ]; then
    echo -e "${GREEN}✓ meta.parquet created successfully${NC}"
else
    echo -e "${RED}❌ Failed to create meta.parquet${NC}"
    exit 1
fi

# Part 2: Build C matrix and cost bounds
echo ""
echo "="*80
echo -e "${GREEN}Part 2: Build cost matrix (C) and Arena Score bounds...${NC}"
echo "="*80
echo ""
echo "Notes:"
echo "  - Use token statistics from Step 2"
echo "  - Compute cost per sample (input_tokens × input_price + output_tokens × output_price)"
echo "  - Generate cost bounds for Arena Score (min/max cost)"
echo ""

python "$TOOLS_DIR/build_cost_matrix_from_tokens.py" \
    --dataset-dir "$OUTPUT_DIR" \
    --pricing "$PRICING_CONFIG" \
    --token-stats "$TOKEN_STATS_DIR" \
    --output "$MATRICES_DIR"

# Check outputs
echo ""
echo "="*80
echo -e "${GREEN}Verifying output files...${NC}"
echo "="*80

if [ -f "$MATRICES_DIR/Y.npz" ]; then
    echo -e "${GREEN}✓ Y.npz (quality matrix)${NC}"
    python -c "import numpy as np; d=np.load('$MATRICES_DIR/Y.npz'); Y=d['Y'] if 'Y' in d.files else d[d.files[0]]; nnz=int((Y!=0).sum()); print(f'  Shape: {Y.shape}, non-zeros: {nnz}')"
else
    echo -e "${RED}❌ Y.npz does not exist${NC}"
    exit 1
fi

if [ -f "$MATRICES_DIR/C.npy" ]; then
    echo -e "${GREEN}✓ C.npy (cost matrix)${NC}"
    python -c "import numpy as np; C = np.load('$MATRICES_DIR/C.npy'); print(f'  Shape: {C.shape}')"
else
    echo -e "${RED}❌ C.npy does not exist${NC}"
    exit 1
fi

if [ -f "$MATRICES_DIR/cost_bounds.json" ]; then
    echo -e "${GREEN}✓ cost_bounds.json${NC}"
    cat "$MATRICES_DIR/cost_bounds.json"
else
    echo -e "${RED}❌ cost_bounds.json does not exist${NC}"
    exit 1
fi

if [ -f "$OUTPUT_DIR/data/registry/meta.parquet" ]; then
    echo -e "${GREEN}✓ data/registry/meta.parquet (metadata)${NC}"
    python -c "import pandas as pd; meta = pd.read_parquet('$OUTPUT_DIR/data/registry/meta.parquet'); print(f'  Num samples: {len(meta)}')"
else
    echo -e "${RED}❌ meta.parquet does not exist${NC}"
    exit 1
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 3: Build Quality/Cost Matrices completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📊 $MATRICES_DIR/Y.npz                - Quality matrix (model correctness)"
echo "  💰 $MATRICES_DIR/C.npy                - Cost matrix (token-based)"
echo "  📏 $MATRICES_DIR/cost_bounds.json     - Arena Score cost bounds"
echo "  📋 data/registry/meta.parquet         - Sample/model metadata"
echo ""
echo "Next step:"
echo "  bash scripts/run_step4_validate_data.sh  # validate data integrity"
echo ""
