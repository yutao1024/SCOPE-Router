#!/bin/bash
# VLM Router Benchmark - Step 2: Calculate Token Statistics
# Calculate accurate token statistics for token-based cost computation

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-vlm_router_data/VLMEvalKit_inference}"
TOKEN_STATS_DIR="${TOKEN_STATS_DIR:-reports/token_statistics}"

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
Usage: bash run_step2_calculate_tokens.sh [OPTIONS]

Description:
  Calculate accurate token statistics for token-based cost computation.
  Extract raw model outputs from VLMEvalKit inference results and use a real tokenizer to count tokens.

Options:
  --output-dir DIR        Output root directory (default: .)
  --vlmevalkit-dir DIR    VLMEvalKit inference directory (default: vlm_router_data/VLMEvalKit_inference)
  --vlmevalkit-inference-dir DIR  Same as --vlmevalkit-dir (a clearer alias)
  --token-stats-dir DIR   Token statistics output directory (default: reports/token_statistics)
  --force                 Force recomputation
  --help, -h              Show this help message

Environment Variables:
  OUTPUT_DIR              Output root directory
  VLMEVALKIT_DIR          VLMEvalKit inference directory
  TOKEN_STATS_DIR         Token statistics output directory

Prerequisites:
  - Step 1 must be completed (BENCHMARKS/ and ORACLE/score/ must exist)

Output:
  - reports/token_statistics/token_statistics_report.txt    Detailed report
  - reports/token_statistics/token_based_costs.csv          Token-based costs
  - reports/token_statistics/token_counts.csv               Token count statistics

Examples:
  # Use default paths
  bash run_step2_calculate_tokens.sh

  # Specify VLMEvalKit directory
  bash run_step2_calculate_tokens.sh --vlmevalkit-dir vlm_router_data/VLMEvalKit_inference

  # Force recomputation
  bash run_step2_calculate_tokens.sh --force

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
    python -c "import pandas, numpy, transformers" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install pandas numpy transformers"
        exit 1
    }
    
    echo -e "${GREEN}✓ Dependency check passed${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    if [ ! -d "$OUTPUT_DIR/BENCHMARKS" ]; then
        echo -e "${RED}❌ BENCHMARKS directory does not exist: $OUTPUT_DIR/BENCHMARKS${NC}"
        echo "Please run Step 1 first: bash scripts/run_step1_build_benchmark.sh"
        exit 1
    fi
    
    if [ ! -d "$OUTPUT_DIR/ORACLE/score" ]; then
        echo -e "${RED}❌ ORACLE/score directory does not exist: $OUTPUT_DIR/ORACLE/score${NC}"
        echo "Please run Step 1 first: bash scripts/run_step1_build_benchmark.sh"
        exit 1
    fi
    
    if [ ! -d "$VLMEVALKIT_DIR" ]; then
        echo -e "${RED}❌ VLMEvalKit directory does not exist: $VLMEVALKIT_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisite check passed${NC}"
}

check_output_exists() {
    local report_file="$TOKEN_STATS_DIR/token_statistics_report.txt"
    local costs_file="$TOKEN_STATS_DIR/token_based_costs.csv"
    
    if [ -f "$report_file" ] && [ -f "$costs_file" ]; then
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
        --vlmevalkit-dir)
            VLMEVALKIT_DIR="$2"
            shift 2
            ;;
        --vlmevalkit-inference-dir)
            VLMEVALKIT_DIR="$2"
            shift 2
            ;;
        --token-stats-dir)
            TOKEN_STATS_DIR="$2"
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
echo -e "${BLUE}VLM Router Benchmark - Step 2: Calculate Token Statistics${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  VLMEvalKit directory: $VLMEVALKIT_DIR"
echo "  Token stats directory: $TOKEN_STATS_DIR"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force recompute: Yes${NC}"
fi
echo ""

# Check dependencies
check_dependencies

# Check prerequisites
check_prerequisites

# Check if output exists
if check_output_exists; then
    if [ "$FORCE" = false ]; then
        echo -e "${YELLOW}⏭️  Token stats already exist; skipping${NC}"
        echo "  Existing files:"
        echo "    - $TOKEN_STATS_DIR/token_statistics_report.txt"
        echo "    - $TOKEN_STATS_DIR/token_based_costs.csv"
        echo "  To recompute, use the --force option"
        echo ""
        echo -e "${GREEN}✅ Step 2 completed (using existing outputs)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Existing token statistics will be overwritten${NC}"
    fi
fi

# Run compute_token_stats
echo "="*80
echo -e "${GREEN}Starting token statistics computation...${NC}"
echo "="*80
echo ""
echo "Notes:"
echo "  - Extract actual model outputs from VLMEvalKit results"
echo "  - Use a real tokenizer to count tokens"
echo "  - Compute token-based cost (input_tokens × input_price + output_tokens × output_price)"
echo "  - This may take a few minutes depending on dataset size"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

python "$TOOLS_DIR/compute_token_stats.py" \
    --dataset_dir "$OUTPUT_DIR" \
    --vlmevalkit_dir "$VLMEVALKIT_DIR" \
    --output_dir "$TOKEN_STATS_DIR"

# Check output
echo ""
echo "="*80
echo -e "${GREEN}Verifying output files...${NC}"
echo "="*80

if [ -f "$TOKEN_STATS_DIR/token_statistics_report.txt" ]; then
    echo -e "${GREEN}✓ token_statistics_report.txt${NC}"
else
    echo -e "${RED}❌ token_statistics_report.txt was not created${NC}"
    exit 1
fi

if [ -f "$TOKEN_STATS_DIR/token_based_costs.csv" ]; then
    echo -e "${GREEN}✓ token_based_costs.csv${NC}"
    lines=$(wc -l < "$TOKEN_STATS_DIR/token_based_costs.csv")
    echo "  Rows: $((lines - 1))"
else
    echo -e "${RED}❌ token_based_costs.csv was not created${NC}"
    exit 1
fi

if [ -f "$TOKEN_STATS_DIR/token_counts.csv" ]; then
    echo -e "${GREEN}✓ token_counts.csv${NC}"
else
    echo -e "${YELLOW}⚠️  token_counts.csv was not created (optional)${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 2: Calculate Token Statistics completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📄 $TOKEN_STATS_DIR/token_statistics_report.txt  - Detailed report"
echo "  📊 $TOKEN_STATS_DIR/token_based_costs.csv        - Token-based cost data"
echo "  📊 $TOKEN_STATS_DIR/token_counts.csv             - Token count statistics"
echo ""
echo "View report:"
echo "  cat $TOKEN_STATS_DIR/token_statistics_report.txt"
echo ""
echo "Next step:"
echo "  bash scripts/run_step3_build_matrices.sh  # build quality/cost matrices"
echo ""
