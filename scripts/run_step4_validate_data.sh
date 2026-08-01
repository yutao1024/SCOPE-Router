#!/bin/bash
# VLM Router Benchmark - Step 4: Validate Data Integrity
# Validate data integrity comprehensively to ensure all data is correct and consistent

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-vlm_router_data/VLMEvalKit_evaluation}"
TOKEN_STATS_DIR="${TOKEN_STATS_DIR:-reports/token_statistics}"
VALIDATION_DIR="${VALIDATION_DIR:-reports/data_integrity}"

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
Usage: bash run_step4_validate_data.sh [OPTIONS]

Description:
  Validate data integrity comprehensively, including:
  - Dataset sample count statistics
  - Model result completeness checks
  - Token statistics validation
  - Quality and cost matrix validation
  - Real inference result confirmation

Options:
  --output-dir DIR        Output root directory (default: .)
  --vlmevalkit-dir DIR    VLMEvalKit output directory (default: vlm_router_data/VLMEvalKit_evaluation)
  --token-stats-dir DIR   Token statistics directory (default: reports/token_statistics)
  --validation-dir DIR    Validation report output directory (default: reports/data_integrity)
  --force                 Force re-validation
  --help, -h              Show this help message

Environment Variables:
  OUTPUT_DIR              Output root directory
  VLMEVALKIT_DIR          VLMEvalKit output directory
  TOKEN_STATS_DIR         Token statistics directory
  VALIDATION_DIR          Validation report output directory

Prerequisites:
  - Step 1 must be completed (BENCHMARKS/ and ORACLE/score/ must exist)
  - Step 2 must be completed (token statistics must exist)
  - Step 3 must be completed (matrices must exist)

Output:
  - reports/data_integrity/validation_report.txt          Detailed validation report
  - reports/data_integrity/sample_counts.csv              Sample count statistics
  - reports/data_integrity/model_completeness.csv         Model completeness
  - reports/data_integrity/matrix_validation.txt          Matrix validation results

Examples:
  # Use default paths
  bash run_step4_validate_data.sh

  # 指定自定义路径
  bash run_step4_validate_data.sh \\
      --output-dir . \\
      --validation-dir custom/validation

  # Force re-validation
  bash run_step4_validate_data.sh --force

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
    python -c "import pandas, numpy, scipy" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install pandas numpy scipy"
        exit 1
    }
    
    echo -e "${GREEN}✓ Dependency check passed${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    local has_error=false
    
    # Check Step 1 outputs
    if [ ! -d "$OUTPUT_DIR/BENCHMARKS" ]; then
        echo -e "${RED}❌ BENCHMARKS directory does not exist${NC}"
        has_error=true
    fi
    
    if [ ! -d "$OUTPUT_DIR/ORACLE/score" ]; then
        echo -e "${RED}❌ ORACLE/score directory does not exist${NC}"
        has_error=true
    fi
    
    # Check Step 2 outputs
    if [ ! -d "$TOKEN_STATS_DIR" ]; then
        echo -e "${RED}❌ Token statistics directory does not exist: $TOKEN_STATS_DIR${NC}"
        has_error=true
    fi
    
    # Check Step 3 outputs
    if [ ! -d "$OUTPUT_DIR/data/matrices" ]; then
        echo -e "${RED}❌ Matrices directory does not exist: $OUTPUT_DIR/data/matrices${NC}"
        has_error=true
    fi
    
    if [ "$has_error" = true ]; then
        echo ""
        echo -e "${RED}Please ensure you have completed Steps 1-3:${NC}"
        echo "  1. bash scripts/run_step1_build_benchmark.sh"
        echo "  2. bash scripts/run_step2_calculate_tokens.sh"
        echo "  3. bash scripts/run_step3_build_matrices.sh"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisite check passed${NC}"
}

check_output_exists() {
    local report_file="$VALIDATION_DIR/validation_report.txt"
    
    if [ -f "$report_file" ]; then
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
        --token-stats-dir)
            TOKEN_STATS_DIR="$2"
            shift 2
            ;;
        --validation-dir)
            VALIDATION_DIR="$2"
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
echo -e "${BLUE}VLM Router Benchmark - Step 4: Validate Data Integrity${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  VLMEvalKit directory: $VLMEVALKIT_DIR"
echo "  Token stats directory: $TOKEN_STATS_DIR"
echo "  Validation report directory: $VALIDATION_DIR"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force re-validation: Yes${NC}"
fi
echo ""

# Check dependencies
check_dependencies

# Check prerequisites
check_prerequisites

# Check if output exists
if check_output_exists; then
    if [ "$FORCE" = false ]; then
        echo -e "${YELLOW}⏭️  Validation report already exists; skipping validation${NC}"
        echo "  Existing file: $VALIDATION_DIR/validation_report.txt"
        echo "  To re-validate, use the --force option"
        echo ""
        echo -e "${GREEN}✅ Step 4 completed (using existing outputs)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Existing validation report will be overwritten${NC}"
    fi
fi

# Run validation
echo "="*80
echo -e "${GREEN}Starting data integrity validation...${NC}"
echo "="*80
echo ""
echo "Validation scope:"
echo "  1. Dataset sample count statistics"
echo "  2. Model result completeness checks"
echo "  3. Token statistics validation"
echo "  4. Quality and cost matrix validation"
echo "  5. Real inference result confirmation"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

python "$TOOLS_DIR/validate_data_integrity.py" \
    --dataset_dir "$OUTPUT_DIR" \
    --vlmevalkit_dir "$VLMEVALKIT_DIR" \
    --token_stats_dir "$TOKEN_STATS_DIR" \
    --output_dir "$VALIDATION_DIR"

# Check outputs
echo ""
echo "="*80
echo -e "${GREEN}Verifying output files...${NC}"
echo "="*80

if [ -f "$VALIDATION_DIR/validation_report.txt" ]; then
    echo -e "${GREEN}✓ validation_report.txt${NC}"
else
    echo -e "${RED}❌ validation_report.txt was not created${NC}"
    exit 1
fi

# List all generated files
if [ -d "$VALIDATION_DIR" ]; then
    echo ""
    echo "Generated files:"
    find "$VALIDATION_DIR" -type f -name "*.txt" -o -name "*.csv" | while read file; do
        echo "  - $(basename $file)"
    done
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 4: Validate Data Integrity completed!${NC}"
echo "================================================================================"
echo ""
echo "Validation report:"
echo "  📄 $VALIDATION_DIR/validation_report.txt"
echo ""
echo "View report:"
echo "  cat $VALIDATION_DIR/validation_report.txt"
echo ""
echo "If validation finds issues, please check outputs from prerequisite steps."
echo ""
echo "Next step:"
echo "  If validation passes, you can proceed to router training (Step 5 and beyond)"
echo ""
