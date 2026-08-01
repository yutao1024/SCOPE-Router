#!/bin/bash
# VLM Router Benchmark - Step 1: Build Benchmark from VLMEvalKit
# Build the Router Benchmark dataset from VLMEvalKit evaluation results

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-vlm_router_data/VLMEvalKit_evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TSV_BASE_DIR="${TSV_BASE_DIR:-vlm_router_data/TSV_images}"
CONFIG_DIR="${CONFIG_DIR:-./config}"

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
Usage: bash run_step1_build_benchmark.sh [OPTIONS]

Options:
  --vlmevalkit-dir DIR    VLMEvalKit output directory (default: vlm_router_data/VLMEvalKit_evaluation)
  --output-dir DIR        Benchmark output directory (default: .)
  --tsv-base-dir DIR      TSV image directory (default: vlm_router_data/TSV_images)
  --config-dir DIR        Config directory (default: ./config)
  --force                 Force rebuild and overwrite existing outputs
  --help, -h              Show this help message

Environment Variables:
  VLMEVALKIT_DIR          VLMEvalKit output directory
  OUTPUT_DIR              Benchmark output directory
  TSV_BASE_DIR            TSV image directory
  CONFIG_DIR              Config directory

Examples:
  # Use default paths
  bash run_step1_build_benchmark.sh

  # Specify VLMEvalKit directory
  bash run_step1_build_benchmark.sh --vlmevalkit-dir vlm_router_data/VLMEvalKit_evaluation

  # Force rebuild
  bash run_step1_build_benchmark.sh --force

Output:
  - BENCHMARKS/           Sample data (*_samples.jsonl)
  - ORACLE/score/         Score data (*.parquet)
  - SPLITS/               Data splits (train.jsonl, dev.jsonl, test.jsonl)
  - data/registry/        Model index (model_index.pkl)

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
    python -c "import pandas, numpy, tqdm, yaml" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install pandas numpy tqdm pyyaml openpyxl"
        exit 1
    }
    
    echo -e "${GREEN}✓ Dependency check passed${NC}"
}

check_input_files() {
    echo "Checking input files..."
    
    if [ ! -d "$VLMEVALKIT_DIR" ]; then
        echo -e "${RED}❌ VLMEvalKit directory does not exist: $VLMEVALKIT_DIR${NC}"
        exit 1
    fi
    
    # Count result files
    xlsx_count=$(find "$VLMEVALKIT_DIR" -name "*.xlsx" 2>/dev/null | wc -l)
    if [ "$xlsx_count" -eq 0 ]; then
        echo -e "${RED}❌ No .xlsx result files found under VLMEvalKit directory${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Found $xlsx_count result files${NC}"
    
    # Check TSV directory (optional)
    if [ -d "$TSV_BASE_DIR" ]; then
        tsv_count=$(find "$TSV_BASE_DIR" -name "*.tsv" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ TSV directory: $tsv_count TSV files${NC}"
    else
        echo -e "${YELLOW}⚠️  TSV directory does not exist; skipping TSV image processing${NC}"
    fi
}

check_output_exists() {
    local sample_count=0
    local score_count=0
    local split_count=0

    if [ -d "$OUTPUT_DIR/BENCHMARKS" ]; then
        sample_count=$(find "$OUTPUT_DIR/BENCHMARKS" -name "*_samples.jsonl" -type f 2>/dev/null | wc -l)
    fi

    if [ -d "$OUTPUT_DIR/ORACLE/score" ]; then
        score_count=$(find "$OUTPUT_DIR/ORACLE/score" -name "*.parquet" -type f 2>/dev/null | wc -l)
    fi

    if [ -d "$OUTPUT_DIR/SPLITS" ]; then
        split_count=$(find "$OUTPUT_DIR/SPLITS" -name "*.jsonl" -type f 2>/dev/null | wc -l)
    fi

    if [ "$sample_count" -gt 0 ] && [ "$score_count" -gt 0 ] && [ "$split_count" -ge 3 ]; then
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
        --vlmevalkit-dir)
            VLMEVALKIT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tsv-base-dir)
            TSV_BASE_DIR="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
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
echo -e "${BLUE}VLM Router Benchmark - Step 1: Build Benchmark${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  VLMEvalKit directory: $VLMEVALKIT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  TSV directory: $TSV_BASE_DIR"
echo "  Config directory: $CONFIG_DIR"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force rebuild: Yes${NC}"
fi
echo ""

# Check dependencies
check_dependencies

# Check input files
check_input_files

# Check if output exists
if check_output_exists; then
    if [ "$FORCE" = false ]; then
        echo -e "${YELLOW}⏭️  Benchmark already exists; skipping build${NC}"
        echo "  Existing directories: BENCHMARKS/, ORACLE/, SPLITS/"
        echo "  To rebuild, use the --force option"
        echo ""
        echo -e "${GREEN}✅ Step 1 completed (using existing outputs)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Existing benchmark outputs will be overwritten${NC}"
    fi
fi

# Run build_benchmark script
echo "="*80
echo -e "${GREEN}Starting benchmark build...${NC}"
echo "="*80
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

python "$TOOLS_DIR/build_benchmark_from_vlmevalkit.py" \
    --vlmevalkit_dir "$VLMEVALKIT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_dir "$CONFIG_DIR" \
    --tsv_base_dir "$TSV_BASE_DIR"

# Check output
echo ""
echo "="*80
echo -e "${GREEN}Verifying output files...${NC}"
echo "="*80

if [ -d "$OUTPUT_DIR/BENCHMARKS" ]; then
    sample_count=$(find "$OUTPUT_DIR/BENCHMARKS" -name "*_samples.jsonl" | wc -l)
    if [ "$sample_count" -gt 0 ]; then
        echo -e "${GREEN}✓ BENCHMARKS/: $sample_count datasets${NC}"
    else
        echo -e "${RED}❌ BENCHMARKS/ contains no *_samples.jsonl files${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ BENCHMARKS/ directory was not created${NC}"
    exit 1
fi

if [ -d "$OUTPUT_DIR/ORACLE/score" ]; then
    score_count=$(find "$OUTPUT_DIR/ORACLE/score" -name "*.parquet" | wc -l)
    if [ "$score_count" -gt 0 ]; then
        echo -e "${GREEN}✓ ORACLE/score/: $score_count score files${NC}"
    else
        echo -e "${RED}❌ ORACLE/score/ contains no parquet score files${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ ORACLE/score/ directory was not created${NC}"
    exit 1
fi

if [ -d "$OUTPUT_DIR/SPLITS" ]; then
    split_count=$(ls -1 "$OUTPUT_DIR/SPLITS"/*.jsonl 2>/dev/null | wc -l)
    if [ "$split_count" -ge 3 ]; then
        echo -e "${GREEN}✓ SPLITS/: $split_count split files${NC}"
    else
        echo -e "${RED}❌ SPLITS/ must contain train/dev/test jsonl files${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ SPLITS/ directory was not created${NC}"
    exit 1
fi

if [ -f "$OUTPUT_DIR/data/registry/model_index.pkl" ]; then
    echo -e "${GREEN}✓ data/registry/model_index.pkl${NC}"
else
    echo -e "${YELLOW}⚠️  model_index.pkl was not created${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 1: Build Benchmark completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📁 BENCHMARKS/       - Sample data ($sample_count datasets)"
echo "  📁 ORACLE/score/     - Score data ($score_count files)"
echo "  📁 SPLITS/           - Data splits ($split_count files)"
echo "  📄 data/registry/    - Model index"
echo ""
echo "Next step:"
echo "  bash scripts/run_all.sh --start-from 2  # continue with subsequent steps"
echo ""
