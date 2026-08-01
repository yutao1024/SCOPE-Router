#!/bin/bash
# VLM Router Benchmark - Step 5: Extract Features
# Extract text and vision features for router training

set -e

# Keep conda/project Python isolated from user-site packages on shared servers.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ============================================================================
# Configuration
# ============================================================================

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TEXT_ENCODER="${TEXT_ENCODER:-BAAI/bge-m3}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SKIP_TEXT=false
SKIP_VISION=false

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
Usage: bash run_step5_extract_features.sh [OPTIONS]

Description:
  Extract text and vision features for router training.
  Uses a pre-trained text encoder and vision encoder to extract feature vectors from samples.

Options:
  --output-dir DIR        Output root directory (default: .)
  --text-encoder MODEL    Text encoder model (default: BAAI/bge-m3)
  --vision-encoder MODEL  Vision encoder model (default: facebook/dinov2-base)
  --device DEVICE         Device (default: cuda)
  --batch-size SIZE       Batch size (default: 32)
  --skip-text             Skip text extraction (vision only)
  --skip-vision           Skip vision extraction (text only)
  --force                 Force re-extraction
  --help, -h              Show this help message

Environment Variables:
  OUTPUT_DIR              Output root directory
  TEXT_ENCODER            Text encoder model
  VISION_ENCODER          Vision encoder model
  DEVICE                  Device (cuda/cpu)
  BATCH_SIZE              Batch size

Prerequisites:
  - Step 1 must be completed (BENCHMARKS/ must exist)
  - PyTorch and transformers must be installed
  - GPU is recommended for faster extraction

Output:
  - EMBEDDINGS/text/bge-m3.parquet          Text features
  - EMBEDDINGS/vision/dinov2-base.parquet   Vision features

Models Used:
  - Text: BAAI/bge-m3 (BGE-M3 multilingual embedding)
  - Vision: facebook/dinov2-base (DINOv2 vision encoder)

Examples:
  # Use default config (GPU)
  bash run_step5_extract_features.sh

  # Use CPU
  bash run_step5_extract_features.sh --device cpu

  # Custom batch size
  bash run_step5_extract_features.sh --batch-size 16

  # Run extraction in two stages
  bash run_step5_extract_features.sh --skip-vision  # text only
  bash run_step5_extract_features.sh --skip-text    # vision only

  # Use different encoders
  bash run_step5_extract_features.sh \\
      --text-encoder sentence-transformers/all-MiniLM-L6-v2 \\
      --vision-encoder google/vit-base-patch16-224

  # Force re-extraction
  bash run_step5_extract_features.sh --force

Note:
  The first run will download models automatically. Network access and enough disk space are required.
  Text encoder: ~2GB, vision encoder: ~300MB

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
    python -c "import torch, transformers, pandas" 2>/dev/null || {
        echo -e "${RED}❌ Missing required Python packages${NC}"
        echo "Please install: pip install torch transformers pandas"
        exit 1
    }
    
    # Check CUDA availability if using cuda device
    if [ "$DEVICE" = "cuda" ]; then
        python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
            echo -e "${YELLOW}⚠️  CUDA is not available; switching to CPU automatically${NC}"
            DEVICE="cpu"
        }
    fi
    
    echo -e "${GREEN}✓ Dependency check passed${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Step 1 outputs
    if [ ! -d "$OUTPUT_DIR/BENCHMARKS" ]; then
        echo -e "${RED}❌ BENCHMARKS directory does not exist: $OUTPUT_DIR/BENCHMARKS${NC}"
        echo "Please run Step 1 first: bash scripts/run_step1_build_benchmark.sh"
        exit 1
    fi
    
    # Count sample files
    sample_count=$(find "$OUTPUT_DIR/BENCHMARKS" -name "*_samples.jsonl" | wc -l)
    if [ "$sample_count" -eq 0 ]; then
        echo -e "${RED}❌ No sample files found in BENCHMARKS directory${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisite check passed ($sample_count datasets)${NC}"
}

check_output_exists() {
    if [ "$SKIP_TEXT" = true ] || [ "$SKIP_VISION" = true ]; then
        return 1
    fi

    # Check for both text and vision embeddings
    local text_file="$OUTPUT_DIR/EMBEDDINGS/text/$(basename $TEXT_ENCODER).parquet"
    local vision_file="$OUTPUT_DIR/EMBEDDINGS/vision/$(basename $VISION_ENCODER).parquet"
    
    # For default encoders, use simplified names
    if [ "$TEXT_ENCODER" = "BAAI/bge-m3" ]; then
        text_file="$OUTPUT_DIR/EMBEDDINGS/text/bge-m3.parquet"
    fi
    if [ "$VISION_ENCODER" = "facebook/dinov2-base" ]; then
        vision_file="$OUTPUT_DIR/EMBEDDINGS/vision/dinov2-base.parquet"
    fi
    
    if [ -f "$text_file" ] && [ -f "$vision_file" ]; then
        return 0  # Exists
    else
        return 1  # Not exists
    fi
}

get_output_files() {
    # Return expected output file paths
    local text_file="$OUTPUT_DIR/EMBEDDINGS/text/$(basename $TEXT_ENCODER).parquet"
    local vision_file="$OUTPUT_DIR/EMBEDDINGS/vision/$(basename $VISION_ENCODER).parquet"
    
    # For default encoders, use simplified names
    if [ "$TEXT_ENCODER" = "BAAI/bge-m3" ]; then
        text_file="$OUTPUT_DIR/EMBEDDINGS/text/bge-m3.parquet"
    fi
    if [ "$VISION_ENCODER" = "facebook/dinov2-base" ]; then
        vision_file="$OUTPUT_DIR/EMBEDDINGS/vision/dinov2-base.parquet"
    fi
    
    echo "$text_file $vision_file"
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
        --text-encoder)
            TEXT_ENCODER="$2"
            shift 2
            ;;
        --vision-encoder)
            VISION_ENCODER="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip-text)
            SKIP_TEXT=true
            shift
            ;;
        --skip-vision)
            SKIP_VISION=true
            shift
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
echo -e "${BLUE}VLM Router Benchmark - Step 5: Extract Features${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Text encoder: $TEXT_ENCODER"
echo "  Vision encoder: $VISION_ENCODER"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
if [ "$FORCE" = true ]; then
    echo -e "  ${RED}Force re-extraction: Yes${NC}"
fi
if [ "$SKIP_TEXT" = true ]; then
    echo "  Skip text: Yes"
fi
if [ "$SKIP_VISION" = true ]; then
    echo "  Skip vision: Yes"
fi
echo ""

if [ "$SKIP_TEXT" = true ] && [ "$SKIP_VISION" = true ]; then
    echo -e "${RED}❌ Cannot use --skip-text and --skip-vision together${NC}"
    exit 1
fi

# Check dependencies
check_dependencies

# Check prerequisites
check_prerequisites

# Check if output exists
if check_output_exists; then
    if [ "$FORCE" = false ]; then
        read text_file vision_file <<< "$(get_output_files)"
        echo -e "${YELLOW}⏭️  Feature files already exist; skipping extraction${NC}"
        echo "  Existing files:"
        echo "    - $text_file"
        echo "    - $vision_file"
        echo "  To re-extract, use the --force option"
        echo ""
        echo -e "${GREEN}✅ Step 5 completed (using existing outputs)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Existing feature files will be overwritten${NC}"
    fi
fi

# Run feature extraction
echo "="*80
echo -e "${GREEN}Starting feature extraction...${NC}"
echo "="*80
echo ""
echo "Notes:"
echo "  - Read all samples from BENCHMARKS/"
echo "  - Use text encoder to extract prompt features"
echo "  - Use vision encoder to extract image features"
echo "  - Save to Parquet format (efficient storage)"
echo ""
echo "Tips:"
if [ "$DEVICE" = "cuda" ]; then
    echo "  - Using GPU acceleration; runtime depends on dataset size and GPU performance"
else
    echo "  - Using CPU; this may take a while"
fi
echo "  - The first run will download models automatically (requires network access)"
echo "  - You can interrupt anytime with Ctrl+C"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTERS_DIR="$(cd "$SCRIPT_DIR/../routers" && pwd)"

EXTRA_ARGS=()
if [ "$SKIP_TEXT" = true ]; then
    EXTRA_ARGS+=(--skip_text)
fi
if [ "$SKIP_VISION" = true ]; then
    EXTRA_ARGS+=(--skip_vision)
fi

python "$ROUTERS_DIR/features/extract_cli.py" \
    --dataset_dir "$OUTPUT_DIR" \
    --text_encoder "$TEXT_ENCODER" \
    --vision_encoder "$VISION_ENCODER" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"

# Check outputs
echo ""
echo "="*80
echo -e "${GREEN}Verifying output files...${NC}"
echo "="*80

read text_file vision_file <<< "$(get_output_files)"

if [ "$SKIP_TEXT" = true ]; then
    echo -e "${YELLOW}⏭️  Text features skipped${NC}"
elif [ -f "$text_file" ]; then
    echo -e "${GREEN}✓ Text features: $text_file${NC}"
    python -c "import pandas as pd; df = pd.read_parquet('$text_file'); print(f'  Num samples: {len(df)}, embedding dim: {df.iloc[0][\"embedding\"].shape if len(df) > 0 else \"unknown\"}')" 2>/dev/null || echo "  (Unable to read details)"
else
    echo -e "${RED}❌ Text feature file was not created${NC}"
    exit 1
fi

if [ "$SKIP_VISION" = true ]; then
    echo -e "${YELLOW}⏭️  Vision features skipped${NC}"
elif [ -f "$vision_file" ]; then
    echo -e "${GREEN}✓ Vision features: $vision_file${NC}"
    python -c "import pandas as pd; df = pd.read_parquet('$vision_file'); print(f'  Num samples: {len(df)}, embedding dim: {df.iloc[0][\"embedding\"].shape if len(df) > 0 else \"unknown\"}')" 2>/dev/null || echo "  (Unable to read details)"
else
    echo -e "${RED}❌ Vision feature file was not created${NC}"
    exit 1
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Step 5: Extract Features completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📊 $text_file    - Text features"
echo "  📊 $vision_file  - Vision features"
echo ""
echo "Feature notes:"
echo "  - Text features: semantic vectors extracted from prompts"
echo "  - Vision features: visual vectors extracted from images"
echo "  - These features will be used to train router models"
echo ""
echo "Next step:"
echo "  You can use these features to train various router models (if you have Step 6 and beyond)"
echo ""
