#!/bin/bash

# ============================================================================
# V-Bridge Evaluation Script - Inference + Evaluation Pipeline
# ============================================================================

set -e  # Exit on any command failure

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration Parameters
# ============================================================================

# Base paths
BASE_PATH="./COF"
CONDA_ENV=""
WORK_DIR="${BASE_PATH}/V-Bridge"
SCRIPT_DIR="${BASE_PATH}/V-Bridge/V-Bridge"

# Inference parameters
CHECKPOINT_MODEL=""
VIDEO_LENGTH=5
NUM_GPUS=2
RUN_NAME="all"

# Experiment names (for composing paths)
MODEL_EXPERIMENT="wan22-ti2v-v-bridge"

# Composed paths
INPUT_ROOT="${BASE_PATH}/data/test/restoration/LQ"
OUTPUT_ROOT="${WORK_DIR}/Restoration/inference/${MODEL_EXPERIMENT}"
MODEL_PATH="${BASE_PATH}/train_output/VideoXFun-Restoration/${MODEL_EXPERIMENT}/${CHECKPOINT_MODEL}"
CONFIG_PATH="${WORK_DIR}/config/wan2.2/wan_civitai_5b.yaml"

# Evaluation parameters
VIDEO_ROOT="${OUTPUT_ROOT}/${RUN_NAME}/${CHECKPOINT_MODEL}"
IMAGE_ROOT="${BASE_PATH}/data/test/restoration/GT"
SAVE_DIR="${WORK_DIR}/Restoration/inference/eval/${MODEL_EXPERIMENT}/${CHECKPOINT_MODEL}"

# ============================================================================
# Function Definitions
# ============================================================================

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

validate_paths() {
    print_info "Validating paths..."

    if [ ! -f "$CONFIG_PATH" ]; then
        print_error "Config file not found: $CONFIG_PATH"
        return 1
    fi

    if [ ! -d "$INPUT_ROOT" ]; then
        print_error "Input directory not found: $INPUT_ROOT"
        return 1
    fi

    print_info "Path validation passed"
    return 0
}

print_config() {
    print_info "========== Running Configuration =========="
    echo "Base path: $BASE_PATH"
    echo "Work directory: $WORK_DIR"
    echo ""
    echo "Inference Configuration:"
    echo "  Input: $INPUT_ROOT"
    echo "  Output: $OUTPUT_ROOT"
    echo "  Model: $MODEL_PATH"
    echo "  Config: $CONFIG_PATH"
    echo "  Video length: $VIDEO_LENGTH"
    echo "  Number of GPUs: $NUM_GPUS"
    echo ""
    echo "Evaluation Configuration:"
    echo "  Video: $VIDEO_ROOT"
    echo "  Image: $IMAGE_ROOT"
    echo "  Save: $SAVE_DIR"
    echo "=========================================="
}

# ============================================================================
# Main Pipeline
# ============================================================================

main() {
    print_info "Starting V-Bridge evaluation pipeline"

    # Initialize environment
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    print_info "Conda environment activated: $CONDA_ENV"

    # Switch to work directory
    cd "$WORK_DIR"
    print_info "Working directory: $(pwd)"

    # Validate paths
    validate_paths || exit 1

    # Print configuration
    print_config

    # Confirm execution
    read -p "Continue execution? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Execution cancelled by user"
        exit 0
    fi

    # Step 1: Inference
    print_info "========== Step 1: Inference =========="
    python "$SCRIPT_DIR/inference.py" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --model_name "$MODEL_PATH" \
        --config_path "$CONFIG_PATH" \
        --video_length "$VIDEO_LENGTH" \
        --num_gpus "$NUM_GPUS" \
        --run_name "$RUN_NAME"

    print_info "Inference completed"

    # Step 2: Evaluation
    print_info "========== Step 2: Evaluation =========="
    mkdir -p "$SAVE_DIR"
    python "$SCRIPT_DIR/eval.py" \
        --video_root "$VIDEO_ROOT" \
        --image_root "$IMAGE_ROOT" \
        --save_dir "$SAVE_DIR"

    print_info "Evaluation completed"
    print_info "Results saved to: $SAVE_DIR"
}

# Execute main function
main "$@"
