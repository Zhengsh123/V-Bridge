# V-Bridge Evaluation Script

An automated inference and evaluation pipeline for video restoration models.

## Overview

This script executes a two-step pipeline:
1. **Inference**: Process low-quality videos using a trained restoration model
2. **Evaluation**: Compare restored videos against ground truth references

## Requirements

- Linux system
- Conda environment manager
- CUDA support (multi-GPU)
- Python environment with required dependencies

## Usage

### Basic Usage

```bash
bash eval.sh
```

The script will automatically:
1. Activate the Conda environment
2. Validate path configurations
3. Display runtime configuration
4. Wait for user confirmation (press `y` to continue)
5. Execute inference step
6. Execute evaluation step

## Configuration

Before running, modify the configuration parameters in the script:

### Base Paths

```bash
BASE_PATH="<your_base_path>"          # Project base path
CONDA_ENV="<your_conda_env>"          # Conda environment path
WORK_DIR="${BASE_PATH}/V-Bridge"      # Working directory
SCRIPT_DIR="<script_directory>"       # Script directory
```

### Inference Parameters

```bash
CHECKPOINT_MODEL=""    # Model checkpoint name
VIDEO_LENGTH=5                        # Video length (frames)
NUM_GPUS=2                            # Number of GPUs to use
RUN_NAME="all"                        # Run identifier
```

### Model and Data Paths

```bash
MODEL_EXPERIMENT="<experiment_name>"  # Model identifier
INPUT_ROOT="<input_directory>"        # Low-quality input videos
MODEL_PATH="<model_weights_path>"     # Model weights path
CONFIG_PATH="<config_file_path>"      # Model configuration file
```

### Evaluation Parameters

```bash
IMAGE_ROOT="<ground_truth_path>"      # Ground truth reference images
SAVE_DIR="<output_directory>"         # Evaluation results save path
```

## Output

### Inference Output

Restored videos are saved to:
```
${OUTPUT_ROOT}/${RUN_NAME}/${CHECKPOINT_MODEL}/
```

### Evaluation Output

Evaluation metrics are saved to the configured `SAVE_DIR`, containing video quality metrics (e.g., PSNR, SSIM).

## Features

- ✅ Automatic path validation
- ✅ Colored log output (info/warning/error)
- ✅ Pre-execution configuration confirmation
- ✅ Automatic error exit on failures
- ✅ Full pipeline automation

## Important Notes

1. **Path Verification**: Ensure all paths are correct before running:
   - Input data directory
   - Model weights path
   - Config file path
   - Ground truth directory

2. **GPU Resources**: Adjust `NUM_GPUS` based on available GPUs

3. **Storage Space**: Ensure sufficient disk space in output directories

4. **Environment**: The script automatically activates the specified Conda environment

## Troubleshooting

If you encounter errors, check:

1. **Path not found**: Verify all configured paths exist
2. **Environment issues**: Confirm Conda environment has required dependencies
3. **GPU issues**: Check CUDA availability and GPU count configuration
4. **Permission issues**: Ensure read/write access to input/output directories

## Related Files

- `eval.sh` - Main evaluation script
- `inference.py` - Inference script
- `eval.py` - Evaluation script
- Model configuration YAML file
