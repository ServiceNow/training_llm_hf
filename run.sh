#!/bin/bash

# Launch Script for SFT Training Pipeline
# Usage: ./launch_training.sh [config_file] [num_gpus]

set -e

# Default values
CONFIG_FILE="config.yaml"
NUM_GPUS=1
MASTER_PORT=12355

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -c, --config FILE    Configuration file (default: config.yaml)"
            echo "  -g, --gpus NUM       Number of GPUs (default: 1)"
            echo "  -p, --port PORT      Master port for distributed training (default: 12355)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or use the --config option to specify one."
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: Number of GPUs must be a positive integer!"
    exit 1
fi

# Check if required dependencies are installed
check_dependency() {
    if ! python -c "import $1" &> /dev/null; then
        echo "Error: Required dependency '$1' is not installed!"
        echo "Please install it using: pip install $1"
        return 1
    fi
}

echo "Checking dependencies..."
check_dependency "torch" || exit 1
check_dependency "transformers" || exit 1
check_dependency "datasets" || exit 1

# Optional dependencies
if ! python -c "import wandb" &> /dev/null; then
    echo "Warning: wandb is not installed. W&B logging will be disabled."
fi

if ! python -c "import peft" &> /dev/null; then
    echo "Warning: peft is not installed. LoRA training will not be available."
fi

# Print configuration
echo "=========================================="
echo "SFT Training Pipeline Launch Configuration"
echo "=========================================="
echo "Configuration file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export MASTER_ADDR="localhost"
export MASTER_PORT="$MASTER_PORT"

# Launch training
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    python sft_main.py --config_file "$CONFIG_FILE"
else
    # Multi-GPU distributed training
    echo "Starting distributed training on $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port="$MASTER_PORT" \
        sft_main.py \
        --config_file "$CONFIG_FILE"
fi

echo "Training completed!"

# Optional: Run evaluation if specified
# Uncomment the following lines if you want to run evaluation after training
# echo "Running final evaluation..."
# python evaluate_model.py --config_file "$CONFIG_FILE" --model_path "./sft_output"