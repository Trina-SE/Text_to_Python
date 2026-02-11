#!/bin/bash
# ==============================================================
# Docstring-to-Code Generation - Single Script Runner
# ==============================================================
# This script builds the Docker image and runs the full pipeline:
#   1. Downloads the CodeSearchNet Python dataset
#   2. Trains all three Seq2Seq models (RNN, LSTM, LSTM+Attention)
#   3. Evaluates models and generates comparison plots
#   4. Creates attention visualizations
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                     # Train and evaluate all models
#   ./run.sh --model rnn         # Train only Vanilla RNN
#   ./run.sh --model lstm        # Train only LSTM
#   ./run.sh --model attention   # Train only LSTM with Attention
#   ./run.sh --eval-only         # Evaluate existing checkpoints
#   ./run.sh --epochs 5          # Train for 5 epochs
# ==============================================================

set -e

IMAGE_NAME="docstring-to-code"
CONTAINER_NAME="docstring-to-code-run"

echo "============================================================"
echo "Docstring-to-Code Generation Pipeline"
echo "============================================================"
echo ""

# Build Docker image
echo "[Step 1/2] Building Docker image..."
docker build -t ${IMAGE_NAME} .
echo "Docker image built successfully."
echo ""

# Check for GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running with GPU support."
    GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU detected. Running on CPU (training will be slower)."
fi

# Remove existing container if present
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Run the container
echo ""
echo "[Step 2/2] Running training and evaluation pipeline..."
echo ""

docker run \
    ${GPU_FLAG} \
    --name ${CONTAINER_NAME} \
    -v "$(pwd)/Checkpoints:/app/Checkpoints" \
    -v "$(pwd)/Attention_Visualizations:/app/Attention_Visualizations" \
    -v "$(pwd)/Evaluation:/app/Evaluation" \
    -v "$(pwd)/Notebook:/app/Notebook" \
    ${IMAGE_NAME} \
    python main.py "$@"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo ""
echo "Outputs:"
echo "  Checkpoints/                  - Model checkpoints"
echo "  Evaluation/                   - Metrics, plots, results.json"
echo "  Attention_Visualizations/     - Attention heatmaps"
echo "============================================================"
