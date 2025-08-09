#!/bin/bash
# Example script for 3-GPU AlphaZero training with CUDA optimization

# Exit on error
set -e

echo "========================================="
echo "AlphaZero 3-GPU Training Example"
echo "========================================="
echo ""

# Check for model file
MODEL_FILE="weights/AlphaZeroNet_20x256.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Please ensure you have a trained model or run supervised learning first."
    exit 1
fi

# Set environment for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
OUTPUT_DIR="curriculum_training_3gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: 3 (IDs: 0,1,2)"
echo "  Games per iteration: 30,000"
echo "  CUDA MCTS: Enabled"
echo ""

# Check if CUDA extensions are built
if [ -f "mcts_cuda.so" ] || [ -f "mcts_cuda.pyd" ]; then
    echo "✓ CUDA extensions found"
else
    echo "! CUDA extensions not found. Building..."
    if [ -f "build_linux.sh" ]; then
        ./build_linux.sh
    else
        python3 setup_extensions.py build
    fi
fi

echo ""
echo "Starting curriculum training..."
echo "This will:"
echo "1. Skip supervised learning (using existing model)"
echo "2. Generate 30,000 games per iteration across 3 GPUs"
echo "3. Train on generated games with reinforcement learning"
echo "4. Repeat for 80 iterations"
echo ""

# Run curriculum training
python3 train_curriculum.py \
    --blocks 20 \
    --filters 256 \
    --skip-supervised \
    --resume-supervised "$MODEL_FILE" \
    --games-per-iter 30000 \
    --rollouts 100 \
    --temperature 1.0 \
    --threads 32 \
    --selfplay-gpus 3 \
    --selfplay-gpu-ids "0,1,2" \
    --use-cuda-mcts \
    --rl-epochs 20 \
    --rl-lr 0.0005 \
    --iterations 80 \
    --distributed \
    --gpus 3 \
    --output-dir "$OUTPUT_DIR" \
    --selfplay-dir "$OUTPUT_DIR/selfplay_data" \
    --weight-decay 0.05

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Results saved in: $OUTPUT_DIR"
echo "Final model: $OUTPUT_DIR/AlphaZeroNet_20x256_rl_iter80.pt"
echo ""
echo "To test the trained model:"
echo "  python3 playchess_cuda.py --model $OUTPUT_DIR/AlphaZeroNet_20x256_rl_iter80.pt --rollouts 1000"