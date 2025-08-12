#!/bin/bash
# Comprehensive curriculum training pipeline for Titan Mini transformer model

echo "=========================================="
echo "Titan Mini Curriculum Training Pipeline"
echo "=========================================="
echo ""

# Configuration
MONTHS_TO_DOWNLOAD=2  # Download 2 months for more data (Titan Mini benefits from more data)
MIN_RATING=750
MAX_GAMES=8000000

# Step 1: Download Lichess games (human players)
echo "Step 1: Downloading Lichess games (human players)..."
echo "This will download ${MONTHS_TO_DOWNLOAD} months of games with ratings >= ${MIN_RATING}"
python download_lichess_games.py \
    --months ${MONTHS_TO_DOWNLOAD} \
    --min-rating ${MIN_RATING} \
    --output-dir games_training_data/reformatted_lichess \
    --output-dir-downloads games_training_data/LiChessData/

# Step 2: Download computer games (for highest level)
echo ""
echo "Step 2: Downloading computer chess engine games..."
python download_computerchess_org_uk.py

# Step 3: Organize games by ELO rating into curriculum stages
echo ""
echo "Step 3: Organizing games by ELO rating..."
echo "Creating 4 stages: beginner (750-1500), intermediate (1500-2400), expert (2400-3000), computer (3000-4000)"

# Organize Lichess games
python organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum \
    --workers 12

# Organize computer games if they exist
if [ -d "games_training_data/reformatted" ]; then
    echo "Organizing computer engine games..."
    python organize_games_by_elo.py \
        --input-dir games_training_data/reformatted \
        --output-dir games_training_data/curriculum \
        --workers 12
fi

# Step 4: Start Titan Mini curriculum training
echo ""
echo "Step 4: Starting Titan Mini curriculum training..."
echo "Model configuration: 13 layers, 512 d_model, 8 heads, 1920 d_ff"
echo "Using enhanced encoder (112 planes) for richer position representation"
echo ""

python train_titan_mini.py \
    --mode curriculum \
    --num-layers 13 \
    --d-model 512 \
    --num-heads 8 \
    --d-ff 1920 \
    --dropout 0.1 \
    --input-planes 112 \
    --batch-size 128 \
    --lr 0.0001 \
    --warmup-epochs 5 \
    --gradient-accumulation 2 \
    --grad-clip 10.0 \
    --dynamic-value-weight \
    --monitor-piece-values \
    --mixed-precision \
    --save-every 10 \
    --log-dir logs/titan_mini_curriculum \
    --checkpoint-dir checkpoints/titan_mini_curriculum \
    --output TitanMini_curriculum.pt

echo ""
echo "=========================================="
echo "Curriculum training complete!"
echo "Model saved to: TitanMini_curriculum.pt"
echo "Logs available at: logs/titan_mini_curriculum/"
echo "Checkpoints at: checkpoints/titan_mini_curriculum/"
echo "=========================================="
echo ""

# Optional: Test the trained model
echo "To test the trained Titan Mini model, run:"
echo "python playchess.py --model TitanMini_curriculum.pt --rollouts 800 --threads 8 --mode h"
echo ""
echo "To monitor training progress:"
echo "tensorboard --logdir logs/titan_mini_curriculum"