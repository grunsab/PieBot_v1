#!/bin/bash
# Example script for curriculum training workflow

echo "==================================="
echo "Curriculum Training Pipeline"
echo "==================================="

# Step 1: Download Lichess games (human players)
echo "Step 1: Downloading Lichess games..."
python download_lichess_games.py \
    --months 1 \
    --min-rating 750 \
    --output-dir games_training_data/reformatted_lichess \
    --output-dir-downloads games_training_data/LiChessData/

# Step 2: Download computer games (optional, for highest level)
echo "Step 2: Downloading computer games..."
python download_computerchess_org_uk.py

# Step 3: Organize games by ELO rating
echo "Step 3: Organizing games by ELO rating..."
python organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum \
    --workers 12

# Also organize computer games if available
if [ -d "games_training_data/reformatted" ]; then
    echo "Organizing computer games..."
    python organize_games_by_elo.py \
        --input-dir games_training_data/reformatted \
        --output-dir games_training_data/curriculum \
        --workers 12
fi

# Step 4: Start curriculum training
echo "Step 4: Starting curriculum training..."
python train.py \
    --mode curriculum \
    --num-blocks 20 \
    --num-filters 256 \
    --batch-size 256 \
    --lr 0.001 \
    --scheduler plateau \
    --dynamic-value-weight \
    --monitor-piece-values \
    --save-every 5 \
    --log-dir logs/curriculum \
    --checkpoint-dir checkpoints/curriculum \
    --verbose

echo "==================================="
echo "Training complete!"
echo "Model saved to: AlphaZeroNet_20x256_curriculum.pt"
echo "Logs available at: logs/curriculum/"
echo "==================================="

# Optional: Test the trained model
echo "To test the trained model, run:"
echo "python playchess.py --model AlphaZeroNet_20x256_curriculum.pt --rollouts 1000 --threads 10 --mode h"