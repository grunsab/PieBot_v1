#!/bin/bash
# Example script for curriculum training workflow

echo "==================================="
echo "Curriculum Training Pipeline"
echo "==================================="

# Configuration
MONTHS_TO_DOWNLOAD=1  # Download 1 month (30GB compressed, ~90M games)
MIN_RATING=800  # Minimum rating for quality games
MAX_GAMES=10000000  # Collect up to 10M games (reasonable for curriculum)

# Step 1: Download Lichess games (human players)
echo "Step 1: Downloading Lichess games..."
echo "  Months: ${MONTHS_TO_DOWNLOAD}"
echo "  Min rating: ${MIN_RATING}"
echo "  Max games: ${MAX_GAMES}"
echo "  Will delete compressed files after processing to save space"
echo ""

python download_lichess_games.py \
    --months ${MONTHS_TO_DOWNLOAD} \
    --min-rating ${MIN_RATING} \
    --max-games ${MAX_GAMES} \
    --delete-after-processing \
    --output-dir games_training_data/reformatted_lichess \
    --output-dir-downloads games_training_data/LiChessData/

# Step 2: Download computer games (optional, for highest level)
echo "Step 2: Downloading computer games..."
python download_computerchess_org_uk.py

# Step 3: Organize games by ELO rating
echo ""
echo "Step 3: Organizing games by ELO rating..."
echo "  Creating curriculum stages: beginner (750-1500), intermediate (1500-2400),"
echo "  expert (2400-3000), computer (3000-4000)"
echo ""

python organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum

# Also organize computer games if available
if [ -d "games_training_data/reformatted" ]; then
    echo "Organizing computer games..."
    python organize_games_by_elo.py \
        --input-dir games_training_data/reformatted \
        --output-dir games_training_data/curriculum
fi

# Display storage usage
echo ""
echo "Checking storage usage..."
if [ -d "games_training_data/curriculum" ]; then
    CURRICULUM_SIZE=$(du -sh games_training_data/curriculum 2>/dev/null | cut -f1)
    echo "  Curriculum data size: ${CURRICULUM_SIZE}"
fi
if [ -d "games_training_data/reformatted_lichess" ]; then
    LICHESS_SIZE=$(du -sh games_training_data/reformatted_lichess 2>/dev/null | cut -f1)
    echo "  Lichess filtered games: ${LICHESS_SIZE}"
fi

# Step 4: Start curriculum training
echo ""
echo "Step 4: Starting curriculum training..."
echo "  Model: AlphaZero 20x256"
echo "  Training mode: Progressive difficulty curriculum"
echo "  Stages: 4 (beginner → intermediate → expert → computer)"
echo ""

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
    --curriculum-state checkpoints/curriculum/curriculum_state.json \
    --verbose

echo ""
echo "==================================="
echo "Training complete!"
echo "Model saved to: AlphaZeroNet_20x256_curriculum.pt"
echo "Logs available at: logs/curriculum/"
echo "Checkpoints at: checkpoints/curriculum/"
echo "==================================="
echo ""

# Storage cleanup suggestion
echo "Storage Management Tips:"
echo "  - After training, you can delete games_training_data/reformatted_lichess/ to free space"
echo "  - Keep games_training_data/curriculum/ for future training runs"
echo "  - Compressed files in LiChessData/ should already be deleted"
echo ""

# Optional: Test the trained model
echo "To test the trained model, run:"
echo "python playchess.py --model AlphaZeroNet_20x256_curriculum.pt --rollouts 1000 --threads 10 --mode h"
echo ""
echo "To resume interrupted training, run this script again."
echo "Training will automatically continue from the saved curriculum state."