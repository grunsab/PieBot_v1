#!/bin/bash
# Mixed curriculum training pipeline for Titan Mini transformer model
# Uses TitanMixedCurriculumDataset to blend different skill levels

echo "=========================================="
echo "Titan Mini Mixed Curriculum Training"
echo "=========================================="
echo ""

# Configuration
MONTHS_TO_DOWNLOAD=1  # Download 1 month (30GB compressed, ~90M games)
MIN_RATING=750  # Minimum rating for quality games
MAX_GAMES=20000000  # Collect up to 20M games (reasonable for curriculum)

# Step 1: Download Lichess games (human players)
echo "Step 1: Downloading Lichess games (human players)..."
echo "This will download ${MONTHS_TO_DOWNLOAD} months of games with ratings >= ${MIN_RATING}"
python3 download_lichess_games.py \
    --months ${MONTHS_TO_DOWNLOAD} \
    --min-rating ${MIN_RATING} \
    --max-games ${MAX_GAMES} \
    --delete-after-processing \
    --output-dir games_training_data/reformatted_lichess \
    --output-dir-downloads games_training_data/LiChessData/

# Step 2: Download computer games (for highest level)
echo ""
echo "Step 2: Downloading computer chess engine games..."
python3 download_computerchess_org_uk.py

# Step 3: Organize games by ELO rating into curriculum stages
echo ""
echo "Step 3: Organizing games by ELO rating..."
echo "Creating 4 stages: beginner (750-1500), intermediate (1500-2400), expert (2400-3000), computer (3000-4000)"

# Organize Lichess games
python3 organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum

# Organize computer games if they exist
if [ -d "games_training_data/reformatted" ]; then
    echo "Organizing computer engine games..."
    python3 organize_games_by_elo.py \
        --input-dir games_training_data/reformatted \
        --output-dir games_training_data/curriculum
fi

# Step 4: Start Titan Mini mixed curriculum training
echo ""
echo "Step 4: Starting Titan Mini mixed curriculum training..."
echo "Model configuration: 13 layers, 512 d_model, 8 heads, 1920 d_ff"
echo "Using enhanced encoder (112 planes) for richer position representation"
echo ""
echo "Mixed training ratios (from TitanMixedCurriculumDataset):"
echo "  - Beginner: 10%"
echo "  - Intermediate: 10%"
echo "  - Expert: 15%"
echo "  - Computer: 65%"
echo ""
echo "Value weights per stage:"
echo "  - Beginner: 3.0"
echo "  - Intermediate: 2.5"
echo "  - Expert: 1.8"
echo "  - Computer: 0.7"
echo ""

python3 train_titan_mini.py \
    --mode mixed \
    --num-layers 13 \
    --d-model 512 \
    --num-heads 8 \
    --d-ff 1920 \
    --dropout 0.1 \
    --input-planes 112 \
    --batch-size 368 \
    --lr 0.0001 \
    --warmup-epochs 5 \
    --gradient-accumulation 2 \
    --grad-clip 10.0 \
    --dynamic-value-weight \
    --monitor-piece-values \
    --mixed-precision \
    --save-every 10 \
    --epochs 300 \
    --log-dir logs/titan_mini_mixed_curriculum \
    --checkpoint-dir checkpoints/titan_mini_mixed_curriculum \
    --output TitanMini_mixed_curriculum.pt

echo ""
echo "=========================================="
echo "Mixed curriculum training complete!"
echo "Model saved to: TitanMini_mixed_curriculum.pt"
echo "Logs available at: logs/titan_mini_mixed_curriculum/"
echo "Checkpoints at: checkpoints/titan_mini_mixed_curriculum/"
echo "=========================================="
echo ""

# Optional: Test the trained model
echo "To test the trained Titan Mini model, run:"
echo "python3 playchess.py --model TitanMini_mixed_curriculum.pt --rollouts 800 --threads 8 --mode h"
echo ""
echo "To monitor training progress:"
echo "tensorboard --logdir logs/titan_mini_mixed_curriculum"
echo ""
echo "Mixed training prevents catastrophic forgetting while maintaining progressive learning"