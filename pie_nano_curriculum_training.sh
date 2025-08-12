#!/bin/bash
# Comprehensive curriculum training pipeline for PieNano V2 model
# Optimized for ~8M parameter architecture (20x256 with 768-dim policy head)

echo "=========================================="
echo "PieNano V2 Curriculum Training Pipeline"
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

# Step 4: Start PieNano V2 curriculum training
echo ""
echo "Step 4: Starting PieNano V2 curriculum training..."
echo "Model configuration: 20 blocks, 256 filters, 768-dim policy head"
echo "Estimated parameters: ~8.2M"
echo ""

# Create directories for checkpoints and logs
mkdir -p weights
mkdir -p logs/pie_nano_curriculum
mkdir -p checkpoints/pie_nano_curriculum

# Run curriculum training
python3 train_pie_nano.py \
    --mode curriculum \
    --num-blocks 20 \
    --num-filters 256 \
    --policy-hidden-dim 768 \
    --batch-size 1024 \
    --lr 0.001 \
    --warmup-epochs 3 \
    --gradient-accumulation 2 \
    --clip-grad-norm 1.0 \
    --weight-decay 0.0001 \
    --label-smoothing 0.05 \
    --dropout 0.1 \
    --scheduler cosine \
    --mixed-precision \
    --save-every 1 \
    --validate-every 1 \
    --curriculum-state checkpoints/pie_nano_curriculum/curriculum_state.json \
    --output weights/PieNanoV2_20x256_curriculum.pt

echo ""
echo "=========================================="
echo "Curriculum training complete!"
echo "Model saved to: weights/PieNanoV2_20x256_curriculum.pt"
echo "Best model at: weights/PieNanoV2_20x256_curriculum_best.pt"
echo "Checkpoints at: checkpoints/pie_nano_curriculum/"
echo "=========================================="
echo ""

# Optional: Test the trained model
echo "To test the trained PieNano V2 model, run:"
echo "python3 playchess.py --model weights/PieNanoV2_20x256_curriculum.pt --rollouts 500 --threads 8 --mode h"
echo ""
echo "For enhanced encoder (112 planes) training, add:"
echo "--use-enhanced-encoder flag to the training command"
echo ""
echo "To resume interrupted training, run the same command again."
echo "The training will automatically resume from the saved state."