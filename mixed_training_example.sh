#!/bin/bash
# Mixed training script - combines different skill levels to prevent catastrophic forgetting
# Uses MixedCurriculumDataset from CurriculumDataset.py

echo "==================================="
echo "Mixed Training Pipeline"
echo "==================================="

# Configuration
MONTHS_TO_DOWNLOAD=1  # Download 1 month (30GB compressed, ~90M games)
MIN_RATING=750  # Minimum rating for quality games
MAX_GAMES=20000000  # Collect up to 20M games

# Step 1: Download Lichess games (human players)
echo "Step 1: Downloading Lichess games..."
echo "  Months: ${MONTHS_TO_DOWNLOAD}"
echo "  Min rating: ${MIN_RATING}"
echo "  Max games: ${MAX_GAMES}"
echo "  Will delete compressed files after processing to save space"
echo ""

python3 download_lichess_games.py \
    --months ${MONTHS_TO_DOWNLOAD} \
    --min-rating ${MIN_RATING} \
    --max-games ${MAX_GAMES} \
    --delete-after-processing \
    --output-dir games_training_data/reformatted_lichess \
    --output-dir-downloads games_training_data/LiChessData/

# Step 2: Download computer games (for highest level)
echo "Step 2: Downloading computer games..."
python3 download_computerchess_org_uk.py

# Step 3: Organize games by ELO rating
echo ""
echo "Step 3: Organizing games by ELO rating..."
echo "  Creating curriculum stages for mixed training"
echo ""

python3 organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum

# Also organize computer games if available
if [ -d "games_training_data/reformatted" ]; then
    echo "Organizing computer games..."
    python3 organize_games_by_elo.py \
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

# Step 4: Start mixed training
echo ""
echo "Step 4: Starting mixed training..."
echo "  Model: AlphaZero 20x256"
echo "  Training mode: Mixed curriculum (all levels simultaneously)"
echo ""
echo "  Default mixing ratios from MixedCurriculumDataset:"
echo "    - beginner (750-1500):     10% (value_weight: 3.0)"
echo "    - intermediate (1500-2400): 10% (value_weight: 2.5)"
echo "    - expert (2400-3000):       15% (value_weight: 1.8)"
echo "    - computer (3000-4000):     65% (value_weight: 0.7)"
echo ""
echo "  This prevents catastrophic forgetting while focusing on strong play"
echo ""

# Create Python script to run mixed training
cat > run_mixed_training.py << 'EOF'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from AlphaZeroNetwork import AlphaZeroNetwork
from CurriculumDataset import MixedCurriculumDataset
from device_utils import get_device, setup_device_optimizations
import os
import time
from datetime import datetime

# Setup device
device = get_device()
setup_device_optimizations(device)

# Create mixed dataset with default ratios
print("Creating mixed curriculum dataset...")
mixed_dataset = MixedCurriculumDataset()

# Create dataloader
batch_size = 256
dataloader = DataLoader(
    mixed_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=(device.type == 'cuda')
)

# Create model
model = AlphaZeroNetwork(num_res_blocks=20, num_filters=256)
model = model.to(device)

# Setup optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Loss functions
value_loss_fn = nn.MSELoss()
policy_loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 500
save_every = 5
checkpoint_dir = "checkpoints/mixed"
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"\nStarting mixed training for {num_epochs} epochs...")
print(f"Batch size: {batch_size}, Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Checkpoints will be saved to: {checkpoint_dir}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle batch data with value weights
        if len(batch) == 4:
            boards, value_targets, policy_targets, value_weights = batch
            value_weights = value_weights.to(device)
        else:
            boards, value_targets, policy_targets = batch
            value_weights = torch.ones(boards.size(0), device=device)
        
        boards = boards.to(device)
        value_targets = value_targets.to(device)
        policy_targets = policy_targets.to(device)
        
        # Forward pass
        value_pred, policy_pred = model(boards)
        
        # Calculate losses with dynamic value weights
        v_loss = value_loss_fn(value_pred.squeeze(), value_targets)
        p_loss = policy_loss_fn(policy_pred, policy_targets)
        
        # Apply value weights (mean of batch weights)
        value_weight = value_weights.mean().item()
        loss = value_weight * v_loss + p_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_value_loss += v_loss.item()
        total_policy_loss += p_loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, V={v_loss.item():.4f}, "
                  f"P={p_loss.item():.4f}, VW={value_weight:.2f}")
    
    # Calculate epoch metrics
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    epoch_time = time.time() - epoch_start
    
    print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Value Loss: {avg_value_loss:.4f}")
    print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
    
    # Update scheduler
    scheduler.step(avg_loss)
    
    # Save checkpoint
    if (epoch + 1) % save_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"mixed_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

# Save final model
final_model_path = "AlphaZeroNet_20x256_mixed.pt"
torch.save(model.state_dict(), final_model_path)
print(f"\nTraining complete! Final model saved to {final_model_path}")
EOF

# Run the mixed training
python3 run_mixed_training.py

# Clean up temporary script
rm run_mixed_training.py

echo ""
echo "==================================="
echo "Mixed training complete!"
echo "Model saved to: AlphaZeroNet_20x256_mixed.pt"
echo "Checkpoints at: checkpoints/mixed/"
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
echo "python3 playchess.py --model AlphaZeroNet_20x256_mixed.pt --rollouts 1000 --threads 10 --mode h"
echo ""
echo "Mixed training benefits:"
echo "  - Maintains performance across all skill levels"
echo "  - Prevents catastrophic forgetting of basic patterns"
echo "  - Uses dynamic value weights per skill level (3.0 for beginner down to 0.7 for computer)"
echo "  - 65% focus on computer games for maximum strength"
echo ""
echo "The mixing ratios and value weights are defined in CurriculumDataset.py"
echo "and can be customized by modifying MixedCurriculumDataset.__init__()"