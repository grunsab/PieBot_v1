# Titan Mini Curriculum Training Guide

## Overview

Curriculum training for Titan Mini addresses the high value loss problem observed in transformer-based chess models. The transformer architecture requires careful progressive training to learn fundamental concepts before advancing to sophisticated positional play.

## Why Curriculum Training for Titan Mini?

Transformer models like Titan Mini face unique challenges:
- **Attention mechanisms** need to learn piece relationships gradually
- **Positional encodings** require exposure to diverse position types
- **Value estimation** benefits from progressive difficulty
- **Deep architecture** (13 layers) needs structured learning progression

## Architecture-Specific Adaptations

### Titan Mini Architecture
- **13 transformer layers** with multi-head attention
- **512 model dimension** with 8 attention heads
- **1920 feedforward dimension**
- **Enhanced encoder** with 112 input planes (vs 16 classic)
- **Relative position bias** for spatial awareness

### Optimized 4-Stage Curriculum

1. **Beginner (750-1500 ELO)** - 20 epochs
   - Focus: Basic piece values and simple tactics
   - Value weight: 2.5 (strong value emphasis)
   - LR multiplier: 0.5 (stable learning)
   - Temperature: 0.20 (smooth targets)

2. **Intermediate (1500-2400 ELO)** - 30 epochs
   - Focus: Positional patterns and strategy
   - Value weight: 1.8
   - LR multiplier: 1.0
   - Temperature: 0.15

3. **Expert (2400-3000 ELO)** - 40 epochs
   - Focus: Advanced strategies and endgames
   - Value weight: 1.2
   - LR multiplier: 0.8
   - Temperature: 0.12

4. **Computer (3000-4000 ELO)** - 60 epochs
   - Focus: Engine-level evaluation
   - Value weight: 0.8
   - LR multiplier: 0.5
   - Temperature: 0.08

## Setup Instructions

### Prerequisites
```bash
# Install required packages
pip install torch tensorboard chess python-chess numpy tqdm zstandard

# Verify CUDA availability (recommended for Titan Mini)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 1. Data Preparation

#### Download Training Data
```bash
# Download 2 months of Lichess games (more data for transformers)
python download_lichess_games.py \
    --months 2 \
    --min-rating 750 \
    --output-dir games_training_data/reformatted_lichess

# Download computer engine games
python download_computerchess_org_uk.py
```

#### Organize by ELO Rating
```bash
# Sort games into curriculum stages
python organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum \
    --workers 8
```

### 2. Training Commands

#### Basic Curriculum Training
```bash
python train_titan_mini.py \
    --mode curriculum \
    --num-layers 13 \
    --d-model 512 \
    --num-heads 8 \
    --d-ff 1920 \
    --input-planes 112 \
    --lr 0.0001 \
    --batch-size 128 \
    --dynamic-value-weight \
    --monitor-piece-values
```

#### Advanced Training with Optimizations
```bash
python train_titan_mini.py \
    --mode curriculum \
    --num-layers 13 \
    --d-model 512 \
    --num-heads 8 \
    --d-ff 1920 \
    --input-planes 112 \
    --lr 0.0001 \
    --warmup-epochs 5 \
    --batch-size 128 \
    --gradient-accumulation 2 \
    --grad-clip 10.0 \
    --mixed-precision \
    --compile \
    --dynamic-value-weight \
    --monitor-piece-values \
    --save-every 10
```

#### Distributed Training (Multi-GPU)
```bash
torchrun --nproc_per_node=4 train_titan_mini.py \
    --mode curriculum \
    --distributed \
    --batch-size-total 512 \
    --num-layers 13 \
    --d-model 512 \
    --input-planes 112 \
    --mixed-precision \
    --dynamic-value-weight
```

#### Resume Training
```bash
python train_titan_mini.py \
    --mode curriculum \
    --resume checkpoints/titan_mini_curriculum/checkpoint_epoch_50.pt \
    --curriculum-state checkpoints/titan_mini_curriculum/curriculum_state_50.json
```

### 3. Mixed Curriculum Mode

Prevent catastrophic forgetting with mixed sampling:
```bash
python train_titan_mini.py \
    --mode mixed-curriculum \
    --num-layers 13 \
    --d-model 512 \
    --input-planes 112 \
    --epochs 200
```

Default mixing ratios (optimized for Titan Mini):
- Beginner: 10%
- Intermediate: 25%
- Expert: 35%
- Computer: 30%

## Custom Configuration

Create `titan_curriculum_config.json`:
```json
{
  "stages": [
    {
      "name": "beginner",
      "data_dir": "games_training_data/curriculum/beginner",
      "elo_range": [750, 1500],
      "epochs": 25,
      "value_weight": 3.0,
      "lr_multiplier": 0.3,
      "temperature": 0.25,
      "enhanced_encoder": true
    },
    {
      "name": "intermediate",
      "data_dir": "games_training_data/curriculum/intermediate",
      "elo_range": [1500, 2400],
      "epochs": 35,
      "value_weight": 2.0,
      "lr_multiplier": 0.8,
      "temperature": 0.18,
      "enhanced_encoder": true
    },
    {
      "name": "expert",
      "data_dir": "games_training_data/curriculum/expert",
      "elo_range": [2400, 3000],
      "epochs": 45,
      "value_weight": 1.5,
      "lr_multiplier": 0.6,
      "temperature": 0.12,
      "enhanced_encoder": true
    },
    {
      "name": "computer",
      "data_dir": "games_training_data/curriculum/computer",
      "elo_range": [3000, 4000],
      "epochs": 70,
      "value_weight": 0.7,
      "lr_multiplier": 0.4,
      "temperature": 0.08,
      "enhanced_encoder": true
    }
  ]
}
```

Use custom config:
```bash
python train_titan_mini.py \
    --mode curriculum \
    --curriculum-config titan_curriculum_config.json
```

## Monitoring Progress

### TensorBoard Metrics
```bash
tensorboard --logdir logs/titan_mini_curriculum
```

Track:
- **Loss curves** per stage
- **Value loss convergence**
- **Piece value evolution** (Titan-specific)
- **Attention patterns** (if logged)
- **Learning rate scheduling**

### Piece Value Monitoring

Monitor how Titan Mini learns piece values:
```python
from titan_piece_value_monitor import TitanPieceValueMonitor
from TitanMiniNetwork import TitanMini

# Load model
model = TitanMini(num_layers=13, d_model=512, num_heads=8, d_ff=1920)
model.load_state_dict(torch.load('TitanMini_curriculum.pt'))

# Analyze piece values
monitor = TitanPieceValueMonitor(model, device='cuda', enhanced_encoder=True)
monitor.print_detailed_report()
```

Expected progression:
- **Stage 1**: Learn basic material values
- **Stage 2**: Refine with positional context
- **Stage 3**: Sophisticated piece coordination
- **Stage 4**: Engine-level dynamic evaluation

## Titan Mini-Specific Tips

### Hyperparameter Guidelines

1. **Learning Rate**
   - Start low: 0.0001 (transformers are sensitive)
   - Use warmup: 5 epochs minimum
   - Stage-specific multipliers: 0.5 → 1.0 → 0.8 → 0.5

2. **Batch Size**
   - Recommended: 128-256 per GPU
   - Use gradient accumulation if memory limited
   - Larger batches help transformer stability

3. **Gradient Clipping**
   - Essential for transformers: 10.0 recommended
   - Prevents attention explosion in early training

4. **Dropout**
   - Keep at 0.1 for curriculum training
   - Can reduce to 0.05 in final stages

### Performance Optimizations

1. **Mixed Precision Training**
   ```bash
   --mixed-precision  # FP16 training, ~2x speedup
   ```

2. **Torch Compile** (PyTorch 2.0+)
   ```bash
   --compile  # Graph optimization, 10-30% speedup
   ```

3. **Gradient Accumulation**
   ```bash
   --gradient-accumulation 2  # Simulate larger batches
   ```

4. **Distributed Training**
   ```bash
   torchrun --nproc_per_node=GPU_COUNT train_titan_mini.py --distributed
   ```

## Common Issues & Solutions

### High Value Loss Persists
- **Increase beginner epochs**: 20 → 30
- **Raise value weight**: 2.5 → 3.5
- **Lower learning rate**: 0.0001 → 0.00005
- **Check enhanced encoder**: Ensure 112 planes active

### Attention Collapse
- **Reduce learning rate**: Critical for transformers
- **Increase gradient clipping**: 10.0 → 20.0
- **Add more warmup**: 5 → 10 epochs
- **Check layer norm**: Ensure proper initialization

### Memory Issues
- **Reduce batch size**: 128 → 64
- **Use gradient accumulation**: Maintain effective batch
- **Enable mixed precision**: Significant memory savings
- **Reduce sequence length**: If using variable lengths

### Slow Convergence
- **Verify enhanced encoder**: 112 planes provide richer features
- **Increase data**: Transformers benefit from more examples
- **Adjust temperature**: Higher for early stages (0.20-0.25)
- **Check positional encoding**: Critical for spatial awareness

## Evaluation

### Test Trained Model
```bash
# Play against Titan Mini
python playchess.py \
    --model TitanMini_curriculum.pt \
    --rollouts 800 \
    --threads 8 \
    --mode h

# Compare curriculum vs standard training
python compare_models.py \
    --model1 TitanMini_curriculum.pt \
    --model2 TitanMini_standard.pt \
    --games 100
```

### Performance Benchmarks

Expected improvements with curriculum training:
- **Value loss**: 40-60% reduction
- **Piece value accuracy**: >90% convergence score
- **Playing strength**: +100-200 ELO
- **Tactical accuracy**: 15-25% improvement
- **Endgame performance**: Significant improvement

## Theory & Research

### Transformer-Specific Benefits

1. **Attention Learning**: Progressive difficulty helps attention heads specialize
2. **Position Understanding**: Gradual complexity aids positional encoding
3. **Feature Hierarchy**: Natural emergence from simple to complex patterns
4. **Stability**: Reduces training instabilities common in transformers

### Comparison with CNN Models

| Aspect | Titan Mini (Transformer) | AlphaZero (CNN) |
|--------|-------------------------|-----------------|
| Data Efficiency | Lower (needs more) | Higher |
| Curriculum Benefit | Very High | High |
| Training Stability | Requires care | More stable |
| Final Performance | Potentially higher | Strong baseline |
| Interpretability | Better (attention) | Limited |

## Future Enhancements

1. **Dynamic Curriculum**: Adjust stages based on loss convergence
2. **Attention Analysis**: Visualize what Titan Mini learns at each stage
3. **Position-Specific Training**: Separate curricula for opening/middle/endgame
4. **Self-Play Integration**: Add RL fine-tuning after curriculum
5. **Cross-Architecture Transfer**: Use curriculum knowledge for larger models

## Quick Start Script

Use the provided script for automated setup:
```bash
./titan_mini_curriculum_training.sh
```

This handles:
- Data download (2 months Lichess + computer games)
- ELO-based organization
- Full curriculum training with optimal settings
- Checkpoint saving and logging

## Support & Troubleshooting

For issues or questions:
1. Check TensorBoard logs for training curves
2. Verify data organization with `ls -la games_training_data/curriculum/`
3. Monitor GPU usage with `nvidia-smi`
4. Test with smaller model first (reduce layers/d_model)
5. Ensure PyTorch version ≥ 1.12 for stability

## Conclusion

Curriculum training is particularly beneficial for Titan Mini's transformer architecture, addressing the unique challenges of attention-based learning in chess. The progressive approach ensures robust piece value learning while building towards sophisticated positional understanding.