# Curriculum Training for AlphaZero Chess Network

## Overview

Curriculum training addresses the high value loss problem observed when training exclusively on high-level games (2850+ ELO). By progressively training on games from beginner to expert level, the network learns fundamental concepts like piece values before advancing to sophisticated positional play.

## Motivation

Training exclusively on computer vs computer games (2850+ ELO) leads to:
- High value loss that doesn't decrease
- Poor understanding of basic piece values
- Missing fundamental chess concepts that emerge in simpler positions

The curriculum approach mimics how AlphaZero naturally progresses during self-play:
1. Early: Focus on material (piece counting)
2. Middle: Tactical patterns emerge
3. Late: Sophisticated positional understanding

## Architecture

### 4-Stage Curriculum

1. **Beginner (750-1500 ELO)** - 10 epochs
   - Focus: Learning piece values and basic tactics
   - Value weight: 2.0 (emphasize value learning)
   - Source: Human games from Lichess

2. **Intermediate (1500-2400 ELO)** - 20 epochs
   - Focus: Positional understanding and strategy
   - Value weight: 1.5
   - Source: Stronger human games

3. **Expert (2400-3000 ELO)** - 30 epochs
   - Focus: Advanced strategy and endgames
   - Value weight: 1.0
   - Source: Titled players and strong humans

4. **Computer (3000-4000 ELO)** - 50 epochs
   - Focus: Deep strategic refinement
   - Value weight: 0.8 (emphasize policy)
   - Source: Computer chess engines

## Setup Instructions

### 1. Download and Prepare Data

#### Download Lichess Games (Human Players)
```bash
# Download 1 month of Lichess games (beginner to expert)
python download_lichess_games.py --months 1 --min-rating 750 --output-dir games_training_data/reformatted_lichess

# Download computer games from CCRL/Computer Chess
python download_computerchess_org_uk.py
```

#### Organize Games by ELO
```bash
# Sort games into curriculum stages
python organize_games_by_elo.py \
    --input-dir games_training_data/reformatted_lichess \
    --output-dir games_training_data/curriculum

# This creates:
# games_training_data/curriculum/
#   ├── beginner/     (750-1500 ELO)
#   ├── intermediate/ (1500-2400 ELO)
#   ├── expert/       (2400-3000 ELO)
#   └── computer/     (3000-4000 ELO)
```

### 2. Start Curriculum Training

#### Basic Curriculum Training
```bash
python train.py \
    --mode curriculum \
    --num-blocks 20 \
    --num-filters 256 \
    --epochs 110 \
    --dynamic-value-weight
```

#### Resume Training
```bash
python train.py \
    --mode curriculum \
    --resume weights/checkpoint.pt \
    --curriculum-state checkpoints/curriculum_state.json
```

#### Custom Configuration
Create a JSON configuration file:
```json
{
  "stages": [
    {
      "name": "beginner",
      "data_dir": "games_training_data/curriculum/beginner",
      "elo_range": [750, 1500],
      "epochs": 15,
      "value_weight": 2.5,
      "soft_targets": true,
      "temperature": 0.15
    },
    {
      "name": "intermediate",
      "data_dir": "games_training_data/curriculum/intermediate",
      "elo_range": [1500, 2400],
      "epochs": 25,
      "value_weight": 1.5,
      "soft_targets": true,
      "temperature": 0.12
    }
  ]
}
```

Then train with:
```bash
python train.py \
    --mode curriculum \
    --curriculum-config my_curriculum.json
```

### 3. Mixed Curriculum Training

To prevent catastrophic forgetting, use mixed curriculum mode:
```bash
python train.py \
    --mode mixed-curriculum \
    --epochs 100
```

This samples from all stages simultaneously with default ratios:
- Beginner: 15%
- Intermediate: 25%
- Expert: 30%
- Computer: 30%

## Command Line Options

### Curriculum-Specific Arguments

- `--mode curriculum`: Enable curriculum training
- `--curriculum-dir`: Directory with organized games (default: `games_training_data/curriculum`)
- `--curriculum-config`: Path to custom curriculum JSON
- `--curriculum-state`: Resume from saved curriculum state
- `--dynamic-value-weight`: Adjust value/policy weights per stage

### Training Arguments

- `--num-blocks`: Residual blocks (10 or 20)
- `--num-filters`: Filters per layer (128 or 256)
- `--batch-size`: Batch size (auto-detected if not specified)
- `--lr`: Learning rate (default: 0.001)
- `--scheduler`: LR scheduler (onecycle, cosine, plateau, none)
- `--mixed-precision`: Use FP16 training for faster speed

## Monitoring Progress

### TensorBoard Metrics
```bash
tensorboard --logdir logs/
```

Track:
- Loss curves per stage
- Value loss convergence
- Piece value evolution
- ELO range progression

### Piece Value Monitoring

The system tracks how the network learns piece values:
```python
from piece_value_monitor import PieceValueMonitor

monitor = PieceValueMonitor(model, device)
monitor.print_piece_value_report()
```

Expected progression:
1. **Beginner stage**: Learn basic piece values (Q=9, R=5, B/N=3, P=1)
2. **Intermediate**: Refine values based on position
3. **Expert/Computer**: Context-dependent piece evaluation

## Tips and Best Practices

### Data Requirements
- Minimum 100k games per stage for good results
- More games in beginner/intermediate stages help value learning
- Balance dataset sizes across stages

### Hyperparameter Tuning
- **Value weight**: Start high (2.0-2.5) for beginners, decrease gradually
- **Temperature**: Higher for beginners (0.15), lower for experts (0.08)
- **Learning rate**: Consider warm restarts between stages
- **Batch size**: Can increase for later stages with cleaner data

### Common Issues

1. **High value loss persists**
   - Increase beginner stage epochs
   - Raise value weight for early stages
   - Check data quality (games need results)

2. **Performance regression between stages**
   - Use mixed-curriculum mode
   - Reduce learning rate between stages
   - Add warmup epochs per stage

3. **Out of memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

## Evaluation

Test the trained model:
```bash
# Play against the AI
python playchess.py \
    --model AlphaZeroNet_20x256_curriculum.pt \
    --rollouts 1000 \
    --threads 10 \
    --mode h
```

Compare against baseline:
```bash
# Curriculum-trained model
python playchess.py --model weights/curriculum_model.pt --mode h

# Standard-trained model  
python playchess.py --model weights/standard_model.pt --mode h
```

## Theory and Research

The curriculum approach combines benefits of:
- **Reinforcement Learning**: Natural progression from simple to complex
- **Supervised Learning**: Computational efficiency
- **Transfer Learning**: Knowledge builds across stages

Key insights:
- AlphaZero's self-play creates implicit curriculum
- Human learning follows similar progression
- Piece values must be learned before positional concepts
- Mixing difficulties prevents catastrophic forgetting

## Future Improvements

Potential enhancements:
1. Adaptive stage switching based on loss convergence
2. Dynamic mixing ratios based on performance
3. Position-specific curriculum (openings, middlegame, endgame)
4. Automatic ELO range optimization
5. Integration with self-play for final refinement