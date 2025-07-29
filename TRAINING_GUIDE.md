# AlphaZero Training Guide

This guide explains how to properly train the AlphaZero chess engine using the hybrid supervised/reinforcement learning approach.

## Overview

The training process consists of two distinct phases:
1. **Supervised Learning**: Bootstrap the network using high-quality human games
2. **Reinforcement Learning**: Improve through self-play

**Important**: These phases should be run sequentially, not mixed, to avoid training instability.

## Phase 1: Supervised Learning

First, train on the CCRL dataset to create a strong initial policy:

```bash
# Standard supervised training
python train.py --mode supervised --epochs 500 --lr 0.001

# Distributed training (multi-GPU)
python -m torch.distributed.launch --nproc_per_node=4 train_distributed.py \
    --mode supervised --epochs 500
```

This phase typically takes 5-7 days on a modern GPU and produces a model playing at 2700-2900 ELO.

### Key Parameters:
- `--epochs`: 300-500 recommended
- `--policy-weight`: Keep at 1.0 for supervised learning
- `--batch-size`: Auto-detected based on GPU memory

## Phase 2: Self-Play Data Generation

Once supervised training converges, generate self-play games:

```bash
# Generate self-play games with MCTS visit distributions
python create_training_games.py \
    --model AlphaZeroNet_20x256.pt \
    --save-format h5 \
    --rollouts 800 \
    --temperature 1.0 \
    --games-to-play 10000 \
    --threads 20
```

### Key Parameters:
- `--save-format h5`: Saves MCTS visit counts for training
- `--temperature`: Controls exploration (1.0 for training, 0.1 for strong play)
- `--rollouts`: More rollouts = stronger play but slower generation

## Phase 3: Reinforcement Learning

Train on self-play data to improve beyond supervised performance:

```bash
# Continue training with RL
python train.py \
    --mode rl \
    --resume AlphaZeroNet_20x256.pt \
    --rl-dir games_training_data/selfplay \
    --epochs 100 \
    --lr 0.0001
```

### Key Considerations:
- Use lower learning rate (0.0001) to avoid catastrophic forgetting
- Monitor performance regularly through play testing
- Generate new self-play data periodically with the improved model

## Mixed Mode (Advanced)

If you must use mixed training, the datasets now support compatible soft targets:

```bash
# Mixed training with soft targets
python train.py \
    --mode mixed \
    --mixed-ratio 0.3 \
    --label-smoothing-temp 0.1
```

**Warning**: Mixed training can be unstable. The recommended approach is sequential training.

## Training Pipeline

The complete training loop:

1. **Initial Supervised Training** (1 week)
   ```bash
   python train.py --mode supervised --epochs 500
   ```

2. **Generate Self-Play Games** (2-3 days)
   ```bash
   python create_training_games.py --save-format h5 --games-to-play 50000
   ```

3. **Reinforcement Learning** (3-5 days)
   ```bash
   python train.py --mode rl --resume model.pt --epochs 100
   ```

4. **Iterate Steps 2-3** with improved models

## Monitoring Progress

1. **Loss Tracking**: Value loss should decrease steadily
2. **Policy Entropy**: Should remain reasonable (not too low = diverse play)
3. **Play Testing**: Regular games against known engines
4. **ELO Estimation**: Use tournaments or rating lists

## Common Issues

### Policy Collapse
If the policy becomes too deterministic:
- Increase temperature during self-play
- Add Dirichlet noise to root node
- Use more diverse training positions

### Catastrophic Forgetting
If the model forgets supervised knowledge:
- Lower the learning rate
- Use smaller RL ratio in mixed mode
- Keep some supervised data in training

### Training Instability
If losses oscillate wildly:
- Don't mix supervised and RL data
- Check for data corruption
- Verify correct loss calculation

## Best Practices

1. **Save Checkpoints Frequently**: Every 10-20 epochs
2. **Version Control Models**: Track which data generated which model
3. **Test Regularly**: Automated testing against reference engines
4. **Monitor GPU Memory**: Batch size affects training stability
5. **Use Distributed Training**: For faster iteration cycles

## Hardware Recommendations

- **Minimum**: RTX 3070 or better (8GB VRAM)
- **Recommended**: RTX 4090 or A100 (24GB+ VRAM)
- **Optimal**: Multi-GPU setup with NVLink

## Next Steps

After achieving strong RL performance:
1. Implement opening book
2. Add endgame tablebases
3. Tune MCTS parameters
4. Create specialized models (tactical/positional)

Remember: The key to AlphaZero's success is the iterative self-improvement loop. Be patient and let the system learn!