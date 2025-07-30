# Multi-GPU Training Guide for AlphaZero

This guide explains how to use multiple GPUs to accelerate training game generation in the AlphaZero implementation.

## Overview

The multi-GPU training system allows you to generate self-play games in parallel across multiple GPUs, significantly reducing the time needed for data generation. Each GPU runs independently, generating its own set of games with unique file names.

## Quick Start

### 1. Basic Multi-GPU Training (3 GPUs)

To use all 3 GPUs for training with curriculum learning:

```bash
python3 train_curriculum.py \
    --selfplay-gpus 3 \
    --games-per-iter 30000 \
    --rollouts 40 \
    --threads 20
```

This will:
- Generate 30,000 games per iteration
- Distribute equally across 3 GPUs (10,000 games each)
- Use 40 MCTS rollouts per move
- Use 20 threads per GPU

### 2. With CUDA-Optimized MCTS

First, build the CUDA extensions:

```bash
# For Linux:
./build_linux.sh

# For Windows:
build_windows.bat
```

Then run with CUDA optimization:

```bash
python3 train_curriculum.py \
    --selfplay-gpus 3 \
    --games-per-iter 30000 \
    --rollouts 100 \
    --threads 32 \
    --use-cuda-mcts
```

### 3. Specific GPU Selection

To use specific GPUs (e.g., GPUs 0, 1, and 2):

```bash
python3 train_curriculum.py \
    --selfplay-gpus 3 \
    --selfplay-gpu-ids "0,1,2" \
    --games-per-iter 30000 \
    --use-cuda-mcts
```

## Standalone Multi-GPU Game Generation

You can also run game generation independently:

```bash
python3 create_training_games_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --games-total 30000 \
    --rollouts 100 \
    --threads-per-gpu 32 \
    --save-format h5 \
    --output-dir games_training_data/selfplay/iter_1 \
    --gpus "0,1,2" \
    --use-cuda-mcts
```

## Performance Considerations

### 1. GPU Memory

Each GPU needs enough memory to:
- Hold the neural network model
- Process batches of positions
- Store MCTS trees

Typical memory usage:
- 20×256 model: ~4-6 GB per GPU
- Adjust batch size if you encounter OOM errors

### 2. Optimal Settings

For RTX 4080/4090 or similar:
```bash
--rollouts 100-200      # Higher rollouts for stronger play
--threads 32-64         # Match CPU cores
--use-cuda-mcts        # Enable CUDA optimization
```

For older GPUs (RTX 3070/3080):
```bash
--rollouts 40-80       # Moderate rollouts
--threads 16-32        # Moderate threading
```

### 3. Heterogeneous GPU Setup

If you have different GPU models, you can weight the distribution:

```bash
python3 create_training_games_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --games-total 30000 \
    --gpus "0,1,2" \
    --gpu-weights "1.0,1.0,0.8" \  # GPU 2 is slower, gets fewer games
    --use-cuda-mcts
```

## File Naming Convention

Games are saved with GPU-specific naming:
- GPU 0: `selfplay_iter1_gpu0_0.h5`, `selfplay_iter1_gpu0_1.h5`, ...
- GPU 1: `selfplay_iter1_gpu1_10000.h5`, `selfplay_iter1_gpu1_10001.h5`, ...
- GPU 2: `selfplay_iter1_gpu2_20000.h5`, `selfplay_iter1_gpu2_20001.h5`, ...

This ensures no file conflicts between GPUs.

## Monitoring Progress

The multi-GPU script provides real-time output from each GPU:

```
[2025-07-30 10:00:00] Found 3 CUDA devices
[2025-07-30 10:00:00]   GPU 0: NVIDIA GeForce RTX 4080 (16.0 GB)
[2025-07-30 10:00:00]   GPU 1: NVIDIA GeForce RTX 4080 (16.0 GB)
[2025-07-30 10:00:00]   GPU 2: NVIDIA GeForce RTX 4080 (16.0 GB)
[2025-07-30 10:00:01] Game distribution plan:
[2025-07-30 10:00:01]   GPU 0: 10000 games (offset 0)
[2025-07-30 10:00:01]   GPU 1: 10000 games (offset 10000)
[2025-07-30 10:00:01]   GPU 2: 10000 games (offset 20000)
[GPU 0] White's turn
[GPU 1] White's turn
[GPU 2] White's turn
...
```

## System Optimization

For best performance on Linux:

```bash
# Run system optimization (requires sudo)
sudo ./optimize_linux_system.sh

# Then run training with optimized launcher
run_alphazero_optimized python3 train_curriculum.py \
    --selfplay-gpus 3 \
    --use-cuda-mcts
```

## Troubleshooting

### 1. CUDA Out of Memory

Reduce batch size or rollouts:
```bash
--rollouts 20
--threads 10
```

### 2. Uneven GPU Utilization

Check with `nvidia-smi`:
```bash
watch -n 1 nvidia-smi
```

Adjust thread count or rollouts per GPU.

### 3. CUDA Extensions Not Found

Rebuild extensions:
```bash
# Clean and rebuild
make clean  # Linux
make

# Or using setup.py
python3 setup_extensions.py clean
python3 setup_extensions.py build
```

### 4. Different Game Generation Speeds

Use GPU weights to balance:
```bash
--gpu-weights "1.2,1.0,0.8"  # Boost GPU 0, reduce GPU 2
```

## Example Training Script

Complete example for 3-GPU training:

```bash
#!/bin/bash
# train_3gpu.sh

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=64

# Run curriculum training
python3 train_curriculum.py \
    --blocks 20 \
    --filters 256 \
    --supervised-epochs 20 \
    --games-per-iter 30000 \
    --rollouts 100 \
    --temperature 1.0 \
    --threads 32 \
    --selfplay-gpus 3 \
    --use-cuda-mcts \
    --rl-epochs 20 \
    --iterations 80 \
    --distributed \
    --gpus 3 \
    --output-dir curriculum_training_3gpu
```

## Expected Performance

With 3x RTX 4080 GPUs and CUDA optimization:
- Single GPU: ~1000-1500 games/hour
- 3 GPUs (parallel): ~3000-4500 games/hour
- Speedup: ~3x for game generation phase

Training iteration time (30k games):
- Single GPU: ~20-30 hours
- 3 GPUs: ~7-10 hours

## Integration with Training Pipeline

The multi-GPU game generation is fully integrated with `train_curriculum.py`:

1. **Phase 1**: Supervised learning (uses training GPUs)
2. **Phase 2**: Multi-GPU game generation (uses selfplay GPUs)
3. **Phase 3**: Reinforcement learning on generated games (uses training GPUs)

The system automatically handles:
- Parallel game generation across GPUs
- Unique file naming per GPU
- Progress monitoring
- Error handling and recovery

## Advanced Usage

### Custom MCTS Implementation

To use a custom MCTS implementation:

1. Name it `MCTS_custom.py`
2. Modify `create_training_games.py` to import it
3. Run with custom flag

### Distributed Training Across Machines

For multi-machine setup, run on each machine:

```bash
# Machine 1 (GPUs 0,1,2)
python3 create_training_games_multigpu.py \
    --model model.pt \
    --games-total 10000 \
    --file-base selfplay_machine1 \
    --gpus "0,1,2"

# Machine 2 (GPUs 0,1,2)  
python3 create_training_games_multigpu.py \
    --model model.pt \
    --games-total 10000 \
    --file-base selfplay_machine2 \
    --gpus "0,1,2"
```

Then combine the generated files for training.