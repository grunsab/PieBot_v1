# Multi-GPU Chess Playing Guide

This guide explains how to use multi-GPU support for playing chess with AlphaZero, which can significantly speed up MCTS (Monte Carlo Tree Search) by parallelizing neural network evaluations across multiple GPUs.

## Overview

Two implementations are available:

1. **playchess.py** - Updated with basic multi-GPU support (loads model on multiple GPUs)
2. **playchess_multigpu.py** - Full multi-GPU implementation with parallel MCTS

## Basic Multi-GPU Support (playchess.py)

The original playchess.py now supports loading models on multiple GPUs:

```bash
# Use specific GPUs
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --gpus 0,1,2,3 --rollouts 1000 --threads 10

# Use GPU 0 and 2
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --gpus 0,2 --rollouts 1000
```

### Command Line Options

- `--gpus`: Comma-separated list of GPU IDs (e.g., "0,1,2,3")
- If not specified, defaults to single GPU (GPU 0) for compatibility

## Advanced Multi-GPU Support (playchess_multigpu.py)

For maximum performance, use the dedicated multi-GPU implementation:

```bash
# Use all available GPUs
python3 playchess_multigpu.py --model weights/AlphaZeroNet_20x256.pt --rollouts 10000

# Use specific number of GPUs
python3 playchess_multigpu.py --model weights/AlphaZeroNet_20x256.pt --gpus 4 --rollouts 10000

# With custom batch size per GPU
python3 playchess_multigpu.py --model weights/AlphaZeroNet_20x256.pt --gpus 8 --batch-size 64
```

### Features

1. **Parallel Neural Network Evaluation**: Distributes position evaluations across GPUs
2. **Batch Processing**: Groups positions for efficient GPU utilization
3. **Load Balancing**: Automatically distributes work to least loaded GPU
4. **Thread Pool**: Manages concurrent MCTS rollouts efficiently

### Command Line Options

- `--gpus`: Number of GPUs to use (default: all available)
- `--batch-size`: Batch size per GPU for neural network evaluation (default: 32)
- `--rollouts`: Total number of MCTS rollouts
- `--verbose`: Show detailed search statistics

## Performance Comparison

### Single GPU
```bash
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --rollouts 1000
# ~100-200 rollouts/second
```

### Multi-GPU (8 GPUs)
```bash
python3 playchess_multigpu.py --model weights/AlphaZeroNet_20x256.pt --gpus 8 --rollouts 10000
# ~500-1000 rollouts/second (5-10x speedup)
```

## Usage Examples

### Playing Against the AI

```bash
# Strong play with multi-GPU
python3 playchess_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --mode h \
    --rollouts 10000 \
    --gpus 4 \
    --verbose

# Ultra-strong play with all GPUs
python3 playchess_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --mode h \
    --rollouts 50000 \
    --verbose
```

### Self-Play Games

```bash
# Watch AI vs AI with multi-GPU
python3 playchess_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --mode s \
    --rollouts 5000 \
    --gpus 8
```

### Performance Profiling

```bash
# Profile with different GPU configurations
python3 playchess_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --mode p \
    --rollouts 1000 \
    --gpus 1

python3 playchess_multigpu.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --mode p \
    --rollouts 1000 \
    --gpus 8
```

## Technical Details

### Multi-GPU MCTS Architecture

1. **Root Node Creation**: Initial position evaluated on GPU 0
2. **Parallel Selection**: Multiple threads select paths through the tree
3. **Batch Evaluation**: Positions queued and evaluated in batches on GPUs
4. **Load Balancing**: Work distributed to GPU with smallest queue
5. **Backpropagation**: Results propagated back through tree (thread-safe)

### Memory Considerations

- Each GPU loads a full copy of the model (~100-500MB)
- MCTS tree is shared in CPU memory
- Batch size affects GPU memory usage

### Optimization Tips

1. **Batch Size**: Larger batches (32-64) improve GPU utilization
2. **Rollouts**: More rollouts = stronger play, but diminishing returns
3. **Thread Count**: Not used in multi-GPU mode (parallelism via GPUs)

## Troubleshooting

### GPU Not Detected
```bash
# Check available GPUs
python3 -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
nvidia-smi
```

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Use fewer GPUs: `--gpus 4`

### Performance Issues
- Ensure GPUs are not being used by other processes
- Check PCIe bandwidth (multi-GPU requires good interconnect)
- Monitor with `nvidia-smi dmon`

## Recommended Settings

### For Playing
- **Casual**: 1000 rollouts, 1-2 GPUs
- **Strong**: 5000 rollouts, 4 GPUs  
- **Strongest**: 10000+ rollouts, 8 GPUs

### For Analysis
- Use maximum GPUs available
- Increase rollouts for critical positions
- Enable verbose mode to see search statistics