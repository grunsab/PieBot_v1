# Performance Optimization Guide

This document describes the high-performance async implementation that dramatically improves the chess engine's speed from 400-800 nodes/second to potentially 10,000+ nodes/second.

## Overview

The original implementation had several performance bottlenecks:
1. **Synchronous neural network evaluation** - GPU sat idle during tree traversal
2. **Small batch sizes** - Only 8-10 positions evaluated at once
3. **Blocking architecture** - All threads wait for neural network results

The async implementation addresses these issues with:
1. **Asynchronous neural network server** - Continuous batching with queue-based architecture
2. **Large batch sizes** - Up to 256+ positions per batch for optimal GPU utilization
3. **Non-blocking MCTS** - Workers continue exploring while waiting for NN results

## Architecture

### AsyncNeuralNetworkServer
- Runs in separate thread, continuously batching requests
- Maintains request queue with configurable max batch size
- Returns Future objects for asynchronous result retrieval
- Tracks performance metrics (batch sizes, latencies, GPU utilization)

### Async MCTS
- Virtual losses prevent workers from exploring same paths
- Workers submit NN requests and continue tree traversal
- Scales to 32-64+ workers efficiently
- Proper cleanup of virtual losses on async completion

### Multi-GPU Support
- NeuralNetworkPool distributes requests across GPUs
- Round-robin load balancing
- Independent servers per GPU for maximum throughput

## Usage

### Play Chess (Interactive)
```bash
# Original version (slow)
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --rollouts 1000 --threads 8

# Async version (fast)
python3 playchess_async.py --model weights/AlphaZeroNet_20x256.pt --rollouts 10000 --threads 32 --batch-size 256

# Multi-GPU
python3 playchess_async.py --model weights/AlphaZeroNet_20x256.pt --rollouts 10000 --threads 32 --gpus 0 1
```

### UCI Engine
```bash
# Original UCI engine
python3 UCI_engine.py --model weights/AlphaZeroNet_20x256.pt --threads 8

# Async UCI engine
python3 UCI_engine_async.py --model weights/AlphaZeroNet_20x256.pt --threads 32 --batch-size 256
```

### Self-Play Training Games
```bash
# Generate training games with async engine
python3 create_training_games_async.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --games 100 \
    --rollouts 800 \
    --threads 32 \
    --batch-size 256 \
    --parallel-games 10 \
    --output training_games \
    --format npz
```

### Benchmarking
```bash
# Compare original vs async performance
python3 benchmark_performance.py \
    --model weights/AlphaZeroNet_20x256.pt \
    --rollouts 1000 \
    --original-threads 8 \
    --async-threads 32 \
    --batch-size 256 \
    --runs 5
```

## Performance Tuning

### Key Parameters

1. **Worker Threads** (`--threads`)
   - Original: 8-10 threads typical
   - Async: 32-64 threads recommended
   - More workers = better GPU utilization

2. **Batch Size** (`--batch-size`)
   - Optimal: 128-512 depending on GPU memory
   - Larger batches = better GPU efficiency
   - Monitor GPU utilization in benchmark output

3. **Max Wait Time** (hardcoded: 1ms)
   - Time to wait for batch to fill
   - Lower = more responsive but smaller batches
   - Higher = larger batches but more latency

### Hardware Considerations

- **GPU Memory**: Larger batch sizes require more VRAM
- **CPU Cores**: More workers need more CPU threads
- **PCIe Bandwidth**: Can become bottleneck with very high throughput

### Expected Performance

On modern hardware (RTX 4090, M4 Pro, etc.):
- Original: 400-1,000 nodes/second
- Async: 5,000-20,000+ nodes/second
- Speedup: 10-50x typical

## Implementation Details

### Virtual Loss Handling
```python
# Add virtual loss during selection
edge.addVirtualLoss()

# Remove single virtual loss after evaluation
edge.removeVirtualLoss()
```

### Async Request Pattern
```python
# Submit evaluation request
future = nn_server.evaluate_async(board)

# Continue other work...

# Get result when needed
value, move_probs = future.result(timeout=0.1)
```

### Batch Processing
- Requests queued until batch full or timeout
- Single GPU kernel launch for entire batch
- Results distributed back through futures

## Monitoring

The async implementation provides detailed performance metrics:
- Nodes per second (NPS)
- Average batch size
- GPU utilization percentage
- Queue wait times
- Inference latencies

Use `--verbose` flag to see detailed statistics during execution.

## Troubleshooting

### Low GPU Utilization
- Increase worker threads
- Increase batch size
- Check for CPU bottlenecks

### High Latency
- Reduce batch size
- Reduce number of parallel games
- Check system resources

### Out of Memory
- Reduce batch size
- Reduce number of workers
- Use smaller model

## Future Optimizations

1. **Persistent Tree Cache** - Reuse subtrees between moves
2. **Distributed MCTS** - Scale across multiple machines
3. **Mixed Precision** - FP16 inference for 2x throughput
4. **Kernel Fusion** - Custom CUDA kernels for MCTS operations
5. **Zero-Copy Tensors** - Eliminate CPU-GPU transfer overhead