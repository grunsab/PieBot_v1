# Performance Profiling Guide

This guide explains how to profile the CPU-bound operations in the PyTorch AlphaZero chess engine.

## Overview

Based on your benchmark results:
- **Theoretical GPU performance**: 5400 nodes/sec (FP16), 4350 nodes/sec (FP32)  
- **Actual performance**: ~1000 nodes/sec
- **Bottleneck**: CPU-bound operations in MCTS and position encoding

## Profiling Tools

### 1. profile_mcts.py - cProfile Integration
Provides high-level profiling of the entire MCTS pipeline.

```bash
# Basic profiling
python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 100 --threads 1

# With component analysis
python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 100 --threads 10 --components

# Save results to file
python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 100 --threads 10 --output profile_results.txt
```

**Output includes:**
- Function-level timing breakdown
- Component-wise performance metrics (encoding, NN inference, UCT calculations, tree traversal)
- Nodes per second measurement

### 2. playchess_instrumented.py - Detailed Timing
Uses MCTS_instrumented.py to collect fine-grained timing data.

```bash
# Run with verbose output to see timing statistics
python playchess_instrumented.py --model weights/AlphaZeroNet_20x256.pt --mode p --rollouts 100 --threads 10 --verbose
```

**Timing data collected:**
- `calcUCT`: Time per UCT calculation
- `UCTSelect`: Time to select best child
- `selectTask`: Tree traversal time
- `neural_network_single`: Single inference time
- `neural_network_batch`: Batch inference time
- `backpropagation`: Tree update time
- `parallel_selection`: Parallel selection overhead
- And more...

### 3. profile_line.py - Line-by-Line Analysis
Requires `pip install line_profiler` for detailed line-level profiling.

```bash
# Profile specific functions
python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function calcUCT
python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function UCTSelect
python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function selectTask
```

**Available functions to profile:**
- `calcUCT`: UCT formula calculation
- `UCTSelect`: Node child selection
- `selectTask`: Tree traversal
- `encodePosition`: Board encoding
- `callNeuralNetwork`: NN inference
- `parallelRollouts`: Parallel rollout coordination
- `expand`: Node expansion
- `updateStats`: Statistics update

## Interpreting Results

### Key Metrics to Watch

1. **UCT Calculations per Second**: Should be >1M for good performance
2. **Tree Traversals per Second**: Shows selection phase efficiency  
3. **Neural Network Batch Efficiency**: Batch should be faster per-position than single
4. **Lock Contention Time**: Time spent waiting for thread locks

### Common Bottlenecks

1. **calcUCT Function**
   - Called for every edge during selection
   - Contains expensive sqrt and division operations
   - Solution: Vectorize with NumPy

2. **Position Encoding**
   - Repeated encoding of same positions
   - Solution: Add caching layer

3. **Thread Synchronization**
   - Lock contention in parallel rollouts
   - Solution: Lock-free data structures or finer-grained locks

4. **Python Object Overhead**
   - Creating many Node/Edge objects
   - Solution: Object pooling or C++ implementation

## Optimization Opportunities

Based on profiling, consider:

1. **Immediate wins:**
   - Vectorize UCT calculations
   - Cache position encodings
   - Use numba JIT for hot functions

2. **Medium-term improvements:**
   - Reduce lock granularity
   - Batch more operations
   - Preallocate memory pools

3. **Long-term solutions:**
   - C++ implementation of MCTS
   - GPU-accelerated tree operations
   - Custom CUDA kernels for parallel selection

## Example Workflow

1. Run initial profile to identify bottlenecks:
   ```bash
   python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --components
   ```

2. Use line profiler on slowest functions:
   ```bash
   python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function UCTSelect
   ```

3. Run instrumented version for detailed timing:
   ```bash
   python playchess_instrumented.py --model weights/AlphaZeroNet_20x256.pt --mode p --verbose
   ```

4. Compare before/after optimization using the same commands.