# Multi-Process MCTS Implementation

This directory contains a multi-process implementation of Monte Carlo Tree Search (MCTS) for the AlphaZero chess engine. The implementation allows MCTS to utilize multiple CPU cores effectively by running tree search in parallel processes.

## Architecture

### Core Components

1. **Shared Memory Tree** (`shared_tree.py`)
   - Tree structure stored in shared memory accessible by all processes
   - Lock-free design for edge selection, with locks only for node updates
   - Efficient memory layout using fixed-size arrays

2. **Inference Server** (`inference_server.py`)
   - Dedicated process for neural network inference
   - Batches requests from multiple workers for GPU efficiency
   - Position caching to avoid redundant encoding

3. **Multi-Process MCTS** (`MCTS_multiprocess.py`)
   - Manages worker processes and coordinates tree search
   - Compatible interface with original MCTS implementation
   - Adaptive virtual loss scaling based on parallelism

### How It Works

1. **Master Process** creates the shared tree and spawns workers
2. **Worker Processes** perform MCTS rollouts in parallel:
   - Selection: Navigate tree using UCT with virtual losses
   - Expansion: Request NN evaluation for leaf nodes
   - Backpropagation: Update tree statistics
3. **Inference Server** batches and processes NN requests
4. **Communication** via multiprocessing queues

## Usage

### Command Line

```bash
# Use multi-process MCTS with UCI engine
python uci_engine.py --multiprocess

# Benchmark comparison
python test_multiprocess_mcts.py --rollouts 10000
```

### UCI Options

When using the UCI engine, you can enable multi-process mode:
```
setoption name UseMultiprocess value true
```

### Python API

```python
import MCTS_multiprocess as MCTS

# Create root node
root = MCTS.Root(board, neural_network)

# Run parallel rollouts
root.parallelRollouts(board, neural_network, num_rollouts=10000)

# Get best move
edge = root.maxNSelect()
best_move = edge.getMove()

# Cleanup
root.cleanup()
```

## Performance Considerations

1. **Process Count**: Default uses `CPU_count - 2` (reserves cores for main + inference)
2. **Batch Size**: Inference server batches up to 64 positions by default
3. **Virtual Loss**: Scaled by `sqrt(num_workers / 10)` to reduce path collisions
4. **Memory**: Shared tree supports up to 2M nodes, 256 edges per node

## Advantages

- **True Parallelism**: Overcomes Python GIL limitations
- **Better CPU Utilization**: Can use all available cores
- **Scalable**: Performance improves with more CPU cores
- **Compatible**: Drop-in replacement for threaded version

## Limitations

- **Memory Overhead**: Shared memory structures have fixed size
- **Process Startup**: Initial overhead for spawning processes
- **Platform Differences**: Performance varies by OS (Linux > macOS > Windows)

## Benchmarking

Run the benchmark script to compare threaded vs multi-process:

```bash
python test_multiprocess_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 10000
```

Expected speedup: 2-4x on modern multi-core systems (8+ cores).