# CUDA MCTS Integration Guide

This guide shows how to use the CUDA-optimized MCTS in your existing code.

## Quick Start

### Option 1: Use the CUDA versions directly

```bash
# For playing chess
python playchess_cuda.py --model weights/model.pt --rollouts 1000 --threads 16

# For UCI engine
python UCI_engine_cuda.py --model weights/model.pt --threads 32
```

### Option 2: Modify existing code minimally

Add this to the beginning of your Python script:

```python
# Try to use CUDA MCTS, fall back to original if not available
try:
    import MCTS_cuda_optimized as MCTS
    print("Using CUDA-optimized MCTS")
except ImportError:
    import MCTS
    print("Using original MCTS")
```

### Option 3: Use the setup utility

```python
from use_cuda_mcts import setup_cuda_mcts
setup_cuda_mcts()  # This makes all subsequent MCTS imports use CUDA version

# Now this imports the CUDA version
import MCTS
```

## Integration Examples

### In playchess.py

To use CUDA MCTS in the original playchess.py without modifying it:

```python
# run_playchess_cuda.py
from use_cuda_mcts import setup_cuda_mcts
setup_cuda_mcts()

# Now import and run the original playchess
import playchess
playchess.main(...)
```

Or modify the import section in playchess.py:

```python
# Replace:
import MCTS

# With:
try:
    import MCTS_cuda_optimized as MCTS
    print("Using CUDA-optimized MCTS implementation")
except ImportError:
    import MCTS
    print("Using original MCTS implementation")
```

### In UCI_engine.py

Similar approach - either use UCI_engine_cuda.py directly or modify the import:

```python
# Replace:
import MCTS

# With:
try:
    import MCTS_cuda_optimized as MCTS
    USING_CUDA = True
except ImportError:
    import MCTS
    USING_CUDA = False
```

### In your custom code

```python
import chess
import torch

# Use CUDA MCTS if available
try:
    import MCTS_cuda_optimized as MCTS
except ImportError:
    import MCTS

# Load your model
model = torch.load('model.pt')

# Create board
board = chess.Board()

# Use MCTS exactly as before - it's a drop-in replacement
root = MCTS.Root(board, model)

# Run rollouts
for _ in range(1000):
    root.parallelRollouts(board.copy(), model, num_threads=16)

# Get best move
best_edge = root.maxNSelect()
best_move = best_edge.getMove()
```

## Configuration

Edit these values in `MCTS_cuda_optimized.py`:

```python
# Batching configuration
BATCH_SIZE = 256  # Increase for better GPU utilization (32-512)
MAX_BATCH_WAIT_TIME = 0.001  # Max wait time for batch (0.0005-0.002)

# Memory configuration
POSITION_CACHE_SIZE = 50000  # Position cache size
MOVE_CACHE_SIZE = 50000  # Legal move cache size

# GPU usage
USE_GPU_TREE = True  # Enable GPU tree operations
```

## Performance Tuning

### For Maximum Speed

```python
# High batch size for throughput
BATCH_SIZE = 512
MAX_BATCH_WAIT_TIME = 0.002

# Use many threads
num_threads = 32

# High rollout count
num_rollouts = 2000
```

### for Low Latency

```python
# Smaller batch size
BATCH_SIZE = 64
MAX_BATCH_WAIT_TIME = 0.0005

# Moderate threads
num_threads = 8

# Lower rollout count
num_rollouts = 500
```

## Troubleshooting

### Check if CUDA MCTS is working

```python
import MCTS_cuda_optimized as MCTS

print(f"C++ available: {MCTS.CPP_AVAILABLE}")
print(f"CUDA available: {MCTS.CUDA_AVAILABLE}")
print(f"Batch size: {MCTS.BATCH_SIZE}")
```

### Build extensions if needed

```bash
# Windows
build_windows.bat

# Linux/Mac
python setup_extensions.py build_ext --inplace
```

### Clear caches if memory issues

```python
import MCTS_cuda_optimized as MCTS

# Clear caches
MCTS.clear_caches()
MCTS.clear_batch_queue()
```

## API Compatibility

The CUDA MCTS is designed as a drop-in replacement with identical API:

- `Root(board, neural_network)` - Create root node
- `parallelRollouts(board, neural_network, num_threads)` - Run parallel rollouts  
- `maxNSelect()` - Get best move by visit count
- `getN()` - Get total visit count
- `getQ()` - Get average Q value
- `getStatisticsString()` - Get statistics string

Additional features in CUDA version:
- `clear_caches()` - Clear position/move caches
- `clear_batch_queue()` - Clear neural network batch queue
- Automatic GPU acceleration when available
- Transparent CPU fallback

## Performance Expectations

With CUDA optimizations on a good GPU (RTX 3080+):

- **Original MCTS**: ~600-1000 nps
- **CPU optimizations only**: ~1000-1500 nps  
- **CUDA optimizations**: ~2000-5000 nps
- **With FP16 model**: ~3000-8000 nps

Factors affecting performance:
- GPU model (newer is better)
- Batch size (larger is more efficient)
- Thread count (match CPU cores)
- Model size (smaller is faster)
- Rollout count (more rollouts = better GPU efficiency)