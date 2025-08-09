# Multi-Process MCTS Implementation Status

## Current Status

The multi-process MCTS implementation is complete but encounters compatibility issues with Python 3.9 due to changes in the multiprocessing module's Manager implementation. Specifically, the `manager_owned` parameter issue prevents proper queue sharing between processes when using the 'spawn' start method (required for macOS).

## The Issue

When running on Python 3.9 with multiprocessing spawn mode, you'll see:
```
TypeError: AutoProxy() got an unexpected keyword argument 'manager_owned'
```

This is a known issue with Python 3.9's multiprocessing Manager when sharing Queue objects between processes.

## Solutions

### Option 1: Upgrade Python (Recommended)

The issue is fixed in Python 3.10+. To upgrade:

```bash
# Using Homebrew on macOS
brew install python@3.11

# Or using pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

### Option 2: Use Fork Mode (Linux Only)

On Linux systems, you can use fork mode instead of spawn:

```python
mp.set_start_method('fork', force=True)
```

Note: This won't work on macOS with Apple Silicon.

### Option 3: Use the Threaded Version

The threaded MCTS implementation (`MCTS_profiling_speedups_v2.py`) works well and provides good performance, though it's limited by Python's GIL.

## Architecture Overview

The multi-process implementation consists of:

1. **Shared Memory Tree** (`shared_tree.py`)
   - Tree nodes and edges stored in shared memory
   - Lock-free edge selection with atomic operations
   - Supports up to 2M nodes

2. **Inference Server** (`inference_server.py`)
   - Dedicated process for neural network evaluation
   - Batches requests for GPU efficiency
   - Position caching

3. **Worker Processes** (`MCTS_multiprocess.py`)
   - Perform parallel tree search
   - Communicate via queues with inference server
   - Virtual loss prevents path collisions

## Performance Expectations

When working properly (Python 3.10+), the multi-process implementation should provide:
- 2-4x speedup over threaded version on 8+ core systems
- Better CPU utilization (overcomes GIL)
- Scalability with core count

## Testing

To test if your environment supports multi-process MCTS:

```bash
# Check Python version
python3 --version

# If Python 3.10+, test multi-process
python3 test_multiprocess_mcts.py --rollouts 1000

# For Python 3.9, use threaded version
python3 playchess.py --model weights/AlphaZeroNet_20x256.pt --rollouts 1000
```

## Future Work

1. Implement a pipe-based communication system that avoids Manager issues
2. Add support for distributed MCTS across multiple machines
3. Optimize shared memory layout for cache efficiency