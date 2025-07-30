# Distributed Training Guide

This guide explains how to use the distributed training implementation for AlphaZero that supports training across multiple GPUs and multiple systems.

## Features

- **Single-node multi-GPU training**: Use all GPUs on a single machine
- **Multi-node multi-GPU training**: Scale across multiple machines with multiple GPUs each
- **Automatic batch size scaling**: Batch size is automatically distributed across GPUs
- **Mixed precision training**: Optional FP16 training for better performance
- **Fault tolerance**: Handles GPU memory constraints gracefully

## Quick Start

### Single Machine with Multiple GPUs

```bash
# Train on all available GPUs (auto-detected)
./launch_distributed_training.sh

# Train on specific number of GPUs (e.g., 4 GPUs)
./launch_distributed_training.sh --gpus 4

# With mixed precision for faster training
./launch_distributed_training.sh --gpus 8 --mixed-precision

# With custom batch size
./launch_distributed_training.sh --gpus 8 --batch-size 2048
```

### Multiple Machines (Multi-node)

For training across multiple systems, run these commands on each node:

**On the master node (node 0):**
```bash
./launch_distributed_training.sh \
    --gpus 8 \
    --nodes 2 \
    --node-rank 0 \
    --master-addr 192.168.1.100
```

**On worker nodes (node 1, 2, etc.):**
```bash
# On node 1
./launch_distributed_training.sh \
    --gpus 8 \
    --nodes 2 \
    --node-rank 1 \
    --master-addr 192.168.1.100

# On node 2 (if using 3+ nodes)
./launch_distributed_training.sh \
    --gpus 8 \
    --nodes 3 \
    --node-rank 2 \
    --master-addr 192.168.1.100
```

## Direct Python Usage

You can also run the distributed training script directly:

```bash
# Single GPU
python3 train_distributed.py

# Multiple GPUs on single node
python3 train_distributed.py --world-size 8

# With torchrun (recommended for PyTorch >= 1.10)
torchrun --nproc_per_node=8 train_distributed.py --batch-size 1024

# Multi-node with torchrun
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=12355 \
    train_distributed.py --batch-size 2048

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=12355 \
    train_distributed.py --batch-size 2048
```

## Command Line Arguments

### Launch Script Options
- `--gpus`: Number of GPUs per node (default: 8)
- `--nodes`: Total number of nodes (default: 1)
- `--node-rank`: Rank of current node (default: 0)
- `--master-addr`: IP address of master node (default: localhost)
- `--master-port`: Port for communication (default: 12355)
- `--batch-size`: Total batch size across all GPUs (default: 1024)
- `--mixed-precision`: Enable mixed precision training

### Python Script Options
- `--world-size`: Total number of processes
- `--rank`: Process rank
- `--local-rank`: Local process rank (set by torchrun)
- `--backend`: Communication backend (nccl for GPU, gloo for CPU)
- `--batch-size`: Total batch size
- `--num-workers`: Data loader workers per process
- `--learning-rate`: Learning rate (default: 0.001)
- `--mixed-precision`: Enable AMP training

## Performance Tips

1. **Batch Size**: The batch size is split across all GPUs. For 8 GPUs with batch size 1024, each GPU processes 128 samples.

2. **Network**: For multi-node training, ensure:
   - Low latency network connection between nodes
   - Firewall allows communication on the master port
   - All nodes can reach the master node IP

3. **Mixed Precision**: Enables faster training with minimal accuracy loss:
   ```bash
   ./launch_distributed_training.sh --gpus 8 --mixed-precision
   ```

4. **Data Loading**: The script automatically optimizes the number of workers based on your hardware.

## Monitoring

The training script will show:
- Loss values for each batch
- Average loss per epoch across all processes
- Model checkpoints saved as `AlphaZeroNet_20x256_distributed.pt`

Only the master process (rank 0) prints progress and saves checkpoints.

## Troubleshooting

1. **NCCL Errors**: Ensure all GPUs are visible:
   ```bash
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.device_count())"
   ```

2. **Connection Issues**: For multi-node:
   - Check firewall settings
   - Verify master node IP is accessible
   - Try different port if 12355 is in use

3. **Out of Memory**: Reduce batch size:
   ```bash
   ./launch_distributed_training.sh --gpus 8 --batch-size 512
   ```

4. **Different GPU Types**: The script handles heterogeneous GPUs but performance may vary.

## Example: 8 GPU System

For a system with 8 NVIDIA GPUs:

```bash
# Basic training
./launch_distributed_training.sh --gpus 8

# High performance settings
./launch_distributed_training.sh \
    --gpus 8 \
    --batch-size 2048 \
    --mixed-precision

# Multi-node (2 machines, 8 GPUs each = 16 total)
# Machine 1:
./launch_distributed_training.sh \
    --gpus 8 \
    --nodes 2 \
    --node-rank 0 \
    --master-addr YOUR_IP

# Machine 2:
./launch_distributed_training.sh \
    --gpus 8 \
    --nodes 2 \
    --node-rank 1 \
    --master-addr MACHINE_1_IP
```

The distributed training will automatically handle data parallelism, gradient synchronization, and model checkpointing across all GPUs.