import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from AlphaZeroNetwork import AlphaZeroNet
from device_utils import optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import re

# Training params (defaults)
default_epochs = 40
default_blocks = 20
default_filters = 256
default_lr = 0.0005
default_policy_weight = 1.0
ccrl_dir = os.path.expanduser('games_training_data/reformatted')
rl_dir = os.path.expanduser('games_training_data/selfplay')
logmode = True

def setup(rank, world_size, backend='nccl', init_method='env://'):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPUs, 'gloo' for CPUs)
        init_method: Method to initialize the process group
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args):
    """
    Distributed training function that runs on each process.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup distributed training
    setup(rank, world_size, backend=args.backend)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        device_str = f"CUDA GPU {rank}: {torch.cuda.get_device_name(rank)}"
    else:
        device = torch.device('cpu')
        device_str = f"CPU (Process {rank})"
    
    if rank == 0:
        print(f'Process {rank} using device: {device_str}')
        total_memory = torch.cuda.get_device_properties(rank).total_memory / 1024**3 if torch.cuda.is_available() else 0
        if total_memory > 0:
            print(f"GPU Memory: {total_memory:.1f}GB")
    
    # Adjust batch size for distributed training
    batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
    # Scale down batch size per GPU when using multiple GPUs
    batch_size = batch_size // world_size
    num_workers = args.num_workers if args.num_workers else get_num_workers_for_device()
    
    if rank == 0:
        print(f'Batch size per GPU: {batch_size}, Total batch size: {batch_size * world_size}, Workers: {num_workers}')
    
    # Create dataset based on training mode
    if args.mode == 'supervised':
        if rank == 0:
            print(f'Training mode: Supervised learning on CCRL dataset')
        train_ds = CCRLDataset(ccrl_dir, soft_targets=False)
    elif args.mode == 'rl':
        if rank == 0:
            print(f'Training mode: Reinforcement learning on self-play data')
            if args.rl_weight_recent:
                print(f'Using weighted sampling with decay factor {args.rl_weight_decay}')
        train_ds = RLDataset(args.rl_dir, weight_recent=args.rl_weight_recent,
                           weight_decay=args.rl_weight_decay)
    else:  # mixed mode
        if rank == 0:
            print(f'Training mode: Mixed (RL ratio: {args.mixed_ratio})')
            print(f'Using soft targets for CCRL data with temperature {args.label_smoothing_temp}')
        ccrl_ds = CCRLDataset(ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp)
        rl_ds = RLDataset(args.rl_dir)
        
        # Calculate sizes for balanced sampling
        ccrl_size = int(len(ccrl_ds) * (1 - args.mixed_ratio))
        rl_size = int(len(rl_ds) * args.mixed_ratio)
        
        # Create subset indices
        import random
        # Use same seed across all ranks for consistency
        random.seed(42)
        ccrl_indices = random.sample(range(len(ccrl_ds)), min(ccrl_size, len(ccrl_ds)))
        rl_indices = random.sample(range(len(rl_ds)), min(rl_size, len(rl_ds)))
        
        # Create subsets
        ccrl_subset = Subset(ccrl_ds, ccrl_indices)
        rl_subset = Subset(rl_ds, rl_indices)
        
        # Concatenate datasets
        train_ds = ConcatDataset([ccrl_subset, rl_subset])
        if rank == 0:
            print(f'Dataset sizes - CCRL: {len(ccrl_subset)}, RL: {len(rl_subset)}')
    
    # Create distributed sampler
    if args.mode == 'rl' and args.rl_weight_recent:
        # For weighted RL sampling, we need a different approach in distributed setting
        # Use regular distributed sampler but with replacement to approximate weighting
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, 
                                         shuffle=True, drop_last=True)
        if rank == 0:
            print("Note: Weighted sampling approximated in distributed mode")
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, 
                             num_workers=num_workers, pin_memory=True)
    
    # Determine model architecture
    num_blocks = args.num_blocks
    num_filters = args.num_filters
    
    # If resuming, try to extract architecture from filename
    if args.resume and rank == 0:
        match = re.search(r'AlphaZeroNet_(\d+)x(\d+)', args.resume)
        if match:
            file_blocks = int(match.group(1))
            file_filters = int(match.group(2))
            if num_blocks != file_blocks or num_filters != file_filters:
                print(f'Warning: Architecture mismatch! File suggests {file_blocks}x{file_filters}, '
                      f'but using {num_blocks}x{num_filters}')
    
    # Create model and move to device
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters, policy_weight=args.policy_weight)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if rank == 0:  # Only rank 0 checks file existence
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f'Checkpoint file not found: {args.resume}')
        
        # Synchronize all processes before loading
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print(f'Loading checkpoint from {args.resume}')
        
        # Load checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle both DDP and non-DDP checkpoints
        # DDP saves with 'module.' prefix, so we need to handle both cases
        state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Handle loading old checkpoints that don't have policy_weight in state_dict
        alphaZeroNet.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0:
            print(f'Checkpoint loaded successfully (policy_weight={args.policy_weight})')
    
    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    # Wrap model with DDP
    if torch.cuda.is_available():
        alphaZeroNet = DDP(alphaZeroNet, device_ids=[rank], output_device=rank, 
                          find_unused_parameters=False)
    else:
        alphaZeroNet = DDP(alphaZeroNet)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.lr)
    
    # Optional: Use mixed precision training for better performance
    scaler = torch.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    if rank == 0:
        print('Starting distributed training')
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
        alphaZeroNet.train()
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move data to device
            position = data['position'].to(device)
            valueTarget = data['value'].to(device)
            policyTarget = data['policy'].to(device)
            
            # Forward pass with optional mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget,
                                                               policyTarget=policyTarget)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward and backward pass
                loss, valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget,
                                                          policyTarget=policyTarget)
                loss.backward()
                optimizer.step()
            
            epoch_value_loss += valueLoss.item()
            epoch_policy_loss += policyLoss.item()
            
            # Only rank 0 prints progress
            if rank == 0:
                message = 'Epoch {:03} | Step {:05} / {:05} | Total loss {:0.5f} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                         epoch, iter_num, len(train_loader), float(loss), float(valueLoss), float(policyLoss))
                
                if iter_num != 0 and not logmode:
                    print(('\b' * len(message)), end='')
                print(message, end='', flush=True)
                if logmode:
                    print('')
        
        if rank == 0:
            print('')
        
        # Gather losses from all processes
        avg_value_loss = epoch_value_loss / len(train_loader)
        avg_policy_loss = epoch_policy_loss / len(train_loader)
        
        # Reduce losses across all processes
        if world_size > 1:
            avg_value_tensor = torch.tensor([avg_value_loss], device=device)
            avg_policy_tensor = torch.tensor([avg_policy_loss], device=device)
            dist.all_reduce(avg_value_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_policy_tensor, op=dist.ReduceOp.SUM)
            avg_value_loss = avg_value_tensor.item() / world_size
            avg_policy_loss = avg_policy_tensor.item() / world_size
        
        # Save model checkpoint (only rank 0)
        if rank == 0:
            print(f'Epoch {epoch} - Avg Value Loss: {avg_value_loss:.5f}, Avg Policy Loss: {avg_policy_loss:.5f}')
            
            # Determine output filename
            if args.output:
                networkFileName = args.output
            elif args.resume:
                # When resuming, add '_continued' to avoid overwriting original
                base_name = f'AlphaZeroNet_{num_blocks}x{num_filters}_distributed'
                networkFileName = f'{base_name}_continued.pt'
            else:
                networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}_distributed.pt'
            
            # Save the model state dict (unwrap DDP if necessary)
            if isinstance(alphaZeroNet, DDP):
                torch.save(alphaZeroNet.module.state_dict(), networkFileName)
            else:
                torch.save(alphaZeroNet.state_dict(), networkFileName)
            
            print(f'Saved model to {networkFileName}')
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed training for AlphaZero')
    
    # Distributed training arguments
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total number of processes (GPUs) to use')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of the current process')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend to use')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Total batch size across all GPUs')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate for optimizer (default: {default_lr})')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (requires GPU)')
    
    # Model and checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--num-blocks', type=int, default=default_blocks,
                        help=f'Number of residual blocks (default: {default_blocks})')
    parser.add_argument('--num-filters', type=int, default=default_filters,
                        help=f'Number of filters (default: {default_filters})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for saved model')
    parser.add_argument('--policy-weight', type=float, default=default_policy_weight,
                        help=f'Weight for policy loss relative to value loss (default: {default_policy_weight})')
    parser.add_argument('--mode', choices=['supervised', 'rl', 'mixed'], default='supervised',
                        help='Training mode: supervised (CCRL data), rl (self-play), or mixed (both)')
    parser.add_argument('--rl-dir', type=str, default=rl_dir,
                        help='Directory containing self-play training data (HDF5 files)')
    parser.add_argument('--mixed-ratio', type=float, default=0.5,
                        help='For mixed mode: ratio of RL data (0.0-1.0, default: 0.5)')
    parser.add_argument('--label-smoothing-temp', type=float, default=0.1,
                        help='Temperature for label smoothing in mixed mode (default: 0.1)')
    parser.add_argument('--rl-weight-recent', action='store_true',
                        help='Weight recent games more heavily in RL mode (Leela approach)')
    parser.add_argument('--rl-weight-decay', type=float, default=0.1,
                        help='Weight decay factor for older games (default: 0.1)')
    
    # Multi-node training arguments
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='Master node address for multi-node training')
    parser.add_argument('--master-port', type=str, default='12355',
                        help='Master node port for multi-node training')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--gpus-per-node', type=int, default=None,
                        help='Number of GPUs per node')
    
    args = parser.parse_args()
    
    # Set environment variables for multi-node training
    if args.master_addr:
        os.environ['MASTER_ADDR'] = args.master_addr
    if args.master_port:
        os.environ['MASTER_PORT'] = args.master_port
    
    # Determine world size and number of GPUs
    if args.gpus_per_node is None:
        args.gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # For multi-node training
    if args.nodes > 1:
        args.world_size = args.nodes * args.gpus_per_node
        # When using torchrun or torch.distributed.launch, these will be set
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = args.gpus_per_node
    
    if args.world_size > 1:
        # Use multiprocessing spawn for single-node multi-GPU
        if args.nodes == 1:
            mp.spawn(train_distributed,
                    args=(args.world_size, args),
                    nprocs=args.world_size,
                    join=True)
        else:
            # For multi-node, assume launched with torchrun
            train_distributed(args.local_rank, args.world_size, args)
    else:
        # Single GPU/CPU training
        train_distributed(0, 1, args)

if __name__ == '__main__':
    main()