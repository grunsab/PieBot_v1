#!/usr/bin/env python3
"""
Training script optimized for Windows/CUDA (RTX 4080)

High-performance training with mixed precision, gradient accumulation,
and multi-GPU support for RTX 4080 and similar GPUs.
"""

import os
import sys
sys.path.append('..')
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from AlphaZeroNetwork import AlphaZeroNet
import argparse
import re
import time
import numpy as np
from tqdm import tqdm

# Windows-specific optimizations
if sys.platform == 'win32':
    # Prevent Windows from sleeping during training
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    
    # Set process priority
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except:
        pass

# Training defaults optimized for RTX 4080
default_epochs = 40
default_blocks = 20
default_filters = 256
default_lr = 0.001  # Higher LR for CUDA with proper scheduling
default_policy_weight = 1.0
default_batch_size = 1024  # Much larger batch size for RTX 4080
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')

def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero network - CUDA optimized')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate (default: {default_lr})')
    parser.add_argument('--batch-size', type=int, default=default_batch_size,
                        help=f'Batch size (default: {default_batch_size})')
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
                        help='Weight recent games more heavily in RL mode')
    parser.add_argument('--rl-weight-decay', type=float, default=0.05,
                        help='Weight decay factor for older games (default: 0.05)')
    
    # CUDA-specific options
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use all available GPUs with DataParallel')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use automatic mixed precision training (default: True)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loader workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer (default: True)')
    parser.add_argument('--benchmark', action='store_true', default=True,
                        help='Enable cuDNN benchmark mode (default: True)')
    parser.add_argument('--compile', action='store_true',
                        help='Compile model with torch.compile (PyTorch 2.0+)')
    
    return parser.parse_args()

def setup_cuda_optimizations():
    """Set up CUDA-specific optimizations."""
    # Enable cuDNN optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 for Ampere GPUs (RTX 30/40 series)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def create_optimizer(model, lr, weight_decay=1e-4):
    """Create optimizer with settings optimized for large batch training."""
    # Use AdamW with decoupled weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8
    )
    return optimizer

def create_lr_scheduler(optimizer, epochs, warmup_epochs=5):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train():
    args = parse_args()
    
    # Set up CUDA optimizations
    setup_cuda_optimizations()
    
    # Device setup
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This version requires an NVIDIA GPU.")
        return
        
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        device = torch.device('cuda')
        multi_gpu = True
    else:
        device = torch.device(f'cuda:{args.device}')
        multi_gpu = False
        
    device_name = torch.cuda.get_device_name(args.device)
    device_memory = torch.cuda.get_device_properties(args.device).total_memory / 1024**3
    print(f'Using device: {device_name} ({device_memory:.1f}GB)')
    
    # Create dataset based on training mode
    if args.mode == 'supervised':
        print(f'Training mode: Supervised learning on CCRL dataset')
        train_ds = CCRLDataset(ccrl_dir, soft_targets=True)
    elif args.mode == 'rl':
        print(f'Training mode: Reinforcement learning on self-play data')
        if args.rl_weight_recent:
            print(f'Using weighted sampling with decay factor {args.rl_weight_decay}')
        train_ds = RLDataset(args.rl_dir, weight_recent=args.rl_weight_recent, 
                           weight_decay=args.rl_weight_decay)
    else:  # mixed mode
        print(f'Training mode: Mixed (RL ratio: {args.mixed_ratio})')
        print(f'Using soft targets with temperature {args.label_smoothing_temp}')
        ccrl_ds = CCRLDataset(ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp)
        rl_ds = RLDataset(args.rl_dir)
        
        # Calculate sizes for balanced sampling
        ccrl_size = int(len(ccrl_ds) * (1 - args.mixed_ratio))
        rl_size = int(len(rl_ds) * args.mixed_ratio)
        
        # Create subset indices
        import random
        ccrl_indices = random.sample(range(len(ccrl_ds)), min(ccrl_size, len(ccrl_ds)))
        rl_indices = random.sample(range(len(rl_ds)), min(rl_size, len(rl_ds)))
        
        # Create subsets
        from torch.utils.data import Subset
        ccrl_subset = Subset(ccrl_ds, ccrl_indices)
        rl_subset = Subset(rl_ds, rl_indices)
        
        # Concatenate datasets
        train_ds = ConcatDataset([ccrl_subset, rl_subset])
        print(f'Dataset sizes - CCRL: {len(ccrl_subset)}, RL: {len(rl_subset)}')
    
    print(f'Total training samples: {len(train_ds)}')
    
    # Create data loader with optimizations
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size // args.gradient_accumulation,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
        drop_last=True  # Drop last incomplete batch for consistency
    )
    
    # Create model
    model_net = AlphaZeroNet(args.num_blocks, args.num_filters)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_net.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f'Resuming from epoch {start_epoch}')
        else:
            model_net.load_state_dict(checkpoint)
    
    # Move model to device
    model_net = model_net.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model_net = torch.compile(model_net, mode='reduce-overhead')
    
    # Multi-GPU setup
    if multi_gpu:
        model_net = DataParallel(model_net)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model_net, args.lr)
    scheduler = create_lr_scheduler(optimizer, args.epochs)
    
    # Mixed precision setup
    scaler = GradScaler() if args.mixed_precision else None
    
    # Loss functions
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f'\nStarting training:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size} (accumulation: {args.gradient_accumulation})')
    print(f'  Learning rate: {args.lr}')
    print(f'  Mixed precision: {args.mixed_precision}')
    print('-' * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        model_net.train()
        
        # Metrics
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_total_loss = 0.0
        batch_times = []
        
        # Progress bar
        pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        optimizer.zero_grad()
        
        for batch_idx, (positions, target_values, target_policies, masks) in enumerate(pbar):
            batch_start = time.time()
            
            # Move to device
            positions = positions.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            target_policies = target_policies.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if args.mixed_precision:
                with autocast():
                    pred_values, pred_policies = model_net(positions, masks)
                    
                    # Compute losses
                    value_loss = value_loss_fn(pred_values.squeeze(), target_values)
                    policy_loss = policy_loss_fn(pred_policies, target_policies)
                    total_loss = value_loss + args.policy_weight * policy_loss
                    
                    # Scale for gradient accumulation
                    total_loss = total_loss / args.gradient_accumulation
                
                # Backward pass
                scaler.scale(total_loss).backward()
                
                # Update weights if accumulation is complete
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                pred_values, pred_policies = model_net(positions, masks)
                
                value_loss = value_loss_fn(pred_values.squeeze(), target_values)
                policy_loss = policy_loss_fn(pred_policies, target_policies)
                total_loss = value_loss + args.policy_weight * policy_loss
                total_loss = total_loss / args.gradient_accumulation
                
                total_loss.backward()
                
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_total_loss += (value_loss.item() + args.policy_weight * policy_loss.item())
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'v_loss': f'{value_loss.item():.4f}',
                'p_loss': f'{policy_loss.item():.4f}',
                'batch_time': f'{batch_time:.3f}s'
            })
        
        # Step scheduler
        scheduler.step()
        
        # Epoch statistics
        num_batches = len(train_dl)
        avg_value_loss = epoch_value_loss / num_batches
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_batch_time = np.mean(batch_times)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Average total loss: {avg_total_loss:.4f}')
        print(f'  Average value loss: {avg_value_loss:.4f}')
        print(f'  Average policy loss: {avg_policy_loss:.4f}')
        print(f'  Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  Average batch time: {avg_batch_time:.3f}s')
        print(f'  Throughput: {args.batch_size/avg_batch_time:.0f} samples/sec')
        
        # Save checkpoint
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            
            # Determine output filename
            if args.output:
                save_path = args.output
            else:
                save_path = f'AlphaZeroNet_{args.num_blocks}x{args.num_filters}_cuda.pt'
            
            # Get the actual model (unwrap DataParallel if needed)
            model_to_save = model_net.module if multi_gpu else model_net
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': avg_total_loss,
                'args': args
            }
            
            if args.mixed_precision:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, save_path)
            print(f'  Saved best model to {save_path}')
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f'  Saved checkpoint to {checkpoint_path}')
    
    print('\nTraining complete!')
    print(f'Best loss: {best_loss:.4f}')
    
    # Save final model
    if args.output:
        final_path = args.output.replace('.pt', '_final.pt')
    else:
        final_path = f'AlphaZeroNet_{args.num_blocks}x{args.num_filters}_cuda_final.pt'
    
    model_to_save = model_net.module if multi_gpu else model_net
    torch.save(model_to_save.state_dict(), final_path)
    print(f'Saved final model to {final_path}')

if __name__ == '__main__':
    train()