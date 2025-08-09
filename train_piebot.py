import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from PieBotNetwork import PieBotNet, PieBotNetConfig
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import re
import time
import math
from tqdm import tqdm

# Training params (defaults)
default_epochs = 500
default_blocks = 45
default_filters = 480
default_lr = 0.001
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')
logmode = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train PieBot enhanced AlphaZero network')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate (default: {default_lr})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: auto-detect based on device)')
    parser.add_argument('--no-lr-scaling', action='store_true',
                        help='Disable automatic learning rate scaling based on batch size')
    parser.add_argument('--num-blocks', type=int, default=default_blocks,
                        help=f'Number of residual blocks (default: {default_blocks})')
    parser.add_argument('--num-filters', type=int, default=default_filters,
                        help=f'Number of filters (default: {default_filters})')
    parser.add_argument('--num-transformer-blocks', type=int, default=2,
                        help='Number of transformer blocks (default: 2)')
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
    parser.add_argument('--use-enhanced-encoder', action='store_true',
                        help='Use enhanced 112-plane encoder instead of 16-plane')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate (default: 1000)')
    parser.add_argument('--scheduler', choices=['cosine', 'onecycle', 'none'], default='cosine',
                        help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--use-se', action='store_true', default=True,
                        help='Use Squeeze-Excitation blocks (default: True)')
    parser.add_argument('--use-depthwise', action='store_true', default=True,
                        help='Use depthwise separable convolutions (default: True)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for policy loss (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    
    return parser.parse_args()

def create_model(args):
    """Create and configure the PieBotNet model."""
    # Determine input planes based on encoder type
    num_input_planes = 112 if args.use_enhanced_encoder else 16
    
    model = PieBotNet(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        num_transformer_blocks=args.num_transformer_blocks,
        num_input_planes=num_input_planes,
        use_se=args.use_se,
        use_depthwise=args.use_depthwise,
        dropout_rate=args.dropout,
        policy_weight=args.policy_weight
    )
    
    return model

def create_optimizer(model, args):
    """Create optimizer with weight decay."""
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bn' in name or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    
    return optimizer

def create_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch * 5,  # Restart every 5 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=args.lr * 0.01  # Minimum LR is 1% of initial
        )
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,  # Peak LR is 10x initial
            total_steps=steps_per_epoch * args.epochs,
            pct_start=0.1,  # 10% of training for warmup
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, dataloader, optimizer, scheduler, scaler, args, device, epoch):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Extract data from batch dictionary
        positions = batch['position'].to(device)
        value_targets = batch['value'].to(device)
        policy_targets = batch['policy'].to(device)
        # Note: batch['mask'] is available if needed for legal move masking
        
        # Mixed precision training
        if not args.no_mixed_precision and device.type == 'cuda':
            with autocast():
                loss, value_loss, policy_loss = model(
                    positions, value_targets, policy_targets
                )
                loss = loss / args.gradient_accumulation
        else:
            loss, value_loss, policy_loss = model(
                positions, value_targets, policy_targets
            )
            loss = loss / args.gradient_accumulation
        
        # Backward pass with gradient accumulation
        if not args.no_mixed_precision and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights after gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            if not args.no_mixed_precision and device.type == 'cuda':
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
        
        # Track losses
        total_loss += loss.item() * args.gradient_accumulation
        total_value_loss += value_loss.item()
        total_policy_loss += policy_loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'value': f'{total_value_loss/num_batches:.4f}',
            'policy': f'{total_policy_loss/num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return total_loss / num_batches, total_value_loss / num_batches, total_policy_loss / num_batches

def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for positions, value_targets, policy_targets in tqdm(dataloader, desc='Validation'):
            positions = positions.to(device)
            value_targets = value_targets.to(device)
            policy_targets = policy_targets.to(device)
            
            loss, value_loss, policy_loss = model(
                positions, value_targets, policy_targets
            )
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
    
    return total_loss / num_batches, total_value_loss / num_batches, total_policy_loss / num_batches

def main():
    args = parse_args()
    
    # Get device
    device, device_str = get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Auto-detect batch size if not specified
    if args.batch_size is None:
        # For enhanced encoder (112 planes), use smaller base due to memory requirements
        # For regular encoder (16 planes), use larger base
        # Aggressive base sizes to maximize GPU utilization
        # Enhanced encoder with transformer blocks can handle large batches
        # Regular encoder can handle even larger batches
        base_batch = 256 if args.use_enhanced_encoder else 4096
        args.batch_size = get_batch_size_for_device(base_batch)
    print(f"Batch size: {args.batch_size}")
    
    # Effective batch size with gradient accumulation
    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(f"Effective batch size: {effective_batch_size}")
    
    # Apply linear scaling rule for learning rate based on batch size
    # Reference batch size is 256, scale linearly from there
    if not args.no_lr_scaling:
        reference_batch_size = 256
        lr_scale = effective_batch_size / reference_batch_size
        scaled_lr = args.lr * lr_scale
        print(f"Base learning rate: {args.lr}")
        print(f"Scaled learning rate (linear scaling): {scaled_lr:.6f}")
        # Use the scaled learning rate
        args.lr = scaled_lr
    
    # Create model
    model = create_model(args)
    model = optimize_for_device(model, device)
    
    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create datasets
    print(f"Loading datasets...")
    
    if args.mode == 'supervised':
        dataset = CCRLDataset(ccrl_dir, enhanced_encoder=args.use_enhanced_encoder)
    elif args.mode == 'rl':
        dataset = RLDataset(args.rl_dir)
    else:  # mixed
        ccrl_dataset = CCRLDataset(ccrl_dir, enhanced_encoder=args.use_enhanced_encoder)
        rl_dataset = RLDataset(args.rl_dir)
        
        # Calculate sizes for mixing
        total_size = len(ccrl_dataset) + len(rl_dataset)
        rl_samples = int(total_size * args.mixed_ratio)
        ccrl_samples = total_size - rl_samples
        
        # Sample from datasets
        ccrl_sampled = torch.utils.data.Subset(
            ccrl_dataset,
            torch.randperm(len(ccrl_dataset))[:ccrl_samples].tolist()
        )
        rl_sampled = torch.utils.data.Subset(
            rl_dataset,
            torch.randperm(len(rl_dataset))[:rl_samples].tolist()
        )
        
        dataset = ConcatDataset([ccrl_sampled, rl_sampled])
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train and validation
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    num_workers = get_num_workers_for_device()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0)
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader))
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if not args.no_mixed_precision and device.type == 'cuda' else None
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_value_loss, train_policy_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args, device, epoch
        )
        
        print(f"\nTrain - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}")
        
        # Validate
        val_loss, val_value_loss, val_policy_loss = validate(model, val_loader, device)
        print(f"Val   - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}")
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Check if this is the last epoch
        is_last_epoch = (epoch == args.epochs - 1)
        
        # Create weights directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': args
        }
        
        # Save best model
        if is_best:
            if args.output:
                best_path = args.output.replace('.pt', '_best.pt')
            else:
                best_path = f'weights/PieBotNet_{args.num_blocks}x{args.num_filters}_best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path} (val_loss: {val_loss:.4f})")
        
        # Save last epoch model
        if is_last_epoch:
            if args.output:
                last_path = args.output
            else:
                last_path = f'weights/PieBotNet_{args.num_blocks}x{args.num_filters}_last.pt'
            torch.save(checkpoint, last_path)
            print(f"Saved final model to {last_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()