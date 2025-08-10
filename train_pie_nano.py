import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR, MultiStepLR
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset
from PieNanoNetwork_v2 import PieNanoV2
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import time
from tqdm import tqdm

# Training params optimized for PieNano
default_epochs = 100
default_blocks = 8
default_filters = 128
default_lr = 0.001
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')
logmode = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train PieNano lightweight chess network')
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
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 2)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs for learning rate (default: 5)')
    parser.add_argument('--scheduler', choices=['cosine', 'onecycle', 'step', 'none'], default='cosine',
                        help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--use-se', action='store_true', default=True,
                        help='Use Squeeze-Excitation blocks (default: True)')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='Label smoothing for policy loss (default: 0.05)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--save-every', type=int, default=20,
                        help='Save checkpoint every N epochs (default: 20)')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Validate every N epochs (default: 1)')
    parser.add_argument('--policy-hidden-dim', type=int, default=256,
                        help='Hidden dimension for policy head (default: 256)')
    
    return parser.parse_args()

def create_model(args):
    """Create and configure the PieNano V2 model."""
    # Determine input planes based on encoder type
    num_input_planes = 112 if args.use_enhanced_encoder else 16
    
    model = PieNanoV2(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        num_input_planes=num_input_planes,
        use_se=args.use_se,
        dropout_rate=args.dropout,
        policy_weight=args.policy_weight,
        policy_hidden_dim=args.policy_hidden_dim
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
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    if args.scheduler == 'cosine':
        # Progressive cosine annealing for better long-term convergence
        # Instead of restarts, use a single cosine decay over all epochs
        # with milestone-based reductions for fine-tuning
        
        if args.epochs <= 50:
            # For shorter training, use original warm restarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=steps_per_epoch * 10,
                T_mult=2,
                eta_min=args.lr * 0.0001
            )
        else:
            # For longer training (100+ epochs), use progressive reduction
            # Cosine annealing over the full training duration
            
            # Create a composite scheduler with milestones
            milestones = [30, 60, 80, 90]  # Reduce LR at these epochs
            gamma = 0.5  # Reduce by 50% at each milestone
            
            # Use MultiStepLR with cosine annealing between milestones
            scheduler = MultiStepLR(
                optimizer,
                milestones=[m * steps_per_epoch // args.gradient_accumulation for m in milestones],
                gamma=gamma
            )
            
            # Alternative: Pure cosine annealing to very low LR
            # scheduler = CosineAnnealingLR(
            #     optimizer,
            #     T_max=total_steps,
            #     eta_min=args.lr * 0.00001  # 0.001% of initial LR
            # )
        
        # Create a warmup scheduler wrapper
        def warmup_scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_scheduler)
        return {'main': scheduler, 'warmup': warmup, 'warmup_steps': warmup_steps}
        
    elif args.scheduler == 'onecycle':
        # Adjusted OneCycleLR for better convergence
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 3,  # Reduced peak LR for stability
            total_steps=total_steps,
            pct_start=0.05,  # Shorter warmup
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            final_div_factor=10000  # End at 0.01% of max LR
        )
        return {'main': scheduler, 'warmup': None, 'warmup_steps': 0}
    elif args.scheduler == 'step':
        # New step-based scheduler option for manual control
        milestones = [30, 50, 70, 85, 95]  # Reduce at these epochs
        scheduler = MultiStepLR(
            optimizer,
            milestones=[m * steps_per_epoch // args.gradient_accumulation for m in milestones],
            gamma=0.3  # Aggressive reduction for fine-tuning
        )
        
        # Create a warmup scheduler wrapper
        def warmup_scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_scheduler)
        return {'main': scheduler, 'warmup': warmup, 'warmup_steps': warmup_steps}
    else:
        return {'main': None, 'warmup': None, 'warmup_steps': 0}

def train_epoch(model, dataloader, optimizer, scheduler, args, device, epoch, step_count):
    """Train for one epoch with gradient accumulation."""
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
        
        # Forward pass
        loss, value_loss, policy_loss = model(
            positions, value_targets, policy_targets
        )
        loss = loss / args.gradient_accumulation
        
        # Backward pass with gradient accumulation
        loss.backward()
        
        # Update weights after gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Learning rate scheduling
            if scheduler['warmup'] and step_count[0] < scheduler['warmup_steps']:
                scheduler['warmup'].step()
            elif scheduler['main'] is not None:
                scheduler['main'].step()
            
            step_count[0] += 1
        
        # Track losses
        total_loss += loss.item() * args.gradient_accumulation
        total_value_loss += value_loss.item()
        total_policy_loss += policy_loss.item()
        num_batches += 1
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'value': f'{total_value_loss/num_batches:.4f}',
            'policy': f'{total_policy_loss/num_batches:.4f}',
            'lr': f'{current_lr:.6f}'
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
        for batch in tqdm(dataloader, desc='Validation'):
            positions = batch['position'].to(device)
            value_targets = batch['value'].to(device)
            policy_targets = batch['policy'].to(device)
            
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
    
    # MPS-specific settings
    is_mps = device.type == 'mps'
    
    # Auto-detect batch size if not specified
    if args.batch_size is None:
        if is_mps:
            # MPS-optimized batch sizes for M4 Pro with 24GB RAM
            if args.use_enhanced_encoder:
                base_batch = 128  # Conservative for 112-plane encoder
            else:
                base_batch = 256  # Good for 16-plane encoder on M4 Pro
        else:
            # CPU or CUDA defaults
            base_batch = 64 if args.use_enhanced_encoder else 128
        
        args.batch_size = get_batch_size_for_device(base_batch)
    
    print(f"Batch size: {args.batch_size}")
    
    # Effective batch size with gradient accumulation
    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(f"Effective batch size: {effective_batch_size}")
    
    # Apply linear scaling rule for learning rate based on batch size
    if not args.no_lr_scaling:
        reference_batch_size = 256
        lr_scale = effective_batch_size / reference_batch_size
        scaled_lr = args.lr * lr_scale
        print(f"Base learning rate: {args.lr}")
        print(f"Scaled learning rate (linear scaling): {scaled_lr:.6f}")
        args.lr = scaled_lr
    
    # Create model
    model = create_model(args)
    model = optimize_for_device(model, device)
    
    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Load checkpoint if resuming
    start_epoch = 0
    step_count = [0]  # Use list to make it mutable for the closure
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        # Use weights_only=False to load checkpoints with saved args
        # This is safe since we're loading our own training checkpoints
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'step_count' in checkpoint:
            step_count[0] = checkpoint['step_count']
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
    
    # Create dataloaders with MPS-optimized settings
    if is_mps:
        # MPS-specific dataloader settings
        num_workers = min(6, get_num_workers_for_device())  # Limit workers on MPS
        persistent_workers = False  # Disable for MPS compatibility
        pin_memory = False  # MPS doesn't use pinned memory
    else:
        num_workers = get_num_workers_for_device()
        persistent_workers = (num_workers > 0)
        pin_memory = (device.type == 'cuda')
    
    print(f"Using {num_workers} data loader workers")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader) // args.gradient_accumulation)
    
    # Training loop
    best_val_loss = float('inf')
    
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_value_loss, train_policy_loss = train_epoch(
            model, train_loader, optimizer, scheduler, args, device, epoch, step_count
        )
        
        print(f"\nTrain - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}")
        
        # Validate periodically
        if (epoch + 1) % args.validate_every == 0:
            val_loss, val_value_loss, val_policy_loss = validate(model, val_loader, device)
            print(f"Val   - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}")
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                # Save best model
                if args.output:
                    best_path = args.output.replace('.pt', '_best.pt')
                else:
                    best_path = f'weights/PieNanoV2_{args.num_blocks}x{args.num_filters}_best.pt'
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler['main'].state_dict() if scheduler['main'] else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'step_count': step_count[0],
                    'args': args
                }
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path} (val_loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            if args.output:
                checkpoint_path = args.output.replace('.pt', f'_epoch{epoch+1}.pt')
            else:
                checkpoint_path = f'weights/PieNanoV2_{args.num_blocks}x{args.num_filters}_epoch{epoch+1}.pt'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler['main'].state_dict() if scheduler['main'] else None,
                'train_loss': train_loss,
                'step_count': step_count[0],
                'args': args
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save last epoch model
        if epoch == args.epochs - 1:
            if args.output:
                last_path = args.output
            else:
                last_path = f'weights/PieNanoV2_{args.num_blocks}x{args.num_filters}_last.pt'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler['main'].state_dict() if scheduler['main'] else None,
                'train_loss': train_loss,
                'step_count': step_count[0],
                'args': args
            }
            torch.save(checkpoint, last_path)
            print(f"Saved final model to {last_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()