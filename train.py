import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from CurriculumDataset import CurriculumDataset, MixedCurriculumDataset
from piece_value_monitor import PieceValueMonitor
from AlphaZeroNetwork import AlphaZeroNet
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import re
import time
import numpy as np
from datetime import datetime
import json

#Training params (defaults)
default_epochs = 200
default_blocks = 20
default_filters = 256
default_lr = 0.001
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')

def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero network')
    
    # Model architecture
    parser.add_argument('--num-blocks', type=int, default=default_blocks,
                        help=f'Number of residual blocks (default: {default_blocks})')
    parser.add_argument('--num-filters', type=int, default=default_filters,
                        help=f'Number of filters (default: {default_filters})')
    
    # Training params
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate (default: {default_lr})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: auto-detect based on device)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0, 0 to disable)')
    
    # Scheduler options
    parser.add_argument('--scheduler', choices=['onecycle', 'cosine', 'plateau', 'none'], default='onecycle',
                        help='Learning rate scheduler (default: onecycle)')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Number of warmup epochs for OneCycleLR (default: 2)')
    
    # Model params
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for saved model')
    parser.add_argument('--policy-weight', type=float, default=default_policy_weight,
                        help=f'Weight for policy loss relative to value loss (default: {default_policy_weight})')
    
    # Data params
    parser.add_argument('--mode', choices=['supervised', 'rl', 'mixed', 'curriculum', 'mixed-curriculum'], default='supervised',
                        help='Training mode: supervised (CCRL), rl (self-play), mixed (both), curriculum (progressive), mixed-curriculum')
    parser.add_argument('--ccrl-dir', type=str, default=ccrl_dir,
                        help='Directory containing CCRL training data')
    parser.add_argument('--rl-dir', type=str, default=rl_dir,
                        help='Directory containing self-play training data (HDF5 files)')
    parser.add_argument('--mixed-ratio', type=float, default=0.5,
                        help='For mixed mode: ratio of RL data (0.0-1.0, default: 0.5)')
    parser.add_argument('--label-smoothing-temp', type=float, default=0.1,
                        help='Temperature for label smoothing in mixed mode (default: 0.1)')
    parser.add_argument('--rl-weight-recent', action='store_true',
                        help='Weight recent games more heavily in RL mode (Leela approach)')
    parser.add_argument('--rl-weight-decay', type=float, default=0.05,
                        help='Weight decay factor for older games (default: 0.05)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    
    # Curriculum learning params
    parser.add_argument('--curriculum-dir', type=str, default='games_training_data/curriculum',
                        help='Directory containing curriculum-organized games')
    parser.add_argument('--curriculum-config', type=str, default=None,
                        help='Path to curriculum configuration JSON file')
    parser.add_argument('--curriculum-state', type=str, default=None,
                        help='Path to saved curriculum state for resuming')
    parser.add_argument('--dynamic-value-weight', action='store_true',
                        help='Use dynamic value loss weighting based on curriculum stage')
    
    # Advanced training
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience in epochs (default: 10, 0 to disable)')
    parser.add_argument('--ema-decay', type=float, default=0,
                        help='Exponential moving average decay (default: 0, disabled)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs/alphazero',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/alphazero',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log training metrics every N batches (default: 100)')
    parser.add_argument('--monitor-piece-values', action='store_true',
                        help='Monitor piece value convergence during training')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def create_data_loaders(args, device):
    """Create training and validation data loaders."""
    
    # Create dataset based on training mode
    if args.mode == 'supervised':
        print(f'Training mode: Supervised learning on CCRL dataset')
        print(f'CCRL directory: {args.ccrl_dir}')
        dataset = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp)
    elif args.mode == 'rl':
        print(f'Training mode: Reinforcement learning on self-play data')
        print(f'RL directory: {args.rl_dir}')
        if args.rl_weight_recent:
            print(f'Using weighted sampling with decay factor {args.rl_weight_decay}')
        dataset = RLDataset(args.rl_dir, weight_recent=args.rl_weight_recent, 
                          weight_decay=args.rl_weight_decay)
    elif args.mode == 'curriculum':
        print(f'Training mode: Curriculum learning (progressive difficulty)')
        print(f'Curriculum directory: {args.curriculum_dir}')
        dataset = CurriculumDataset(args.curriculum_config)
        
        # Load saved state if resuming
        if args.curriculum_state and os.path.exists(args.curriculum_state):
            dataset.load_state(args.curriculum_state)
        
        # Return curriculum dataset directly for special handling
        return dataset, None
    elif args.mode == 'mixed-curriculum':
        print(f'Training mode: Mixed curriculum learning')
        print(f'Curriculum directory: {args.curriculum_dir}')
        dataset = MixedCurriculumDataset(args.curriculum_config)
    else:  # mixed mode
        print(f'Training mode: Mixed (RL ratio: {args.mixed_ratio})')
        print(f'Using soft targets with temperature {args.label_smoothing_temp}')
        ccrl_ds = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp)
        rl_ds = RLDataset(args.rl_dir)
        
        # Calculate sizes for balanced sampling
        ccrl_size = int(len(ccrl_ds) * (1 - args.mixed_ratio))
        rl_size = int(len(rl_ds) * args.mixed_ratio)
        
        # Create subset indices
        import random
        ccrl_indices = random.sample(range(len(ccrl_ds)), min(ccrl_size, len(ccrl_ds)))
        rl_indices = random.sample(range(len(rl_ds)), min(rl_size, len(rl_ds)))
        
        # Create subsets
        ccrl_subset = Subset(ccrl_ds, ccrl_indices)
        rl_subset = Subset(rl_ds, rl_indices)
        
        # Concatenate datasets
        dataset = ConcatDataset([ccrl_subset, rl_subset])
        print(f'Dataset sizes - CCRL: {len(ccrl_subset)}, RL: {len(rl_subset)}')
    
    # Split into train and validation
    total_size = len(dataset)
    val_size = int(total_size * args.validation_split)
    train_size = total_size - val_size
    
    # Use random_split for proper train/val separation
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    print(f'Dataset split - Train: {len(train_dataset)}, Validation: {len(val_dataset)}')
    
    # Get optimal batch size and workers
    batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
    num_workers = get_num_workers_for_device()
    
    # Create data loaders
    if args.mode == 'rl' and args.rl_weight_recent and not isinstance(dataset, ConcatDataset):
        # Use weighted sampler for RL mode with recent game weighting
        train_sampler = WeightedRLSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
    else:
        # Standard shuffled data loader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, scaler, args, epoch, writer, device, ema=None):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, data in enumerate(train_loader):
        # Move data to device
        position = data['position'].to(device)
        valueTarget = data['value'].to(device)
        policyTarget = data['policy'].to(device)
        
        # Mixed precision training
        if args.mixed_precision and device.type == 'cuda':
            with autocast():
                loss, valueLoss, policyLoss = model(position, valueTarget=valueTarget,
                                                    policyTarget=policyTarget)
                loss = loss / args.gradient_accumulation
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler and args.scheduler == 'onecycle':
                    scheduler.step()
                
                if ema:
                    ema.update()
        else:
            loss, valueLoss, policyLoss = model(position, valueTarget=valueTarget,
                                               policyTarget=policyTarget)
            loss = loss / args.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler and args.scheduler == 'onecycle':
                    scheduler.step()
                
                if ema:
                    ema.update()
        
        # Track losses
        total_loss += loss.item() * args.gradient_accumulation
        total_value_loss += valueLoss.item()
        total_policy_loss += policyLoss.item()
        num_batches += 1
        
        # Log to tensorboard
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            if writer:
                writer.add_scalar('Train/Loss', loss.item() * args.gradient_accumulation, global_step)
                writer.add_scalar('Train/ValueLoss', valueLoss.item(), global_step)
                writer.add_scalar('Train/PolicyLoss', policyLoss.item(), global_step)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            if args.verbose:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * train_loader.batch_size / elapsed
                print(f'Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item() * args.gradient_accumulation:.4f} '
                      f'Value: {valueLoss.item():.4f} '
                      f'Policy: {policyLoss.item():.4f} '
                      f'Samples/s: {samples_per_sec:.1f}')
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    
    return avg_loss, avg_value_loss, avg_policy_loss


def validate(model, val_loader, args, epoch, writer, device):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in val_loader:
            # Move data to device
            position = data['position'].to(device)
            valueTarget = data['value'].to(device)
            policyTarget = data['policy'].to(device)
            
            # Forward pass - need to temporarily switch to train mode to get losses
            model.train()
            loss, valueLoss, policyLoss = model(position, valueTarget=valueTarget,
                                               policyTarget=policyTarget)
            model.eval()
            
            total_loss += loss.item()
            total_value_loss += valueLoss.item()
            total_policy_loss += policyLoss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/ValueLoss', avg_value_loss, epoch)
        writer.add_scalar('Val/PolicyLoss', avg_policy_loss, epoch)
    
    return avg_loss, avg_value_loss, avg_policy_loss


def save_checkpoint(model, optimizer, scheduler, epoch, args, best_val_loss, checkpoint_path, ema=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args),
        'model_config': {
            'num_blocks': args.num_blocks,
            'num_filters': args.num_filters,
            'policy_weight': args.policy_weight
        }
    }
    
    if ema:
        checkpoint['ema_state_dict'] = ema.shadow
    
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')


def train_curriculum(args, device):
    """Special training loop for curriculum learning."""
    # Setup directories and logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_subdir = os.path.join(args.log_dir, f'curriculum_{timestamp}')
    writer = SummaryWriter(log_subdir)
    print(f'Tensorboard logs: {log_subdir}')
    
    # Create curriculum dataset
    curriculum_dataset = CurriculumDataset(args.curriculum_config)
    
    # Load saved state if resuming
    if args.curriculum_state and os.path.exists(args.curriculum_state):
        curriculum_dataset.load_state(args.curriculum_state)
    
    # Determine model architecture
    num_blocks = args.num_blocks
    num_filters = args.num_filters
    
    # Create and optimize model for the device
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters, policy_weight=args.policy_weight)
    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in alphaZeroNet.parameters())
    trainable_params = sum(p.numel() for p in alphaZeroNet.parameters() if p.requires_grad)
    print(f'Model initialized: {num_blocks}x{num_filters}')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Initialize optimizer
    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.lr)
    
    # Initialize scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        print('Using ReduceLROnPlateau scheduler')
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print('Mixed precision training enabled')
    
    # Initialize EMA
    ema = EMA(alphaZeroNet, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f'Exponential moving average enabled with decay {args.ema_decay}')
    
    # Load checkpoint if resuming
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            alphaZeroNet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if ema and checkpoint.get('ema_state_dict'):
                ema.shadow = checkpoint['ema_state_dict']
            print(f'Resumed from checkpoint')
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    global_epoch = 0
    
    # Training loop through curriculum stages
    while curriculum_dataset.current_stage_idx < len(curriculum_dataset.stages):
        current_stage = curriculum_dataset.get_current_stage()
        stage_info = curriculum_dataset.get_stage_info()
        
        print(f"\n{'='*60}")
        print(f"Starting Stage: {current_stage.name.upper()}")
        print(f"ELO Range: {current_stage.elo_range[0]}-{current_stage.elo_range[1]}")
        print(f"Epochs: {current_stage.epochs}")
        print(f"Value Weight: {current_stage.value_weight}")
        print(f"{'='*60}\n")
        
        # Update model's policy weight if using dynamic weighting
        if args.dynamic_value_weight:
            alphaZeroNet.policy_weight = 1.0 / current_stage.value_weight
            print(f"Updated model value/policy weight ratio: {current_stage.value_weight}:{alphaZeroNet.policy_weight}")
        
        # Create data loaders for current stage
        dataset = curriculum_dataset.current_dataset
        if dataset is None:
            curriculum_dataset._load_current_stage()
            dataset = curriculum_dataset.current_dataset
        
        # Split into train and validation
        total_size = len(dataset)
        val_size = int(total_size * args.validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
        num_workers = get_num_workers_for_device()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
        
        print(f'Stage dataset - Train: {len(train_dataset)}, Validation: {len(val_dataset)}')
        
        # Train for the specified number of epochs for this stage
        for stage_epoch in range(current_stage.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_value_loss, train_policy_loss = train_epoch(
                alphaZeroNet, train_loader, optimizer, None, scaler, 
                args, global_epoch, writer, device, ema
            )
            
            # Validate
            val_loss, val_value_loss, val_policy_loss = validate(
                alphaZeroNet, val_loader, args, global_epoch, writer, device
            )
            
            # Step scheduler
            if scheduler:
                scheduler.step(val_loss)
            
            # Log curriculum info
            if writer:
                writer.add_scalar('Curriculum/Stage', curriculum_dataset.current_stage_idx, global_epoch)
                writer.add_scalar('Curriculum/ValueWeight', current_stage.value_weight, global_epoch)
                writer.add_scalar('Curriculum/ELOMin', current_stage.elo_range[0], global_epoch)
                writer.add_scalar('Curriculum/ELOMax', current_stage.elo_range[1], global_epoch)
            
            # Monitor piece values if requested
            if args.monitor_piece_values and (global_epoch + 1) % 5 == 0:
                piece_monitor = PieceValueMonitor(alphaZeroNet, device)
                piece_monitor.print_piece_value_report()
                if writer:
                    piece_monitor.log_to_tensorboard(writer, global_epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'\n[{current_stage.name}] Epoch {stage_epoch+1}/{current_stage.epochs} (Global: {global_epoch+1}) - Time: {epoch_time:.1f}s')
            print(f'Train - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint periodically
            if (global_epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'curriculum_checkpoint_epoch_{global_epoch+1}.pt')
                save_checkpoint(alphaZeroNet, optimizer, scheduler, global_epoch, args, 
                              best_val_loss, checkpoint_path, ema)
                
                # Save curriculum state
                curriculum_state_path = os.path.join(args.checkpoint_dir, f'curriculum_state_{global_epoch+1}.json')
                curriculum_dataset.save_state(curriculum_state_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, 'curriculum_best_model.pt')
                save_checkpoint(alphaZeroNet, optimizer, scheduler, global_epoch, args, 
                              best_val_loss, best_path, ema)
                print(f'New best validation loss: {best_val_loss:.4f}')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping (per stage)
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print(f'\nEarly stopping triggered for stage {current_stage.name} after {stage_epoch+1} epochs')
                break
            
            curriculum_dataset.on_epoch_end()
            global_epoch += 1
            
            print('-' * 60)
        
        # Advance to next stage
        if not curriculum_dataset.advance_stage():
            print(f"\n{'='*60}")
            print("Curriculum training complete!")
            print(f"{'='*60}")
            break
        
        # Reset early stopping counter for new stage
        early_stopping_counter = 0
    
    # Apply EMA if enabled
    if ema:
        print('\nApplying EMA weights to final model...')
        ema.apply_shadow()
    
    # Save final model
    if args.output:
        networkFileName = args.output
    else:
        networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}_curriculum.pt'
    
    torch.save(alphaZeroNet.state_dict(), networkFileName)
    print(f'\nCurriculum training complete! Final model saved to {networkFileName}')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    # Save training summary
    summary = {
        'model': f'{num_blocks}x{num_filters}',
        'training_mode': 'curriculum',
        'stages_completed': curriculum_dataset.current_stage_idx,
        'total_stages': len(curriculum_dataset.stages),
        'epochs_trained': global_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'args': vars(args)
    }
    
    summary_path = os.path.join(args.checkpoint_dir, 'curriculum_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Training summary saved to {summary_path}')
    
    # Close tensorboard writer
    writer.close()


def train():
    args = parse_args()
    
    # Get optimal device and configure for training
    device, device_str = get_optimal_device()
    print(f'Using device: {device_str}')
    
    # Handle curriculum training separately
    if args.mode == 'curriculum':
        train_curriculum(args, device)
        return
    
    # Setup directories and logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_subdir = os.path.join(args.log_dir, f'run_{timestamp}')
    writer = SummaryWriter(log_subdir)
    print(f'Tensorboard logs: {log_subdir}')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, device)
    
    # Adjust batch size if specified
    batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
    print(f'Batch size: {batch_size}, Gradient accumulation: {args.gradient_accumulation}')
    print(f'Effective batch size: {batch_size * args.gradient_accumulation}')
    
    # Determine model architecture
    num_blocks = args.num_blocks
    num_filters = args.num_filters
    
    # If resuming, try to extract architecture from filename
    if args.resume:
        match = re.search(r'AlphaZeroNet_(\d+)x(\d+)', args.resume)
        if match:
            file_blocks = int(match.group(1))
            file_filters = int(match.group(2))
            if num_blocks != file_blocks or num_filters != file_filters:
                print(f'Warning: Architecture mismatch! File suggests {file_blocks}x{file_filters}, '
                      f'but using {num_blocks}x{num_filters}')
    
    # Create and optimize model for the device
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters, policy_weight=args.policy_weight)
    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in alphaZeroNet.parameters())
    trainable_params = sum(p.numel() for p in alphaZeroNet.parameters() if p.requires_grad)
    print(f'Model initialized: {num_blocks}x{num_filters}')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Initialize optimizer
    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.lr)
    
    # Initialize scheduler
    scheduler = None
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr * 10,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        print(f'Using OneCycleLR scheduler with {args.warmup_epochs} warmup epochs')
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        print('Using CosineAnnealingWarmRestarts scheduler')
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        print('Using ReduceLROnPlateau scheduler')
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print('Mixed precision training enabled')
    
    # Initialize EMA
    ema = EMA(alphaZeroNet, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f'Exponential moving average enabled with decay {args.ema_decay}')
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state
            alphaZeroNet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Load EMA state if available
            if ema and checkpoint.get('ema_state_dict'):
                ema.shadow = checkpoint['ema_state_dict']
            
            print(f'Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}')
        else:
            raise FileNotFoundError(f'Checkpoint file not found: {args.resume}')
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print('=' * 60)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_value_loss, train_policy_loss = train_epoch(
            alphaZeroNet, train_loader, optimizer, scheduler, scaler, 
            args, epoch, writer, device, ema
        )
        
        # Validate
        val_loss, val_value_loss, val_policy_loss = validate(
            alphaZeroNet, val_loader, args, epoch, writer, device
        )
        

        piece_monitor = PieceValueMonitor(alphaZeroNet, device)
        piece_monitor.print_piece_value_report()

        # Step schedulers that need validation loss
        if args.scheduler == 'plateau' and scheduler:
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine' and scheduler:
            scheduler.step()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{args.epochs} - Time: {epoch_time:.1f}s')
        print(f'Train - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(alphaZeroNet, optimizer, scheduler, epoch, args, 
                          best_val_loss, checkpoint_path, ema)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(alphaZeroNet, optimizer, scheduler, epoch, args, 
                          best_val_loss, best_path, ema)
            print(f'New best validation loss: {best_val_loss:.4f}')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation loss: {best_val_loss:.4f}')
            break
        
        # Save current model
        networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}_latest.pt'
        torch.save(alphaZeroNet.state_dict(), networkFileName)
        
        print('-' * 60)
    
    # Apply EMA if enabled
    if ema:
        print('\nApplying EMA weights to final model...')
        ema.apply_shadow()
    
    # Save final model
    if args.output:
        networkFileName = args.output
    else:
        networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}.pt'
    
    torch.save(alphaZeroNet.state_dict(), networkFileName)
    print(f'\nTraining complete! Final model saved to {networkFileName}')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    # Save training summary
    summary = {
        'model': f'{num_blocks}x{num_filters}',
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'training_time': time.time() - epoch_start_time * (epoch + 1 - start_epoch),
        'args': vars(args)
    }
    
    summary_path = os.path.join(args.checkpoint_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Training summary saved to {summary_path}')
    
    # Close tensorboard writer
    writer.close()


if __name__ == '__main__':
    train()