import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from TitanMiniNetwork import TitanMini, count_parameters
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import time
import json
import contextlib
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import signal


# Training params (defaults)
default_epochs = 500
default_num_layers = 13
default_d_model = 512
default_num_heads = 8
default_d_ff = 1920
default_lr = 0.0001  # Lower learning rate for more stable training
default_warmup_epochs = 5  # Shorter warmup with lower initial LR
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Titan-Mini transformer model')
    
    # Model architecture
    parser.add_argument('--num-layers', type=int, default=default_num_layers,
                        help=f'Number of transformer layers (default: {default_num_layers})')
    parser.add_argument('--d-model', type=int, default=default_d_model,
                        help=f'Model dimension (default: {default_d_model})')
    parser.add_argument('--num-heads', type=int, default=default_num_heads,
                        help=f'Number of attention heads (default: {default_num_heads})')
    parser.add_argument('--d-ff', type=int, default=default_d_ff,
                        help=f'Feedforward dimension (default: {default_d_ff})')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    
    # Training params
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate (default: {default_lr})')
    parser.add_argument('--warmup-epochs', type=int, default=default_warmup_epochs,
                        help=f'Number of warmup epochs (default: {default_warmup_epochs})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: auto-detect based on device)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping value (default: 10.0)')
    
    # Model params
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for saved model')
    parser.add_argument('--policy-weight', type=float, default=default_policy_weight,
                        help=f'Weight for policy loss relative to value loss (default: {default_policy_weight})')
    parser.add_argument('--input-planes', type=int, default=112,
                        help='Number of input planes (16 for classic, 112 for enhanced encoder)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Use a tiny synthetic dataset for a fast training dry-run')
    parser.add_argument('--dry-samples', type=int, default=64,
                        help='Number of synthetic samples to generate for dry-run (default: 64)')
    
    # Data params
    parser.add_argument('--mode', choices=['supervised', 'rl', 'mixed'], default='supervised',
                        help='Training mode: supervised (CCRL data), rl (self-play), or mixed (both)')
    parser.add_argument('--ccrl-dir', type=str, default=ccrl_dir,
                        help='Directory containing CCRL training data')
    parser.add_argument('--rl-dir', type=str, default=rl_dir,
                        help='Directory containing self-play training data (HDF5 files)')
    parser.add_argument('--mixed-ratio', type=float, default=0.7,
                        help='For mixed mode: ratio of RL data (0.0-1.0, default: 0.7)')
    parser.add_argument('--label-smoothing-temp', type=float, default=0.15,
                        help='Temperature for label smoothing (default: 0.15)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    
    # Advanced training
    parser.add_argument('--swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa-start', type=int, default=75,
                        help='Epoch to start SWA (default: 75)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training (requires PyTorch 2.0+)')
    
    # Distributed
    parser.add_argument('--distributed', action='store_true',
                        help='Enable multi-GPU DistributedDataParallel (torchrun recommended)')
    parser.add_argument('--dist-backend', type=str, default=None,
                        help="DDP backend (default auto: 'nccl' for CUDA)")
    parser.add_argument('--batch-size-total', type=int, default=None,
                        help='Global batch size across ranks (if set, per-rank batch = total/world_size)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs/titan_mini',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/titan_mini',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
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


class SyntheticTitanDataset(Dataset):
    """Tiny synthetic dataset for quick dry-runs."""
    def __init__(self, num_samples: int, input_planes: int):
        self.n = max(1, int(num_samples))
        self.c = int(input_planes)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        position = torch.randn(self.c, 8, 8)
        # Now using Tanh activation, so values should be in [-1, 1] range
        # -1 = loss, 0 = draw, 1 = win (from current player's perspective)
        value = torch.rand(1).item() * 2 - 1  # [-1, 1] range for Tanh
        policy = torch.randint(0, 72 * 64, (1,), dtype=torch.long).squeeze(0)
        mask = torch.ones(72, 8, 8, dtype=torch.int64)  # allow all moves for simplicity
        return position, torch.tensor(value, dtype=torch.float32), policy, mask


def create_data_loaders(args, *, distributed=False, rank=0, world_size=1, is_main=True, device=None):
    """Create training and validation data loaders."""
    # Detect device for DataLoader tuning (pin_memory and default batch size)
    if device is None:
        device, _ = get_optimal_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Dry-run synthetic dataset path
    if args.dry_run:
        print('Dry-run mode: using synthetic random dataset')
        n_train = args.dry_samples
        n_val = max(1, n_train // 4)
        train_dataset = SyntheticTitanDataset(n_train, args.input_planes)
        val_dataset = SyntheticTitanDataset(n_val, args.input_planes)

        batch_size = args.batch_size if args.batch_size else (2 if device.type == 'mps' else 4)
        num_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

        print(f'Training samples (dry): {len(train_dataset)}, Validation samples (dry): {len(val_dataset)}')
        return train_loader, val_loader

    # Create dataset based on training mode
    if args.mode == 'supervised':
        print(f'Training mode: Supervised learning on CCRL dataset')
        enhanced = args.input_planes > 16
        dataset = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp, enhanced_encoder=enhanced)
    elif args.mode == 'rl':
        print('Training mode: Reinforcement learning on self-play data')
        dataset = RLDataset(args.rl_dir)
    else:  # mixed mode
        print(f'Training mode: Mixed (RL ratio: {args.mixed_ratio})')
        enhanced = args.input_planes > 16
        ccrl_ds = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp, enhanced_encoder=enhanced)
        rl_ds = RLDataset(args.rl_dir)

        # Calculate sizes for balanced sampling
        total_size = max(len(ccrl_ds), len(rl_ds))
        rl_size = int(total_size * args.mixed_ratio)
        ccrl_size = total_size - rl_size

        # Create subsets with replacement if necessary
        ccrl_indices = np.random.choice(len(ccrl_ds), ccrl_size, replace=True)
        rl_indices = np.random.choice(len(rl_ds), rl_size, replace=True)

        ccrl_subset = Subset(ccrl_ds, ccrl_indices)
        rl_subset = Subset(rl_ds, rl_indices)

        dataset = ConcatDataset([ccrl_subset, rl_subset])
        print(f'Dataset sizes - CCRL: {len(ccrl_subset)}, RL: {len(rl_subset)}')

    # Split into train and validation
    val_size = int(len(dataset) * args.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if is_main:
        print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    # Get optimal batch size and workers
    if args.batch_size_total is not None and distributed and world_size > 1:
        batch_size = max(1, args.batch_size_total // world_size)
    elif args.batch_size is not None:
        batch_size = args.batch_size
    else:
        # Titan-Mini is smaller, so we can use larger batch sizes
        batch_size = 128 if device.type == 'mps' else (get_batch_size_for_device() // 4)
    num_workers = get_num_workers_for_device()

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed and world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed and world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size * 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, scaler, args, epoch, writer, ema=None, *, is_main=True, distributed=False):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Support both dict-style and tuple batches
        if isinstance(batch, dict):
            position = batch['position']
            value_target = batch['value']
            policy_target = batch['policy']
            policy_mask = batch.get('mask')
        else:
            position, value_target, policy_target, policy_mask = batch

        # Adapt plane count if needed
        if position.dim() == 4 and position.size(1) != args.input_planes:
            c = position.size(1)
            if c < args.input_planes:
                pad = torch.zeros(position.size(0), args.input_planes - c, 8, 8, dtype=position.dtype)
                position = torch.cat([position, pad], dim=1)
            else:
                position = position[:, :args.input_planes]

        # Move to device
        device = next(model.parameters()).device
        position = position.to(device)
        # Ensure value target shape is [batch, 1] (avoid extra singleton dim)
        value_target = value_target.to(device).float()
        if value_target.dim() == 1:
            value_target = value_target.view(-1, 1)
        elif value_target.dim() == 2 and value_target.size(1) != 1:
            value_target = value_target[:, :1]
        
        # Assert value targets are in expected [-1, 1] range for Tanh activation
        assert value_target.min() >= -1 and value_target.max() <= 1, \
            f"Value targets out of range [-1, 1]: min={value_target.min().item():.4f}, max={value_target.max().item():.4f}"
        
        policy_target = policy_target.to(device)
        if policy_mask is not None:
            policy_mask = policy_mask.to(device)
        
        # Mixed precision training (supports CUDA and MPS)
        if args.mixed_precision:
            use_amp = hasattr(torch, 'autocast') and device.type in ('cuda', 'mps')
            amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                loss, value_loss, policy_loss = model(position, value_target, policy_target, policy_mask)
                loss = loss / args.gradient_accumulation

            if scaler is not None:  # CUDA path with GradScaler
                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    # Use more aggressive gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    if ema is not None:
                        ema.update()
            else:  # MPS/CPU autocast without scaler
                loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    # Use more aggressive gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    if ema is not None:
                        ema.update()
        else:
            loss, value_loss, policy_loss = model(position, value_target, policy_target, policy_mask)
            loss = loss / args.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Use more aggressive gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                if ema is not None:
                    ema.update()
        
        # Track losses
        total_loss += loss.item() * args.gradient_accumulation
        total_value_loss += value_loss.item()
        total_policy_loss += policy_loss.item()
        num_batches += 1
        
        # Log to tensorboard
        if is_main and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/ValueLoss', value_loss.item(), global_step)
                writer.add_scalar('Train/PolicyLoss', policy_loss.item(), global_step)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                
                # Monitor value distribution to detect saturation
                with torch.no_grad():
                    model_module = model.module if hasattr(model, 'module') else model
                    if hasattr(model_module, 'value_head'):
                        # Get raw value outputs (after Tanh)
                        model_module.eval()
                        test_value, _ = model_module(position[:min(4, position.size(0))])
                        model_module.train()
                        writer.add_histogram('Train/ValueOutputs', test_value, global_step)
                        writer.add_scalar('Train/ValueMean', test_value.mean().item(), global_step)
                        writer.add_scalar('Train/ValueStd', test_value.std().item(), global_step)
            
            elapsed = time.time() - start_time
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Value: {value_loss.item():.4f} '
                  f'Policy: {policy_loss.item():.4f} '
                  f'Time: {elapsed:.1f}s')
    
    avg_loss = total_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    
    return avg_loss, avg_value_loss, avg_policy_loss


def validate(model, val_loader, args, epoch, writer, *, device=None, distributed=False):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                position = batch['position']
                value_target = batch['value']
                policy_target = batch['policy']
                policy_mask = batch.get('mask')
            else:
                position, value_target, policy_target, policy_mask = batch

            # Adapt plane count if needed
            if position.dim() == 4 and position.size(1) != args.input_planes:
                c = position.size(1)
                if c < args.input_planes:
                    pad = torch.zeros(position.size(0), args.input_planes - c, 8, 8, dtype=position.dtype)
                    position = torch.cat([position, pad], dim=1)
                else:
                    position = position[:, :args.input_planes]

            # Move to device
            device = next(model.parameters()).device
            position = position.to(device)
            value_target = value_target.to(device).float()
            if value_target.dim() == 1:
                value_target = value_target.view(-1, 1)
            elif value_target.dim() == 2 and value_target.size(1) != 1:
                value_target = value_target[:, :1]
            
            # Assert value targets are in expected [-1, 1] range for validation too
            assert value_target.min() >= -1 and value_target.max() <= 1, \
                f"Validation value targets out of range [-1, 1]: min={value_target.min().item():.4f}, max={value_target.max().item():.4f}"
            
            policy_target = policy_target.to(device)
            if policy_mask is not None:
                policy_mask = policy_mask.to(device)

            # Forward pass - model returns (value, policy) in eval mode
            # We need to switch to training mode temporarily to get losses
            model.train()
            loss, value_loss, policy_loss = model(position, value_target, policy_target, policy_mask)
            model.eval()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1

    if distributed and dist.is_initialized():
        totals = torch.tensor([total_loss, total_value_loss, total_policy_loss, float(num_batches)], device=device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        avg_loss = (totals[0] / totals[3]).item()
        avg_value_loss = (totals[1] / totals[3]).item()
        avg_policy_loss = (totals[2] / totals[3]).item()
    else:
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches

    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/ValueLoss', avg_value_loss, epoch)
        writer.add_scalar('Val/PolicyLoss', avg_policy_loss, epoch)
    
    return avg_loss, avg_value_loss, avg_policy_loss


def save_checkpoint(model, optimizer, scheduler, epoch, args, best_val_loss, checkpoint_path):
    """Save model checkpoint with full configuration metadata."""
    # Get the actual model config from the model itself
    model_instance = model if not isinstance(model, DDP) else model.module
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args),
        'model_config': {
            'num_layers': args.num_layers,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'policy_weight': args.policy_weight,
            'input_planes': args.input_planes,
            'use_wdl': getattr(model_instance, 'use_wdl', True),
            'legacy_value_head': False  # Always False for new training
        }
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')


def train():
    args = parse_args()
    # DDP setup
    env_world = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = args.distributed or env_world > 1

    rank = 0
    local_rank = 0
    world_size = 1
    if distributed:
        try:
            backend = args.dist_backend or ('nccl' if torch.cuda.is_available() else 'gloo')
            if not dist.is_initialized():
                # Set timeout for initialization
                import datetime
                timeout = datetime.timedelta(seconds=300)
                dist.init_process_group(backend=backend, init_method='env://', timeout=timeout)
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device('cuda', local_rank)
                device_str = f'CUDA GPU {local_rank} (rank {rank}/{world_size})'
            else:
                device, device_str = get_optimal_device()
        except Exception as e:
            print(f"Error initializing distributed training: {e}")
            print("Falling back to single GPU/CPU training")
            distributed = False
            device, device_str = get_optimal_device()
    else:
        device, device_str = get_optimal_device()
    is_main = (rank == 0)
    if is_main:
        print(f'Using device: {device_str}')

    # Setup graceful shutdown on terminal/daemon signals
    def _graceful_exit(signum, frame):
        try:
            if dist.is_available() and dist.is_initialized():
                if is_main:
                    print(f"Received signal {signum}; destroying process group...")
                dist.destroy_process_group()
        except Exception:
            pass
        finally:
            os._exit(0)

    for _sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _graceful_exit)
        except Exception:
            pass

    # directories and writer
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir) if is_main else None

    # model
    model = TitanMini(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        policy_weight=args.policy_weight,
        input_planes=args.input_planes,
        use_wdl=True,  # Use Win-Draw-Loss value head for richer signal
        legacy_value_head=False,  # Use the modern 3-layer architecture
    )
    model = model.to(device)
    if is_main:
        print(f'Titan-Mini initialized with {count_parameters(model):,} parameters')
        print(f'Model size: ~{count_parameters(model) * 4 / (1024 * 1024):.2f} MB')

    # compile and optimize
    if args.compile and hasattr(torch, 'compile'):
        if is_main:
            print('Compiling model with torch.compile()...')
        model = torch.compile(model)
    model = optimize_for_device(model, device)

    # DDP wrap
    if distributed and world_size > 1:
        if torch.cuda.is_available():
            # Ensure all processes are ready before wrapping
            if dist.is_initialized():
                dist.barrier()
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,  # Set to False for better performance
                broadcast_buffers=True,
            )
        else:
            model = DDP(model, find_unused_parameters=False)

    # data
    if is_main:
        print(f"Creating data loaders...")
        print(f"Mode: {args.mode}")
        print(f"CCRL dir: {args.ccrl_dir} (exists: {os.path.exists(args.ccrl_dir)})")
        print(f"RL dir: {args.rl_dir} (exists: {os.path.exists(args.rl_dir)})")
    
    # Synchronize before data loading
    if distributed and dist.is_initialized():
        dist.barrier()
        if is_main:
            print("All processes synchronized, loading data...")
    
    train_loader, val_loader = create_data_loaders(args, distributed=distributed, rank=rank, world_size=world_size, is_main=is_main, device=device)

    # optimizer and scheduler
    # Use stronger weight decay to prevent parameter drift
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    try:
        pct = warmup_steps / total_steps if total_steps > 0 else 0.1
        if not (0.0 < pct < 1.0):
            raise ValueError('Invalid pct_start for OneCycleLR')
        scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=total_steps, pct_start=pct,
                               anneal_strategy='cos', div_factor=10, final_div_factor=100)
    except Exception:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=max(1, len(train_loader)//2), gamma=0.1)

    # Use new torch.amp.GradScaler API when available (silence deprecation)
    if args.mixed_precision and torch.cuda.is_available():
        try:
            from torch.amp import GradScaler as AmpGradScaler  # PyTorch 2.1+
            scaler = AmpGradScaler('cuda')
        except Exception:
            from torch.cuda.amp import GradScaler as CudaGradScaler
            scaler = CudaGradScaler()
    else:
        scaler = None
    ema = EMA(model, decay=0.9999) if args.swa and is_main else None

    # resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        # In case of DDP, model may be wrapped
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if is_main:
            print(f'Resumed from checkpoint: {args.resume} (epoch {start_epoch})')

    # training loop
    if is_main:
        print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if is_main:
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_value_loss, train_policy_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args, epoch, writer, ema, is_main=is_main, distributed=distributed
        )
        if is_main:
            print(f'Train - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}')

        val_loss, val_value_loss, val_policy_loss = validate(
            model, val_loader, args, epoch, writer, device=device, distributed=distributed
        )
        if is_main:
            print(f'Val - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}')

        if is_main and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'titan_mini_epoch_{epoch + 1}.pt')
            save_checkpoint(model if not isinstance(model, DDP) else model.module, optimizer, scheduler, epoch, args, best_val_loss, checkpoint_path)

        if is_main and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'titan_mini_best.pt')
            save_checkpoint(model if not isinstance(model, DDP) else model.module, optimizer, scheduler, epoch, args, best_val_loss, best_path)
            print(f'New best validation loss: {best_val_loss:.4f}')

        if args.swa and epoch >= args.swa_start and ema and is_main:
            ema.apply_shadow()
            swa_val_loss, _, _ = validate(model, val_loader, args, epoch, writer, device=device, distributed=distributed)
            print(f'SWA Val Loss: {swa_val_loss:.4f}')
            if swa_val_loss < best_val_loss:
                best_val_loss = swa_val_loss
                swa_path = os.path.join(args.checkpoint_dir, 'titan_mini_swa.pt')
                save_checkpoint(model if not isinstance(model, DDP) else model.module, optimizer, scheduler, epoch, args, best_val_loss, swa_path)
                print(f'New best SWA validation loss: {best_val_loss:.4f}')
            ema.restore()

    if is_main:
        output_path = args.output or os.path.join('weights', f'TitanMini_{args.num_layers}x{args.d_model}.pt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if ema:
            ema.apply_shadow()
        
        model_instance = model if not isinstance(model, DDP) else model.module
        
        # Save with full configuration metadata
        final_checkpoint = {
            'model_state_dict': model_instance.state_dict(),
            'model_config': {
                'num_layers': args.num_layers,
                'd_model': args.d_model,
                'num_heads': args.num_heads,
                'd_ff': args.d_ff,
                'dropout': args.dropout,
                'policy_weight': args.policy_weight,
                'input_planes': args.input_planes,
                'use_wdl': getattr(model_instance, 'use_wdl', True),
                'legacy_value_head': False
            },
            'args': vars(args)
        }
        torch.save(final_checkpoint, output_path)
        print(f'\nTraining complete! Model saved to {output_path} with config metadata')
        if writer is not None:
            writer.close()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
