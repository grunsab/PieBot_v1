# train_titan_mini.py (Revised)
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Imports assumed to be correct based on the user's environment
from CCRLDataset import CCRLDataset
try:
    # Optional imports for RL/Curriculum
    from RLDataset import RLDataset, WeightedRLSampler
    from TitanCurriculumDataset import TitanCurriculumDataset, TitanMixedCurriculumDataset
except ImportError:
    # Define dummy classes if imports fail (for robustness)
    class RLDataset(Dataset):
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
    class TitanCurriculumDataset(Dataset): pass
    class TitanMixedCurriculumDataset(Dataset): pass
    print("Note: RLDataset or TitanCurriculumDataset not found. Modes 'rl', 'curriculum' might fail.")

from titan_piece_value_monitor import TitanPieceValueMonitor
# Import the corrected TitanMiniNetwork
from TitanMiniNetwork import TitanMini, count_parameters
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import time
import json
import contextlib
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# Use standard GradScaler/autocast imports
from torch.cuda.amp import GradScaler, autocast
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import signal


# Training params (defaults remain the same)
default_epochs = 500
default_num_layers = 13
default_d_model = 512
default_num_heads = 8
default_d_ff = 1920
# Increased default LR slightly as auxiliary losses provide stabilization
default_lr = 0.0002 
default_warmup_epochs = 5
default_policy_weight = 1.0

# Directory handling (robustness)
try:
    ccrl_dir = os.path.abspath('games_training_data/reformatted/')
    rl_dir = os.path.abspath('games_training_data/selfplay/')
except Exception:
    ccrl_dir = 'games_training_data/reformatted/'
    rl_dir = 'games_training_data/selfplay/'


def parse_args():
    # (Implementation remains largely the same, adjusted defaults)
    parser = argparse.ArgumentParser(description='Train Titan-Mini transformer model')
    
    # Model architecture
    parser.add_argument('--num-layers', type=int, default=default_num_layers)
    parser.add_argument('--d-model', type=int, default=default_d_model)
    parser.add_argument('--num-heads', type=int, default=default_num_heads)
    parser.add_argument('--d-ff', type=int, default=default_d_ff)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training params
    parser.add_argument('--resume', type=str)
    parser.add_argument('--epochs', type=int, default=default_epochs)
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--warmup-epochs', type=int, default=default_warmup_epochs)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    # Set default grad clip to 1.0 for stability
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0)')
    
    # Model params
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--policy-weight', type=float, default=default_policy_weight)
    parser.add_argument('--input-planes', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dry-samples', type=int, default=64)
    
    # Data params (label smoothing temp default adjusted slightly)
    parser.add_argument('--mode', choices=['supervised', 'rl', 'mixed', 'curriculum', 'mixed-curriculum'], default='supervised')
    parser.add_argument('--ccrl-dir', type=str, default=ccrl_dir)
    parser.add_argument('--rl-dir', type=str, default=rl_dir)
    parser.add_argument('--mixed-ratio', type=float, default=0.7)
    parser.add_argument('--label-smoothing-temp', type=float, default=0.10)
    parser.add_argument('--validation-split', type=float, default=0.1)
    
    # Curriculum learning params
    parser.add_argument('--curriculum-dir', type=str, default='games_training_data/curriculum')
    parser.add_argument('--curriculum-config', type=str, default=None)
    parser.add_argument('--curriculum-state', type=str, default=None)
    parser.add_argument('--dynamic-value-weight', action='store_true')
    parser.add_argument('--monitor-piece-values', action='store_true')
    
    # Advanced training
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa-start', type=int, default=75)
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--compile', action='store_true')
    
    # Distributed
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist-backend', type=str, default=None)
    parser.add_argument('--batch-size-total', type=int, default=None)
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs/titan_mini')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/titan_mini')
    parser.add_argument('--save-every', type=int, default=10)
    
    return parser.parse_args()

# (EMA, SyntheticTitanDataset, create_data_loaders implementations are assumed to be present 
# from the user's provided context and remain unchanged. Omitted for brevity.)

def train_epoch(model, train_loader, optimizer, scheduler, scaler, args, epoch, writer, ema=None, *, is_main=True, distributed=False):
    """Train for one epoch."""
    model.train()

    # ---- Configure Auxiliary Losses (Dynamic Weighting) ----
    # This logic now correctly interacts with the implemented auxiliary losses in TitanMini.
    model_module = model.module if hasattr(model, 'module') else model

    # Check if the model supports auxiliary losses
    if hasattr(model_module, 'material_weight'):
        # Fractional progress through training (0..1)
        progress = (epoch) / max(1, args.epochs)

        # Schedule defined in the user's provided code
        if progress < 0.10:
            # Strong anchors early on
            model_module.material_weight = 0.15
            # model_module.material_scale_cp = 800.0 # (If used in network)
            model_module.wdl_weight = 0.55
            model_module.calibration_weight = 0.15
        elif progress < 0.30:
            model_module.material_weight = 0.08
            # model_module.material_scale_cp = 700.0
            model_module.wdl_weight = 0.58
            model_module.calibration_weight = 0.20
        else:
            # Relax anchors later
            model_module.material_weight = 0.05
            # model_module.material_scale_cp = 600.0
            model_module.wdl_weight = 0.60
            model_module.calibration_weight = 0.25
        
        if is_main and writer is not None:
            writer.add_scalar('Train/Aux_MaterialWeight', model_module.material_weight, epoch)
            writer.add_scalar('Train/Aux_CalibrationWeight', model_module.calibration_weight, epoch)
            writer.add_scalar('Train/Aux_WDLWeight', model_module.wdl_weight, epoch)

    # -------------------------------------------------------

    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    # (Data loader handling remains the same)
    if hasattr(train_loader, '__getitem__') and not hasattr(train_loader, 'batch_size'):
        from torch.utils.data import DataLoader
        # Use provided batch size or default if None
        default_batch_size = args.batch_size if args.batch_size else 256
        data_loader = DataLoader(train_loader, batch_size=default_batch_size, shuffle=True)
    else:
        data_loader = train_loader
    
    loader_len = len(data_loader)
    device = next(model.parameters()).device

    # Determine AMP settings (Prefer BF16 if available)
    use_amp = args.mixed_precision and device.type in ('cuda', 'mps')
    amp_dtype = torch.float16
    bf16_supported = (device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    
    if use_amp and bf16_supported:
        amp_dtype = torch.bfloat16


    for batch_idx, batch in enumerate(data_loader):
        # (Data loading and preprocessing remains the same)
        if isinstance(batch, dict):
            position = batch['position']; value_target = batch['value']
            policy_target = batch['policy']; policy_mask = batch.get('mask')
        else:
            position, value_target, policy_target, policy_mask = batch

        # (Plane adaptation logic remains the same)
        if position.dim() == 4 and position.size(1) != args.input_planes:
            c = position.size(1)
            if c < args.input_planes:
                pad = torch.zeros(position.size(0), args.input_planes - c, 8, 8, dtype=position.dtype)
                position = torch.cat([position, pad], dim=1)
            else:
                position = position[:, :args.input_planes]

        # Move to device and prepare targets
        position = position.to(device)
        value_target = value_target.to(device).float()
        
        # Handle different target shapes (Scalar vs WDL)
        if value_target.dim() == 1:
            value_target = value_target.view(-1, 1)
        # Ensure compatibility if target is [B, N] where N!=1 and N!=3
        elif value_target.dim() == 2 and value_target.size(1) > 1 and value_target.size(1) != 3:
             value_target = value_target[:, :1]
        
        # Basic range check/clamping for scalar targets
        if value_target.size(1) == 1:
            if value_target.min() < -1.001 or value_target.max() > 1.001:
                 value_target = torch.clamp(value_target, -1.0, 1.0)

        
        policy_target = policy_target.to(device)
        if policy_mask is not None:
            policy_mask = policy_mask.to(device)
        
        # Mixed precision training (AMP)
        
        amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else contextlib.nullcontext()

        with amp_ctx:
            # Model forward pass calculates total loss including auxiliary losses
            loss, value_loss, policy_loss = model(position, value_target, policy_target, policy_mask)

        # FIX 3: Robust check for non-finite loss immediately
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected (Epoch {epoch}, Batch {batch_idx}). Skipping batch.")
            # Ensure optimizer state is cleared to prevent contamination
            optimizer.zero_grad(set_to_none=True)
            if scaler: scaler.update() # Important: Update scaler if optimizer step is skipped
            continue

        loss = loss / args.gradient_accumulation

        if scaler is not None and use_amp:  # CUDA/GPU path with GradScaler
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping (using the provided grad_clip argument)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # Use set_to_none for efficiency

                if scheduler is not None:
                    scheduler.step()

                if ema is not None:
                    ema.update()
        else:  # FP32 or MPS/CPU path
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                if ema is not None:
                    ema.update()
        
        # Track losses
        # Note: value_loss and policy_loss here refer to the primary losses (WDL/Value and Policy)
        current_loss_val = loss.item() * args.gradient_accumulation
        total_loss += current_loss_val
        total_value_loss += value_loss.item()
        total_policy_loss += policy_loss.item()
        num_batches += 1
        
        # Log to tensorboard
        if is_main and batch_idx % 100 == 0:
            global_step = epoch * loader_len + batch_idx
            if writer is not None:
                writer.add_scalar('Train/Loss', current_loss_val, global_step)
                # Renamed for clarity as these are the primary components
                writer.add_scalar('Train/ValueLoss_Primary', value_loss.item(), global_step)
                writer.add_scalar('Train/PolicyLoss', policy_loss.item(), global_step)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                
                # (Value distribution monitoring remains the same)
                # ...
            
            elapsed = time.time() - start_time
            print(f'Epoch {epoch} [{batch_idx}/{loader_len}] '
                  f'Loss: {current_loss_val:.4f} '
                  f'Value (Primary): {value_loss.item():.4f} '
                  f'Policy: {policy_loss.item():.4f} '
                  f'Time: {elapsed:.1f}s')
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0

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
    
    # For curriculum datasets, estimate steps per epoch
    if args.mode == 'curriculum':
        # Get actual game count for current stage from the curriculum dataset
        current_stage_games = train_loader.get_current_stage_size()
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 256
        steps_per_epoch = current_stage_games // batch_size
        
        if is_main:
            stage_info = train_loader.get_stage_info()
            stage_name = stage_info.get('current_stage', 'unknown')
            print(f"Curriculum mode: {steps_per_epoch:,} steps per epoch ({current_stage_games:,} games in '{stage_name}' stage, batch size {batch_size})")
    else:
        steps_per_epoch = len(train_loader)
    
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    try:
        pct = warmup_steps / total_steps if total_steps > 0 else 0.1
        if not (0.0 < pct < 1.0):
            raise ValueError('Invalid pct_start for OneCycleLR')
        scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=total_steps, pct_start=pct,
                               anneal_strategy='cos', div_factor=10, final_div_factor=100)
    except Exception:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=max(1, steps_per_epoch//2), gamma=0.1)

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
        # Check if train_loader has a sampler (i.e., it's a DataLoader, not a curriculum dataset)
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_value_loss, train_policy_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args, epoch, writer, ema, is_main=is_main, distributed=distributed
        )
        if is_main:
            print(f'Train - Loss: {train_loss:.4f}, Value: {train_value_loss:.4f}, Policy: {train_policy_loss:.4f}')

        if val_loader is not None:
            val_loss, val_value_loss, val_policy_loss = validate(
                model, val_loader, args, epoch, writer, device=device, distributed=distributed
            )
            if is_main:
                print(f'Val - Loss: {val_loss:.4f}, Value: {val_value_loss:.4f}, Policy: {val_policy_loss:.4f}')
        else:
            # No validation in curriculum mode
            val_loss = float('inf')
            val_value_loss = float('inf')
            val_policy_loss = float('inf')
            if is_main:
                print('Validation skipped (curriculum mode)')

        if args.monitor_piece_values:
            enhanced = args.input_planes > 16
            monitor = TitanPieceValueMonitor(model, device='cuda', enhanced_encoder=enhanced)
            monitor.print_detailed_report()

        if is_main and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'titan_mini_epoch_{epoch + 1}.pt')
            save_checkpoint(model if not isinstance(model, DDP) else model.module, optimizer, scheduler, epoch, args, best_val_loss, checkpoint_path)

        if is_main and val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'titan_mini_best.pt')
            save_checkpoint(model if not isinstance(model, DDP) else model.module, optimizer, scheduler, epoch, args, best_val_loss, best_path)
            print(f'New best validation loss: {best_val_loss:.4f}')

        if args.swa and epoch >= args.swa_start and ema and is_main and val_loader is not None:
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
