import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset
from TitanNetwork import Titan, count_parameters
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import time
import contextlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import signal

# Defaults aligned with Luna but can differ for Titan
default_epochs = 500
default_num_layers = 15
default_d_model = 1024
default_num_heads = 16
default_d_ff = 4096
default_lr = 5e-6
default_warmup_epochs = 20
default_scheduler = 'cosine'  # Better for long training
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')


def parse_args():
    p = argparse.ArgumentParser('Train Titan transformer model')
    # Model
    p.add_argument('--num-layers', type=int, default=default_num_layers)
    p.add_argument('--d-model', type=int, default=default_d_model)
    p.add_argument('--num-heads', type=int, default=default_num_heads)
    p.add_argument('--d-ff', type=int, default=default_d_ff)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--policy-weight', type=float, default=default_policy_weight)
    p.add_argument('--input-planes', type=int, default=112, help='112 for enhanced encoder')
    p.add_argument('--grad-checkpointing', action='store_true')

    # Train
    p.add_argument('--epochs', type=int, default=default_epochs)
    p.add_argument('--lr', type=float, default=default_lr)
    p.add_argument('--warmup-epochs', type=int, default=default_warmup_epochs)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--batch-size-total', type=int, default=None)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--grad-clip', type=float, default=0.5)  # More aggressive clipping for stability
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--scheduler', choices=['onecycle', 'cosine', 'multistep', 'plateau'], default=default_scheduler)
    p.add_argument('--min-lr', type=float, default=1e-8, help='Minimum learning rate')
    p.add_argument('--lr-patience', type=int, default=10, help='Patience for ReduceLROnPlateau')

    # Data
    p.add_argument('--mode', choices=['supervised', 'rl', 'mixed'], default='mixed')
    p.add_argument('--ccrl-dir', type=str, default=ccrl_dir)
    p.add_argument('--rl-dir', type=str, default=rl_dir)
    p.add_argument('--mixed-ratio', type=float, default=0.7)
    p.add_argument('--label-smoothing-temp', type=float, default=0.15)
    p.add_argument('--validation-split', type=float, default=0.1)

    # Dry run
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-samples', type=int, default=64)

    # Distributed
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--dist-backend', type=str, default=None)

    # Logging
    p.add_argument('--log-dir', type=str, default='logs/titan')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints/titan')
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


class SyntheticTitanDataset(Dataset):
    def __init__(self, n, input_planes):
        self.n = int(max(1, n))
        self.c = int(input_planes)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        position = torch.randn(self.c, 8, 8)
        value = torch.rand(1).item() * 2 - 1
        policy = torch.randint(0, 72 * 64, (1,), dtype=torch.long).squeeze(0)
        mask = torch.ones(72, 8, 8, dtype=torch.int64)
        return position, torch.tensor([value], dtype=torch.float32), policy, mask


def create_data_loaders(args, *, distributed=False, rank=0, world_size=1, is_main=True):
    device, _ = get_optimal_device()
    if args.dry_run:
        n_train = args.dry_samples
        n_val = max(1, n_train // 4)
        train_ds = SyntheticTitanDataset(n_train, args.input_planes)
        val_ds = SyntheticTitanDataset(n_val, args.input_planes)
        bs = args.batch_size if args.batch_size else (2 if device.type == 'mps' else 4)
        num_workers = 0
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=False)
        if is_main:
            print(f'Training samples (dry): {len(train_ds)}, Validation samples (dry): {len(val_ds)}')
        return train_loader, val_loader

    # Real datasets
    if args.mode == 'supervised':
        ds = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp, enhanced_encoder=(args.input_planes > 16))
    elif args.mode == 'rl':
        ds = RLDataset(args.rl_dir)
    else:
        ccrl_ds = CCRLDataset(args.ccrl_dir, soft_targets=True, temperature=args.label_smoothing_temp, enhanced_encoder=(args.input_planes > 16))
        rl_ds = RLDataset(args.rl_dir)
        total = max(len(ccrl_ds), len(rl_ds))
        rl_size = int(total * args.mixed_ratio)
        ccrl_size = total - rl_size
        ccrl_idx = np.random.choice(len(ccrl_ds), ccrl_size, replace=True)
        rl_idx = np.random.choice(len(rl_ds), rl_size, replace=True)
        ds = ConcatDataset([Subset(ccrl_ds, ccrl_idx), Subset(rl_ds, rl_idx)])

    val_size = int(len(ds) * args.validation_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    if is_main:
        print(f'Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}')

    if args.batch_size_total and distributed and world_size > 1:
        bs = max(1, args.batch_size_total // world_size)
    elif args.batch_size:
        bs = args.batch_size
    else:
        bs = 64 if device.type == 'mps' else (get_batch_size_for_device() // 6)
    workers = get_num_workers_for_device()

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed and world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed and world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=workers, pin_memory=(device.type == 'cuda'), persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=max(1, bs * 2), shuffle=False, sampler=val_sampler,
                            num_workers=workers, pin_memory=(device.type == 'cuda'), persistent_workers=workers > 0)
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, args, epoch, writer, *, is_main=True, warmup_scheduler=None):
    model.train()
    total_loss = total_v = total_p = 0.0
    steps = 0
    device = next(model.parameters()).device
    start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, dict):
            position, value_t, policy_t, mask = batch['position'], batch['value'], batch['policy'], batch.get('mask')
        else:
            position, value_t, policy_t, mask = batch

        # Pad/crop planes if needed
        if position.dim() == 4 and position.size(1) != args.input_planes:
            c = position.size(1)
            if c < args.input_planes:
                pad = torch.zeros(position.size(0), args.input_planes - c, 8, 8, dtype=position.dtype)
                position = torch.cat([position, pad], dim=1)
            else:
                position = position[:, :args.input_planes]

        position = position.to(device)
        value_t = value_t.to(device).float()
        if value_t.dim() == 1:
            value_t = value_t.view(-1, 1)
        elif value_t.dim() == 2 and value_t.size(1) != 1:
            value_t = value_t[:, :1]
        policy_t = policy_t.to(device)
        mask = mask.to(device) if mask is not None else None

        use_amp = args.mixed_precision and device.type in ('cuda', 'mps') and hasattr(torch, 'autocast')
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else contextlib.nullcontext()

        scaler = None
        if args.mixed_precision and torch.cuda.is_available():
            # Initialize scaler once at the beginning of training
            if not hasattr(train_epoch, "_scaler"):
                try:
                    from torch.amp import GradScaler as AmpGradScaler
                    train_epoch._scaler = AmpGradScaler('cuda')
                except Exception:
                    from torch.cuda.amp import GradScaler as CudaGradScaler
                    train_epoch._scaler = CudaGradScaler()
            scaler = train_epoch._scaler

        if scaler is not None:
            with amp_ctx:
                loss, v_loss, p_loss = model(position, value_target=value_t, policy_target=policy_t, policy_mask=mask)
                loss = loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                # Monitor gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
                    print(f"WARNING: Large/invalid gradient norm: {grad_norm:.2f} at epoch {epoch} batch {batch_idx}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None and args.scheduler != 'plateau':
                    scheduler.step()
        else:
            with amp_ctx:
                loss, v_loss, p_loss = model(position, value_target=value_t, policy_target=policy_t, policy_mask=mask)
                loss = loss / args.gradient_accumulation
            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Monitor gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
                    print(f"WARNING: Large/invalid gradient norm: {grad_norm:.2f} at epoch {epoch} batch {batch_idx}")
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and args.scheduler != 'plateau':
                    scheduler.step()

        total_loss += loss.item() * args.gradient_accumulation
        total_v += v_loss.item()
        total_p += p_loss.item()
        steps += 1

        if is_main and batch_idx % 100 == 0:
            gs = epoch * len(train_loader) + batch_idx
            if writer:
                writer.add_scalar('Train/Loss', loss.item(), gs)
                writer.add_scalar('Train/ValueLoss', v_loss.item(), gs)
                writer.add_scalar('Train/PolicyLoss', p_loss.item(), gs)
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss {loss.item():.4f} V {v_loss.item():.4f} P {p_loss.item():.4f} t {time.time()-start:.1f}s')

    return total_loss / steps, total_v / steps, total_p / steps


def validate(model, val_loader, args, epoch, writer, *, device=None, distributed=False):
    model.eval()
    tot = tot_v = tot_p = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                position, value_t, policy_t, mask = batch['position'], batch['value'], batch['policy'], batch.get('mask')
            else:
                position, value_t, policy_t, mask = batch

            if position.dim() == 4 and position.size(1) != args.input_planes:
                c = position.size(1)
                if c < args.input_planes:
                    pad = torch.zeros(position.size(0), args.input_planes - c, 8, 8, dtype=position.dtype)
                    position = torch.cat([position, pad], dim=1)
                else:
                    position = position[:, :args.input_planes]

            device = next(model.parameters()).device
            position = position.to(device)
            value_t = value_t.to(device).float()
            if value_t.dim() == 1:
                value_t = value_t.view(-1, 1)
            elif value_t.dim() == 2 and value_t.size(1) != 1:
                value_t = value_t[:, :1]
            policy_t = policy_t.to(device)
            if mask is not None:
                mask = mask.to(device)

            loss, v_loss, p_loss = model(position, value_target=value_t, policy_target=policy_t, policy_mask=mask)
            tot += loss.item()
            tot_v += v_loss.item()
            tot_p += p_loss.item()
            n += 1

    avg = tot / n
    avg_v = tot_v / n
    avg_p = tot_p / n
    if writer:
        writer.add_scalar('Val/Loss', avg, epoch)
        writer.add_scalar('Val/ValueLoss', avg_v, epoch)
        writer.add_scalar('Val/PolicyLoss', avg_p, epoch)
    return avg, avg_v, avg_p


def train():
    args = parse_args()

    # DDP
    env_world = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = args.distributed or env_world > 1
    rank = 0
    local_rank = 0
    world_size = 1
    if distributed:
        backend = args.dist_backend or ('nccl' if torch.cuda.is_available() else 'gloo')
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            device_str = f'CUDA GPU {local_rank} (rank {rank}/{world_size})'
        else:
            device, device_str = get_optimal_device()
    else:
        device, device_str = get_optimal_device()
    is_main = (rank == 0)
    if is_main:
        print(f'Using device: {device_str}')

    # signals
    def _graceful_exit(signum, frame):
        try:
            if dist.is_available() and dist.is_initialized():
                if is_main:
                    print(f"Signal {signum}; destroying process group...")
                dist.destroy_process_group()
        finally:
            os._exit(0)

    for _sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _graceful_exit)
        except Exception:
            pass

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir) if is_main else None

    # model
    model = Titan(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        policy_weight=args.policy_weight,
        input_planes=args.input_planes,
        use_gradient_checkpointing=args.grad_checkpointing,
    ).to(device)
    if is_main:
        print(f'Titan initialized with {count_parameters(model):,} parameters')

    if args.compile and hasattr(torch, 'compile'):
        if is_main:
            print('Compiling model...')
        model = torch.compile(model)
    model = optimize_for_device(model, device)

    if distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_loader, val_loader = create_data_loaders(args, distributed=distributed, rank=rank, world_size=world_size, is_main=is_main)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = len(train_loader) * args.warmup_epochs // args.gradient_accumulation
    
    # Create scheduler based on args
    if args.scheduler == 'onecycle':
        pct = warmup_steps / total_steps if total_steps > 0 else 0.1
        try:
            if not (0.0 < pct < 1.0):
                raise ValueError
            scheduler = OneCycleLR(optimizer, max_lr=args.lr * 2, total_steps=total_steps, pct_start=pct,
                                   anneal_strategy='cos', div_factor=25, final_div_factor=10000)
        except Exception:
            scheduler = None
    elif args.scheduler == 'cosine':
        # Cosine annealing to very low LR for long training
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    elif args.scheduler == 'multistep':
        # Progressive reduction at specific epochs
        milestones = [100, 200, 300, 400, 450]  # Epochs where LR drops
        milestones_steps = [m * len(train_loader) // args.gradient_accumulation for m in milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones_steps, gamma=0.3)
    elif args.scheduler == 'plateau':
        # Reduce on plateau - will be called differently
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, 
                                      min_lr=args.min_lr, verbose=True if is_main else False)
    else:
        scheduler = None
    
    # Create warmup scheduler if needed
    warmup_scheduler = None
    if args.warmup_epochs > 0 and args.scheduler != 'onecycle':
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and is_main:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                if isinstance(model, DDP):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available and matching
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
                    print("Scheduler will start fresh with current settings")
            
            # Get epoch to resume from
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            # Get best validation loss if available
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
            
            print(f"Resumed from epoch {start_epoch}")
            if best_val_loss != float('inf'):
                print(f"Best validation loss so far: {best_val_loss:.4f}")
        else:
            print(f"Warning: Resume checkpoint {args.resume} not found, starting from scratch")
    
    # train loop (short for dry-run)
    if is_main:
        print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch})...")
        print(f"Scheduler: {args.scheduler}, Initial LR: {args.lr}, Min LR: {args.min_lr}")
    
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Pass the appropriate scheduler to train_epoch
        if args.scheduler != 'plateau':
            if warmup_scheduler is not None and epoch < args.warmup_epochs:
                epoch_scheduler = warmup_scheduler
            else:
                epoch_scheduler = scheduler
        else:
            epoch_scheduler = None
        tr_l, tr_v, tr_p = train_epoch(model, train_loader, optimizer, epoch_scheduler, args, epoch, writer, is_main=is_main, warmup_scheduler=warmup_scheduler)
        if is_main:
            print(f'Train - Loss: {tr_l:.4f}, Value: {tr_v:.4f}, Policy: {tr_p:.4f}')
        va_l, va_v, va_p = validate(model, val_loader, args, epoch, writer)
        if is_main:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Val - Loss: {va_l:.4f}, Value: {va_v:.4f}, Policy: {va_p:.4f}, LR: {current_lr:.2e}')
            
            # Step plateau scheduler if used
            if args.scheduler == 'plateau' and scheduler is not None:
                scheduler.step(va_l)
            
            # Track best model
            if va_l < best_val_loss:
                best_val_loss = va_l
                best_ckpt = os.path.join(args.checkpoint_dir, 'titan_best.pt')
                torch.save({'epoch': epoch,
                            'model_state_dict': (model if not isinstance(model, DDP) else model.module).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': va_l}, best_ckpt)
                print(f'New best model saved to {best_ckpt} (val_loss: {va_l:.4f})')

        if is_main and (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f'titan_epoch_{epoch+1}.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': (model if not isinstance(model, DDP) else model.module).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'val_loss': va_l,
                        'args': vars(args)}, ckpt)
            print(f'Saved checkpoint to {ckpt}')

    if is_main:
        out = args.output or os.path.join('weights', f'Titan_{args.num_layers}x{args.d_model}.pt')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save((model if not isinstance(model, DDP) else model.module).state_dict(), out)
        print(f'Training complete. Model saved to {out}')
        if writer:
            writer.close()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
