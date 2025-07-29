
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from CCRLDataset import CCRLDataset
from RLDataset import RLDataset, WeightedRLSampler
from AlphaZeroNetwork import AlphaZeroNet
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device, get_num_workers_for_device
import argparse
import re

#Training params (defaults)
default_epochs = 40
default_blocks = 20
default_filters = 256
default_lr = 0.001
default_policy_weight = 1.0
ccrl_dir = os.path.abspath('games_training_data/reformatted/')
rl_dir = os.path.abspath('games_training_data/selfplay/')
logmode=True

def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero network')
    parser.add_argument('--resume', type=str, default="AlphaZeroNet_20x256_distributed.pt",
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help=f'Learning rate (default: {default_lr})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: auto-detect based on device)')
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
    return parser.parse_args()

def train():
    args = parse_args()
    # Get optimal device and configure for training
    device, device_str = get_optimal_device()
    print(f'Using device: {device_str}')
    
    # Optimize batch size and num_workers for the device
    batch_size = args.batch_size if args.batch_size else get_batch_size_for_device()
    num_workers = get_num_workers_for_device()
    print(f'Batch size: {batch_size}, Workers: {num_workers}')
    
    # Create dataset based on training mode
    if args.mode == 'supervised':
        print(f'Training mode: Supervised learning on CCRL dataset')
        train_ds = CCRLDataset(ccrl_dir, soft_targets=False)
    elif args.mode == 'rl':
        print(f'Training mode: Reinforcement learning on self-play data')
        if args.rl_weight_recent:
            print(f'Using weighted sampling with decay factor {args.rl_weight_decay}')
        train_ds = RLDataset(args.rl_dir, weight_recent=args.rl_weight_recent, 
                           weight_decay=args.rl_weight_decay)
    else:  # mixed mode
        print(f'Training mode: Mixed (RL ratio: {args.mixed_ratio})')
        # Use soft targets for CCRL in mixed mode for compatibility
        print(f'Using soft targets for CCRL data with temperature {args.label_smoothing_temp}')
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
    
    # Create data loader with appropriate sampler
    if args.mode == 'rl' and args.rl_weight_recent:
        # Use weighted sampler for RL mode with recent game weighting
        sampler = WeightedRLSampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                                num_workers=num_workers)
    else:
        # Standard shuffled data loader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)

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
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            # Handle loading old checkpoints that don't have policy_weight in state_dict
            alphaZeroNet.load_state_dict(checkpoint, strict=False)
            print(f'Checkpoint loaded successfully (policy_weight={args.policy_weight})')
        else:
            raise FileNotFoundError(f'Checkpoint file not found: {args.resume}')

    alphaZeroNet = optimize_for_device(alphaZeroNet, device)
    
    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.lr)
    mseLoss = nn.MSELoss()

    print( 'Starting training' )

    for epoch in range(start_epoch, args.epochs):
        
        alphaZeroNet.train()
        for iter_num, data in enumerate( train_loader ):

            optimizer.zero_grad()

            # Move data to device
            position = data[ 'position' ].to(device)
            valueTarget = data[ 'value' ].to(device)
            policyTarget = data[ 'policy' ].to(device)

            # You can manually examine some the training data here

            loss, valueLoss, policyLoss = alphaZeroNet( position, valueTarget=valueTarget,
                                 policyTarget=policyTarget )

            loss.backward()

            optimizer.step()

            message = 'Epoch {:03} | Step {:05} / {:05} | Total loss {:0.5f} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                     epoch, iter_num, len( train_loader ), float( loss ), float( valueLoss ), float( policyLoss ) )
            
            if iter_num != 0 and not logmode:
                print( ('\b' * len(message) ), end='' )
            print( message, end='', flush=True )
            if logmode:
                print('')
        
        print( '' )
        
        # Determine output filename
        if args.output:
            networkFileName = args.output
        elif args.resume:
            # When resuming, add '_continued' to avoid overwriting original
            base_name = f'AlphaZeroNet_{num_blocks}x{num_filters}'
            networkFileName = f'{base_name}_continued.pt'
        else:
            networkFileName = f'AlphaZeroNet_{num_blocks}x{num_filters}.pt'

        torch.save(alphaZeroNet.state_dict(), networkFileName)

        print(f'Saved model to {networkFileName}')

if __name__ == '__main__':

    train()
