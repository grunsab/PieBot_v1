#!/usr/bin/env python3
"""
Multi-GPU parallel game generation for AlphaZero training.
Launches multiple processes, each using a different GPU to generate games independently.
"""

import argparse
import os
import sys
import multiprocessing as mp
import subprocess
import time
from datetime import datetime
import torch

def log(message, gpu_id=None):
    """Log message with timestamp and GPU ID"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_str = f"[GPU {gpu_id}]" if gpu_id is not None else ""
    print(f"[{timestamp}] {gpu_str} {message}")

def check_cuda_availability():
    """Check CUDA availability and return GPU count"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU game generation requires CUDA.")
    
    gpu_count = torch.cuda.device_count()
    log(f"Found {gpu_count} CUDA devices")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    return gpu_count

def generate_games_on_gpu(gpu_id, args, games_per_gpu, offset):
    """
    Generate games on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        args: Command line arguments
        games_per_gpu: Number of games to generate on this GPU
        offset: File numbering offset for this GPU
    """
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log(f"Starting game generation: {games_per_gpu} games, offset {offset}", gpu_id)
    
    # Build command for create_training_games.py
    cmd = [
        sys.executable,
        'create_training_games.py',
        '--model', args.model,
        '--save-format', args.save_format,
        '--rollouts', str(args.rollouts),
        '--temperature', str(args.temperature),
        '--games-to-play', str(games_per_gpu),
        '--threads', str(args.threads_per_gpu),
        '--output-dir', args.output_dir,
        '--file-base', f'{args.file_base}_gpu{gpu_id}',
        '--offset', str(offset),
        '--iteration', str(args.iteration)
    ]
    
    # Add CUDA optimization flag if requested
    if args.use_cuda_mcts:
        cmd.extend(['--use-cuda-mcts'])
    
    # Add verbose flag if set
    if args.verbose:
        cmd.append('--verbose')
    
    # Log the command
    log(f"Executing: {' '.join(cmd)}", gpu_id)
    
    # Run the command
    start_time = time.time()
    
    try:
        # Use subprocess to run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        )
        
        # Stream output with GPU ID prefix
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[GPU {gpu_id}] {line.rstrip()}")
        
        # Wait for completion
        returncode = process.wait()
        
        if returncode != 0:
            raise RuntimeError(f"Game generation failed with return code {returncode}")
        
        elapsed = time.time() - start_time
        games_per_hour = games_per_gpu / elapsed * 3600
        
        log(f"Completed {games_per_gpu} games in {elapsed:.1f}s ({games_per_hour:.0f} games/hour)", gpu_id)
        
    except Exception as e:
        log(f"Error during game generation: {e}", gpu_id)
        raise

def calculate_distribution(total_games, num_gpus, gpu_weights=None):
    """
    Calculate how many games each GPU should generate.
    
    Args:
        total_games: Total number of games to generate
        num_gpus: Number of GPUs to use
        gpu_weights: Optional weights for each GPU (for heterogeneous setups)
    
    Returns:
        List of (games_count, offset) tuples for each GPU
    """
    if gpu_weights is None:
        # Equal distribution
        base_games = total_games // num_gpus
        remainder = total_games % num_gpus
        
        distribution = []
        offset = 0
        
        for i in range(num_gpus):
            games = base_games + (1 if i < remainder else 0)
            distribution.append((games, offset))
            offset += games
    else:
        # Weighted distribution
        total_weight = sum(gpu_weights)
        distribution = []
        offset = 0
        allocated = 0
        
        for i, weight in enumerate(gpu_weights):
            if i == len(gpu_weights) - 1:
                # Last GPU gets any remaining games due to rounding
                games = total_games - allocated
            else:
                games = int(total_games * weight / total_weight)
                allocated += games
            
            distribution.append((games, offset))
            offset += games
    
    return distribution

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU parallel game generation for AlphaZero')
    
    # Model and game parameters
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--games-total', type=int, required=True, help='Total number of games to generate')
    parser.add_argument('--rollouts', type=int, default=40, help='MCTS rollouts per move')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection')
    parser.add_argument('--threads-per-gpu', type=int, default=20, help='Threads per GPU for MCTS')
    
    # Output configuration
    parser.add_argument('--output-dir', default='games_training_data/selfplay', help='Output directory')
    parser.add_argument('--file-base', default='selfplay', help='Base name for output files')
    parser.add_argument('--save-format', choices=['pgn', 'h5'], default='h5', help='Output format')
    parser.add_argument('--iteration', type=int, default=0, help='Training iteration number')
    
    # GPU configuration
    parser.add_argument('--gpus', help='Comma-separated list of GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--gpu-weights', help='Comma-separated weights for heterogeneous GPUs')
    
    # Optimization options
    parser.add_argument('--use-cuda-mcts', action='store_true', 
                       help='Use CUDA-optimized MCTS (requires built extensions)')
    
    # Other options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Show distribution without running')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    available_gpus = check_cuda_availability()
    
    # Parse GPU list
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
        # Validate GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available (only {available_gpus} GPUs found)")
    else:
        # Use all available GPUs
        gpu_ids = list(range(available_gpus))
    
    num_gpus = len(gpu_ids)
    log(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    # Parse GPU weights if provided
    gpu_weights = None
    if args.gpu_weights:
        gpu_weights = [float(w.strip()) for w in args.gpu_weights.split(',')]
        if len(gpu_weights) != num_gpus:
            raise ValueError(f"Number of weights ({len(gpu_weights)}) must match number of GPUs ({num_gpus})")
    
    # Calculate game distribution
    distribution = calculate_distribution(args.games_total, num_gpus, gpu_weights)
    
    log("Game distribution plan:")
    for i, (gpu_id, (games, offset)) in enumerate(zip(gpu_ids, distribution)):
        log(f"  GPU {gpu_id}: {games} games (offset {offset})")
    
    if args.dry_run:
        log("Dry run complete - no games generated")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA MCTS is available
    if args.use_cuda_mcts:
        try:
            # Test import
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import MCTS_cuda_optimized
            log("CUDA-optimized MCTS available")
        except ImportError:
            log("WARNING: CUDA-optimized MCTS not available, falling back to standard MCTS")
            args.use_cuda_mcts = False
    
    # Launch parallel processes
    log(f"Starting parallel game generation on {num_gpus} GPUs...")
    start_time = time.time()
    
    processes = []
    for gpu_id, (games, offset) in zip(gpu_ids, distribution):
        p = mp.Process(
            target=generate_games_on_gpu,
            args=(gpu_id, args, games, offset)
        )
        p.start()
        processes.append((p, gpu_id))
        
        # Small delay to avoid race conditions
        time.sleep(0.5)
    
    # Wait for all processes to complete
    log("Waiting for all GPUs to complete...")
    
    failed = False
    for p, gpu_id in processes:
        p.join()
        if p.exitcode != 0:
            log(f"GPU {gpu_id} failed with exit code {p.exitcode}")
            failed = True
    
    # Calculate statistics
    total_time = time.time() - start_time
    total_games_per_hour = args.games_total / total_time * 3600
    
    if not failed:
        log(f"\nSuccess! Generated {args.games_total} games in {total_time:.1f}s")
        log(f"Overall throughput: {total_games_per_hour:.0f} games/hour")
        log(f"Average per GPU: {total_games_per_hour/num_gpus:.0f} games/hour")
    else:
        log("\nSome GPUs failed - check logs above")
        sys.exit(1)

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()