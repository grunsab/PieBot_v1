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

def generate_games_on_process(gpu_id, proc_id, args, games_per_process, offset, result_queue):
    """
    Generate games on a specific process assigned to a GPU.
    
    Args:
        gpu_id: GPU device ID
        proc_id: Process ID within the GPU
        args: Command line arguments
        games_per_process: Number of games to generate on this process
        offset: File numbering offset for this process
        result_queue: Queue to store position count results
    """
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    process_label = f"{gpu_id}.{proc_id}"
    log(f"Starting game generation: {games_per_process} games, offset {offset}", process_label)
    
    # Build command for create_training_games.py
    cmd = [
        sys.executable,
        'create_training_games.py',
        '--model', args.model,
        '--save-format', args.save_format,
        '--rollouts', str(args.rollouts),
        '--temperature', str(args.temperature),
        '--games-to-play', str(games_per_process),
        '--threads', str(args.threads_per_gpu),
        '--output-dir', args.output_dir,
        '--file-base', f'{args.file_base}_gpu{gpu_id}_proc{proc_id}',
        '--offset', str(offset),
        '--iteration', str(args.iteration)
    ]
    
    
    # Add verbose flag if set
    if args.verbose:
        cmd.append('--verbose')
    
    # Log the command
    log(f"Executing: {' '.join(cmd)}", process_label)
    
    # Run the command
    start_time = time.time()
    
    positions_generated = 0
    
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
        
        # Stream output with process label prefix
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[GPU {process_label}] {line.rstrip()}")
                # Parse position count
                if line.startswith("TOTAL_POSITIONS:"):
                    try:
                        positions_generated = int(line.split(":")[1].strip())
                    except:
                        pass
        
        # Wait for completion
        returncode = process.wait()
        
        if returncode != 0:
            raise RuntimeError(f"Game generation failed with return code {returncode}")
        
        elapsed = time.time() - start_time
        games_per_hour = games_per_process / elapsed * 3600
        
        log(f"Completed {games_per_process} games in {elapsed:.1f}s ({games_per_hour:.0f} games/hour)", process_label)
        
        # Put result in queue
        result_queue.put((gpu_id, proc_id, positions_generated))
        
    except Exception as e:
        log(f"Error during game generation: {e}", process_label)
        # Put error result in queue
        result_queue.put((gpu_id, proc_id, -1))
        raise

def calculate_distribution(total_games, num_processes, gpu_weights=None, num_gpus=None, processes_per_gpu=1):
    """
    Calculate how many games each process should generate.
    
    Args:
        total_games: Total number of games to generate
        num_processes: Total number of processes across all GPUs
        gpu_weights: Optional weights for each GPU (for heterogeneous setups)
        num_gpus: Number of GPUs (required if gpu_weights is provided)
        processes_per_gpu: Number of processes per GPU
    
    Returns:
        List of (games_count, offset) tuples for each process
    """
    if gpu_weights is None:
        # Equal distribution
        base_games = total_games // num_processes
        remainder = total_games % num_processes
        
        distribution = []
        offset = 0
        
        for i in range(num_processes):
            games = base_games + (1 if i < remainder else 0)
            distribution.append((games, offset))
            offset += games
    else:
        # Weighted distribution - first allocate games to GPUs, then split among processes
        total_weight = sum(gpu_weights)
        gpu_games = []
        allocated = 0
        
        # Calculate games per GPU based on weights
        for i, weight in enumerate(gpu_weights):
            if i == len(gpu_weights) - 1:
                # Last GPU gets any remaining games due to rounding
                games = total_games - allocated
            else:
                games = int(total_games * weight / total_weight)
                allocated += games
            gpu_games.append(games)
        
        # Now distribute games among processes for each GPU
        distribution = []
        offset = 0
        
        for gpu_idx, gpu_total in enumerate(gpu_games):
            # Distribute this GPU's games among its processes
            base_games = gpu_total // processes_per_gpu
            remainder = gpu_total % processes_per_gpu
            
            for proc_idx in range(processes_per_gpu):
                games = base_games + (1 if proc_idx < remainder else 0)
                distribution.append((games, offset))
                offset += games
    
    return distribution

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU parallel game generation for AlphaZero')
    
    # Model and game parameters
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--games-total', type=int, required=True, help='Total number of games to generate')
    parser.add_argument('--rollouts', type=int, default=40, help='MCTS rollouts. Total rollouts is this times number of threads.')
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
    
    
    # Other options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Show distribution without running')
    parser.add_argument('--num-processes', type=int, default=1, 
                       help='Number of CPU processes per GPU (default: 1)')
    
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
    
    # Calculate game distribution across all processes
    total_processes = num_gpus * args.num_processes
    distribution = calculate_distribution(args.games_total, total_processes, gpu_weights, num_gpus, args.num_processes)
    
    log("Game distribution plan:")
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total_games = 0
        for proc_idx in range(args.num_processes):
            process_idx = gpu_idx * args.num_processes + proc_idx
            games, offset = distribution[process_idx]
            gpu_total_games += games
            log(f"  GPU {gpu_id} Process {proc_idx}: {games} games (offset {offset})")
        log(f"  GPU {gpu_id} Total: {gpu_total_games} games")
    
    if args.dry_run:
        log("Dry run complete - no games generated")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # Launch parallel processes
    log(f"Starting parallel game generation on {num_gpus} GPUs with {args.num_processes} processes per GPU...")
    start_time = time.time()
    
    # Create queue for collecting position counts
    result_queue = mp.Queue()
    
    processes = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for proc_idx in range(args.num_processes):
            process_idx = gpu_idx * args.num_processes + proc_idx
            games, offset = distribution[process_idx]
            
            p = mp.Process(
                target=generate_games_on_process,
                args=(gpu_id, proc_idx, args, games, offset, result_queue)
            )
            p.start()
            processes.append((p, gpu_id, proc_idx))
            
            # Small delay to avoid race conditions
            time.sleep(0.5)
    
    # Wait for all processes to complete
    log("Waiting for all processes to complete...")
    
    failed = False
    position_counts = {}
    
    for p, gpu_id, proc_id in processes:
        p.join()
        if p.exitcode != 0:
            log(f"GPU {gpu_id} Process {proc_id} failed with exit code {p.exitcode}")
            failed = True
    
    # Collect position counts from queue
    total_positions = 0
    while not result_queue.empty():
        gpu_id, proc_id, positions = result_queue.get()
        if positions >= 0:
            position_counts[(gpu_id, proc_id)] = positions
            total_positions += positions
        else:
            log(f"GPU {gpu_id} Process {proc_id} failed to report positions")
    
    # Calculate statistics
    total_time = time.time() - start_time
    total_games_per_hour = args.games_total / total_time * 3600
    
    # Calculate positions per hour using actual counts
    total_positions_per_hour = total_positions / total_time * 3600 if total_positions > 0 else 0
    
    if not failed:
        log(f"\nSuccess! Generated {args.games_total} games in {total_time:.1f}s")
        log(f"Overall throughput: {total_games_per_hour:.0f} games/hour")
        if total_positions > 0:
            log(f"Total positions: {total_positions:,} ({total_positions_per_hour:,.0f} positions/hour)")
            log(f"Average positions per game: {total_positions/args.games_total:.1f}")
            log(f"Average per GPU: {total_games_per_hour/num_gpus:.0f} games/hour ({total_positions_per_hour/num_gpus:,.0f} positions/hour)")
            log(f"Average per process: {total_games_per_hour/total_processes:.0f} games/hour ({total_positions_per_hour/total_processes:,.0f} positions/hour)")
        else:
            log("Position counting not available (using PGN format or older create_training_games.py)")
            log(f"Average per GPU: {total_games_per_hour/num_gpus:.0f} games/hour")
            log(f"Average per process: {total_games_per_hour/total_processes:.0f} games/hour")
        if args.num_processes > 1:
            log(f"Using {args.num_processes} processes per GPU improved throughput by allowing parallel CPU work")
    else:
        log("\nSome processes failed - check logs above")
        sys.exit(1)

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()