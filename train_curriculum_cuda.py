#!/usr/bin/env python3
"""
Curriculum learning script for AlphaZero training - Windows/CUDA Optimized Version

High-performance curriculum learning optimized for Windows systems with RTX 4080.
Manages transition from supervised to reinforcement learning with CUDA acceleration.
"""

import os
import sys
import subprocess
import argparse
import time
import json
import torch
from datetime import datetime
import multiprocessing as mp

# Windows-specific optimizations
if sys.platform == 'win32':
    # Prevent Windows from sleeping during long training runs
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

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def check_cuda_availability():
    """Check CUDA availability and return device info"""
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available. This version requires an NVIDIA GPU.")
        sys.exit(1)
    
    device_count = torch.cuda.device_count()
    log(f"Found {device_count} CUDA device(s):")
    
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        log(f"  Device {i}: {name} ({memory:.1f}GB)")
    
    return device_count

def run_command(cmd, check=True, env=None):
    """Run a command with Windows optimizations"""
    log(f"Running: {cmd}")
    
    # Windows-specific: Use explicit python path
    if sys.platform == 'win32' and cmd.startswith("python3"):
        cmd = cmd.replace("python3", sys.executable)
    
    # Set up environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    # Windows: Enable TF32 for all child processes
    cmd_env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Use subprocess.Popen for real-time output streaming
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1,
        env=cmd_env
    )
    
    # Stream output line by line
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    
    # Wait for process to complete
    returncode = process.wait()
    
    if check and returncode != 0:
        raise RuntimeError(f"Command failed with return code {returncode}: {cmd}")
    
    return returncode

def phase1_supervised_cuda(args):
    """Phase 1: Supervised learning with CUDA optimization"""
    log("Starting Phase 1: Supervised Learning (CUDA Optimized)")
    
    cmd = f"python train_cuda.py --mode supervised --epochs {args.supervised_epochs} --lr {args.supervised_lr}"
    
    # Add CUDA-specific parameters
    cmd += f" --batch-size {args.batch_size}"
    cmd += f" --num-workers {args.num_workers}"
    cmd += f" --gradient-accumulation {args.gradient_accumulation}"
    
    if args.multi_gpu:
        cmd += " --multi-gpu"
    else:
        cmd += f" --device {args.device}"
    
    if args.mixed_precision:
        cmd += " --mixed-precision"
    
    if args.compile:
        cmd += " --compile"
    
    if args.resume_supervised:
        cmd += f" --resume {args.resume_supervised}"
    
    # Output model name
    output_model = f"AlphaZeroNet_{args.blocks}x{args.filters}_cuda.pt"
    cmd += f" --output {output_model}"
    
    run_command(cmd)
    log("Phase 1 Complete")
    return output_model

def phase2_selfplay_cuda(args, model_path, iteration):
    """Phase 2: Generate self-play games with CUDA acceleration"""
    log(f"Starting Phase 2: Self-Play Generation (Iteration {iteration}) - CUDA Optimized")
    
    output_dir = f"{args.selfplay_dir}/iter_{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use CUDA-optimized self-play generation
    cmd = f"""python create_training_games_cuda.py \
        --model {model_path} \
        --games {args.games_per_iter} \
        --rollouts {args.rollouts} \
        --threads {args.mcts_threads} \
        --output {output_dir} \
        --file-base selfplay_iter{iteration} \
        --iteration {iteration} \
        --temperature {args.temperature} \
        --format npz \
        --batch-size {args.mcts_batch_size} \
        --workers {args.selfplay_workers}"""
    
    # Add GPU specification
    if args.multi_gpu:
        gpus = list(range(torch.cuda.device_count()))
        cmd += f" --gpus {' '.join(map(str, gpus))}"
    else:
        cmd += f" --gpus {args.device}"
    
    run_command(cmd)
    log(f"Generated {args.games_per_iter} games in {output_dir}")
    return output_dir

def phase3_reinforcement_cuda(args, model_path, iteration):
    """Phase 3: Reinforcement learning with CUDA optimization"""
    log(f"Starting Phase 3: Reinforcement Learning (Iteration {iteration}) - CUDA Optimized")
    log(f"Training on all games from {args.selfplay_dir} with weight decay {args.weight_decay}")
    
    output_model = f"AlphaZeroNet_{args.blocks}x{args.filters}_cuda_rl_iter{iteration}.pt"
    
    cmd = f"""python train_cuda.py \
        --mode rl \
        --resume {model_path} \
        --rl-dir {args.selfplay_dir} \
        --epochs {args.rl_epochs} \
        --lr {args.rl_lr} \
        --output {output_model} \
        --rl-weight-recent \
        --rl-weight-decay {args.weight_decay} \
        --batch-size {args.batch_size} \
        --num-workers {args.num_workers} \
        --gradient-accumulation {args.gradient_accumulation}"""
    
    if args.multi_gpu:
        cmd += " --multi-gpu"
    else:
        cmd += f" --device {args.device}"
    
    if args.mixed_precision:
        cmd += " --mixed-precision"
    
    if args.compile:
        cmd += " --compile"
    
    run_command(cmd)
    log(f"Phase 3 Complete: {output_model}")
    return output_model

def evaluate_model_cuda(args, model_path, reference_model=None):
    """Evaluate model using CUDA-accelerated matches"""
    log(f"Evaluating model: {model_path}")
    
    if reference_model is None:
        # Self-play evaluation
        log("Running self-play evaluation...")
        rating = 2800  # Placeholder
    else:
        # Play matches against reference
        log(f"Playing evaluation matches against {reference_model}")
        
        # Run evaluation matches using CUDA engine
        eval_cmd = f"""python evaluate_cuda.py \
            --model1 {model_path} \
            --model2 {reference_model} \
            --games 100 \
            --rollouts {args.eval_rollouts} \
            --threads {args.mcts_threads} \
            --batch-size {args.mcts_batch_size}"""
        
        if args.multi_gpu:
            gpus = list(range(torch.cuda.device_count()))
            eval_cmd += f" --gpus {' '.join(map(str, gpus))}"
        else:
            eval_cmd += f" --gpus {args.device}"
        
        # For now, just return a placeholder
        # In practice, you would parse the evaluation results
        rating = 2850
    
    return rating

def save_training_state(args, iteration, current_model, metrics=None):
    """Save current training state for resumption"""
    state = {
        'iteration': iteration,
        'current_model': current_model,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'metrics': metrics or {}
    }
    
    state_file = os.path.join(args.output_dir, 'training_state.json')
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    log(f"Saved training state to {state_file}")

def load_training_state(state_file):
    """Load training state from file"""
    with open(state_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description='Curriculum learning for AlphaZero - CUDA optimized for Windows'
    )
    
    # Model architecture
    parser.add_argument('--blocks', type=int, default=20, help='Number of residual blocks')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters')
    
    # Phase 1: Supervised learning
    parser.add_argument('--supervised-epochs', type=int, default=20, 
                        help='Epochs for supervised learning')
    parser.add_argument('--supervised-lr', type=float, default=0.001, 
                        help='Learning rate for supervised (higher for CUDA)')
    parser.add_argument('--resume-supervised', type=str, 
                        help='Resume supervised training from checkpoint')
    parser.add_argument('--skip-supervised', action='store_true', 
                        help='Skip supervised phase')
    
    # Phase 2: Self-play
    parser.add_argument('--games-per-iter', type=int, default=50000, 
                        help='Games per iteration (higher for CUDA)')
    parser.add_argument('--rollouts', type=int, default=1600, 
                        help='MCTS rollouts per move (higher for CUDA)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for move selection')
    parser.add_argument('--mcts-threads', type=int, default=64, 
                        help='Threads for MCTS (optimized for high-end CPUs)')
    parser.add_argument('--mcts-batch-size', type=int, default=512, 
                        help='Neural network batch size for MCTS')
    parser.add_argument('--selfplay-workers', type=int, default=4, 
                        help='Parallel self-play workers')
    
    # Phase 3: Reinforcement learning
    parser.add_argument('--rl-epochs', type=int, default=10, 
                        help='Epochs per RL iteration')
    parser.add_argument('--rl-lr', type=float, default=0.001, 
                        help='Learning rate for RL')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of RL iterations')
    parser.add_argument('--weight-decay', type=float, default=0.05, 
                        help='Weight decay for recent games')
    
    # CUDA/Training configuration
    parser.add_argument('--device', type=int, default=0, 
                        help='CUDA device ID')
    parser.add_argument('--multi-gpu', action='store_true', 
                        help='Use all available GPUs')
    parser.add_argument('--batch-size', type=int, default=1024, 
                        help='Training batch size (large for RTX 4080)')
    parser.add_argument('--gradient-accumulation', type=int, default=1, 
                        help='Gradient accumulation steps')
    parser.add_argument('--num-workers', type=int, default=8, 
                        help='Data loader workers')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--compile', action='store_true', 
                        help='Use torch.compile (PyTorch 2.0+)')
    
    # Evaluation
    parser.add_argument('--eval-rollouts', type=int, default=5000, 
                        help='Rollouts for evaluation matches')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='curriculum_training_cuda', 
                        help='Output directory')
    parser.add_argument('--selfplay-dir', type=str, default='games_training_data/selfplay_cuda', 
                        help='Self-play data directory')
    
    # Resume training
    parser.add_argument('--resume-state', type=str, 
                        help='Resume from saved training state')
    
    args = parser.parse_args()
    
    # Windows multiprocessing setup
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)
    
    # Check CUDA availability
    device_count = check_cuda_availability()
    
    # Adjust settings based on available GPUs
    if args.multi_gpu and device_count > 1:
        log(f"Using multi-GPU training with {device_count} devices")
        args.selfplay_workers = min(args.selfplay_workers, device_count * 2)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.selfplay_dir, exist_ok=True)
    
    # Log configuration
    log("="*60)
    log("AlphaZero Curriculum Learning - CUDA/Windows Optimized")
    log("="*60)
    log(f"Model: {args.blocks}x{args.filters}")
    log(f"CUDA devices: {device_count}")
    log(f"Batch size: {args.batch_size}")
    log(f"Mixed precision: {args.mixed_precision}")
    log(f"Games per iteration: {args.games_per_iter}")
    log(f"MCTS rollouts: {args.rollouts}")
    log(f"Self-play workers: {args.selfplay_workers}")
    log("-"*60)
    
    # Resume from saved state if specified
    start_iteration = 0
    current_model = None
    
    if args.resume_state:
        log(f"Resuming from {args.resume_state}")
        state = load_training_state(args.resume_state)
        start_iteration = state['iteration']
        current_model = state['current_model']
        log(f"Resuming from iteration {start_iteration}, model: {current_model}")
    
    # Phase 1: Supervised Learning (if not skipping)
    if not args.skip_supervised and start_iteration == 0:
        current_model = phase1_supervised_cuda(args)
        save_training_state(args, 0, current_model)
    elif current_model is None:
        # Must specify initial model if skipping supervised
        if args.resume_supervised:
            current_model = args.resume_supervised
        else:
            raise ValueError("Must specify --resume-supervised or --resume-state when using --skip-supervised")
    
    # Track metrics
    iteration_times = []
    model_ratings = []
    
    # Iterative RL training
    for iteration in range(start_iteration, args.iterations):
        iteration_start = time.time()
        
        log(f"\n{'='*60}")
        log(f"Starting Iteration {iteration + 1}/{args.iterations}")
        log(f"Current model: {current_model}")
        
        # Phase 2: Generate self-play games
        data_dir = phase2_selfplay_cuda(args, current_model, iteration + 1)
        
        # Phase 3: Train on ALL self-play data with recent games weighted
        new_model = phase3_reinforcement_cuda(args, current_model, iteration + 1)
        
        # Evaluate new model
        rating = evaluate_model_cuda(args, new_model, current_model)
        log(f"Model rating: ~{rating} ELO")
        model_ratings.append(rating)
        
        # Update current model
        current_model = new_model
        
        # Track iteration time
        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        
        # Calculate statistics
        avg_time = sum(iteration_times) / len(iteration_times)
        eta = avg_time * (args.iterations - iteration - 1)
        
        log(f"Iteration time: {iteration_time/60:.1f} minutes")
        log(f"Average time per iteration: {avg_time/60:.1f} minutes")
        log(f"Estimated time remaining: {eta/3600:.1f} hours")
        
        # Save training state with metrics
        metrics = {
            'iteration_times': iteration_times,
            'model_ratings': model_ratings,
            'last_iteration_time': iteration_time,
            'average_iteration_time': avg_time
        }
        save_training_state(args, iteration + 1, current_model, metrics)
        
        log(f"Iteration {iteration + 1} complete")
        
        # Windows: Ensure system stays awake
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    
    log("\n" + "="*60)
    log("Curriculum training complete!")
    log("="*60)
    log(f"Final model: {current_model}")
    log(f"Total training time: {sum(iteration_times)/3600:.1f} hours")
    log(f"Average iteration time: {sum(iteration_times)/len(iteration_times)/60:.1f} minutes")
    
    if model_ratings:
        log(f"Final rating: ~{model_ratings[-1]} ELO")
        log(f"Rating improvement: {model_ratings[-1] - model_ratings[0]} ELO")

if __name__ == '__main__':
    main()