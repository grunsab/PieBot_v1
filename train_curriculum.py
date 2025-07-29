#!/usr/bin/env python3
"""
Curriculum learning script for AlphaZero training.
Automatically manages the transition from supervised to reinforcement learning.
"""

import os
import subprocess
import argparse
import time
import json
from datetime import datetime

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(cmd, check=True):
    """Run a command and stream output to stdout"""
    log(f"Running: {cmd}")
    # Use subprocess.Popen to stream output in real-time
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Stream output line by line
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    
    # Wait for process to complete
    returncode = process.wait()
    
    if check and returncode != 0:
        raise RuntimeError(f"Command failed with return code {returncode}: {cmd}")
    
    return returncode

def phase1_supervised(args):
    """Phase 1: Supervised learning on CCRL dataset"""
    log("Starting Phase 1: Supervised Learning")
    
    cmd = f"python3 train.py --mode supervised --epochs {args.supervised_epochs} --lr {args.supervised_lr}"
    
    if args.distributed:
        cmd = f"python3 -m torch.distributed.launch --nproc_per_node={args.gpus} train_distributed.py --mode supervised --epochs {args.supervised_epochs} --lr {args.supervised_lr}"
    
    if args.resume_supervised:
        cmd += f" --resume {args.resume_supervised}"
    
    run_command(cmd)
    log("Phase 1 Complete")
    return f"AlphaZeroNet_{args.blocks}x{args.filters}.pt"

def phase2_selfplay(args, model_path, iteration):
    """Phase 2: Generate self-play games"""
    log(f"Starting Phase 2: Self-Play Generation (Iteration {iteration})")
    
    output_dir = f"{args.selfplay_dir}/iter_{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"""python3 create_training_games.py \
        --model {model_path} \
        --save-format h5 \
        --rollouts {args.rollouts} \
        --temperature {args.temperature} \
        --games-to-play {args.games_per_iter} \
        --threads {args.threads} \
        --output-dir {output_dir} \
        --file-base selfplay_iter{iteration} \
        --iteration {iteration}"""
    
    run_command(cmd)
    log(f"Generated {args.games_per_iter} games in {output_dir}")
    return output_dir

def phase3_reinforcement(args, model_path, iteration):
    """Phase 3: Reinforcement learning on ALL self-play data with recent games weighted"""
    log(f"Starting Phase 3: Reinforcement Learning (Iteration {iteration})")
    log(f"Training on all games from {args.selfplay_dir} with weight decay {args.weight_decay}")
    
    output_model = f"AlphaZeroNet_{args.blocks}x{args.filters}_rl_iter{iteration}.pt"
    
    cmd = f"""python3 train.py \
        --mode rl \
        --resume {model_path} \
        --rl-dir {args.selfplay_dir} \
        --epochs {args.rl_epochs} \
        --lr {args.rl_lr} \
        --output {output_model} \
        --rl-weight-recent \
        --rl-weight-decay {args.weight_decay}"""
    
    if args.distributed:
        cmd = f"""python3 -m torch.distributed.launch --nproc_per_node={args.gpus} train_distributed.py \
            --mode rl \
            --resume {model_path} \
            --rl-dir {args.selfplay_dir} \
            --epochs {args.rl_epochs} \
            --lr {args.rl_lr} \
            --output {output_model} \
            --rl-weight-recent \
            --rl-weight-decay {args.weight_decay}"""
    
    run_command(cmd)
    log(f"Phase 3 Complete: {output_model}")
    return output_model

def evaluate_model(model_path, reference_model=None):
    """Simple evaluation by playing games against reference"""
    # This is a placeholder - implement actual evaluation
    log(f"Evaluating model: {model_path}")
    # In practice, you would play games against reference engines
    # and estimate ELO rating
    return 2800  # Dummy rating

def save_training_state(args, iteration, current_model):
    """Save current training state for resumption"""
    state = {
        'iteration': iteration,
        'current_model': current_model,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
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
    parser = argparse.ArgumentParser(description='Curriculum learning for AlphaZero')
    
    # Model architecture
    parser.add_argument('--blocks', type=int, default=20, help='Number of residual blocks')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters')
    
    # Phase 1: Supervised learning
    parser.add_argument('--supervised-epochs', type=int, default=20, help='Epochs for supervised learning')
    parser.add_argument('--supervised-lr', type=float, default=0.0005, help='Learning rate for supervised')
    parser.add_argument('--resume-supervised', type=str, help='Resume supervised training from checkpoint')
    parser.add_argument('--skip-supervised', action='store_true', help='Skip supervised phase')
    
    # Phase 2: Self-play
    parser.add_argument('--games-per-iter', type=int, default=15000, help='Games per iteration')
    parser.add_argument('--rollouts', type=int, default=50, help='MCTS rollouts per thread')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection')
    parser.add_argument('--threads', type=int, default=20, help='Threads for MCTS')
    
    # Phase 3: Reinforcement learning
    parser.add_argument('--rl-epochs', type=int, default=20, help='Epochs per RL iteration')
    parser.add_argument('--rl-lr', type=float, default=0.0005, help='Learning rate for RL')
    parser.add_argument('--iterations', type=int, default=80, help='Number of RL iterations')
    
    # Training configuration
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs for distributed training')
    parser.add_argument('--output-dir', type=str, default='curriculum_training', help='Output directory')
    parser.add_argument('--selfplay-dir', type=str, default='games_training_data/selfplay', help='Self-play data directory')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay for recent games (lambda in exp(-lambda * age))')
    
    # Resume training
    parser.add_argument('--resume-state', type=str, help='Resume from saved training state')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.selfplay_dir, exist_ok=True)
    
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
        current_model = phase1_supervised(args)
        save_training_state(args, 0, current_model)
    elif current_model is None:
        # Must specify initial model if skipping supervised
        raise ValueError("Must specify --resume-supervised or --resume-state when using --skip-supervised")
    
    # Iterative RL training
    for iteration in range(start_iteration, args.iterations):
        log(f"\n{'='*60}")
        log(f"Starting Iteration {iteration + 1}/{args.iterations}")
        log(f"Current model: {current_model}")
        
        # Phase 2: Generate self-play games
        data_dir = phase2_selfplay(args, current_model, iteration + 1)
        
        # Phase 3: Train on ALL self-play data with recent games weighted
        new_model = phase3_reinforcement(args, current_model, iteration + 1)
        
        # Evaluate new model
        rating = evaluate_model(new_model, current_model)
        log(f"Model rating: ~{rating} ELO")
        
        # Update current model
        current_model = new_model
        
        # Save training state
        save_training_state(args, iteration + 1, current_model)
        
        log(f"Iteration {iteration + 1} complete")
    
    log("\nCurriculum training complete!")
    log(f"Final model: {current_model}")

if __name__ == '__main__':
    main()