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
import tempfile

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

    filepath_of_model = f"CurriculumAlphaZeroNet_{args.blocks}x{args.filters}.pt"
    
    cmd = f"python3 train.py --mode supervised --epochs {args.supervised_epochs} --lr {args.supervised_lr} --output {filepath_of_model}"
    
    if args.distributed:
        cmd = f"python3 -m torch.distributed.launch --nproc_per_node={args.gpus} train_distributed.py --mode supervised --epochs {args.supervised_epochs} --lr {args.supervised_lr}"
    
    if args.resume_supervised:
        cmd += f" --resume {args.resume_supervised}"
    
    run_command(cmd)
    log("Phase 1 Complete")
    return filepath_of_model

def phase2_selfplay(args, model_path, iteration):
    """Phase 2: Generate self-play games"""
    log(f"Starting Phase 2: Self-Play Generation (Iteration {iteration})")
    
    output_dir = f"{args.selfplay_dir}/iter_{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if multi-GPU generation is enabled
    if args.selfplay_gpus and args.selfplay_gpus > 1:
        log(f"Using multi-GPU game generation with {args.selfplay_gpus} GPUs")
        
        # Use multi-GPU script
        cmd = f"""python3 create_training_games_multigpu.py \
            --model {model_path} \
            --games-total {args.games_per_iter} \
            --save-format h5 \
            --rollouts {args.rollouts} \
            --temperature {args.temperature} \
            --threads-per-gpu {args.threads} \
            --output-dir {output_dir} \
            --file-base selfplay_iter{iteration} \
            --iteration {iteration}"""
        
        # Add GPU specification if provided
        if args.selfplay_gpu_ids:
            cmd += f" --gpus {args.selfplay_gpu_ids}"
        else:
            cmd += f" --gpus {','.join(map(str, range(args.selfplay_gpus)))}"
        
        # Add CUDA optimization if requested
        if args.use_cuda_mcts:
            cmd += " --use-cuda-mcts"
    else:
        # Single GPU or CPU generation
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
        
        if args.use_cuda_mcts:
            cmd += " --use-cuda-mcts"
    
    run_command(cmd)
    log(f"Generated {args.games_per_iter} games in {output_dir}")
    return output_dir

def phase3_reinforcement(args, model_path, iteration):
    """Phase 3: Reinforcement learning on ALL self-play data with recent games weighted"""
    log(f"Starting Phase 3: Reinforcement Learning (Iteration {iteration})")
    log(f"Training on all games from {args.selfplay_dir} with weight decay {args.weight_decay}")
    
    output_model = f"CurriculumAlphaZeroNet_{args.blocks}x{args.filters}_rl_iter{iteration}.pt"
    
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

def evaluate_model(model_path, reference_model=None, games=100, rollouts=50, threads=40):
    """
    Evaluate a model by playing games against a reference model.
    
    Args:
        model_path: Path to the model to evaluate
        reference_model: Path to the reference model (if None, always returns True)
        games: Number of games to play
        rollouts: MCTS rollouts per move
        threads: Threads for MCTS
    
    Returns:
        tuple: (is_better, win_rate) - whether new model is better and its win rate
    """
    if reference_model is None:
        log(f"No reference model provided, accepting new model: {model_path}")
        return True, 1.0
    
    log(f"Evaluating model: {model_path}")
    log(f"Against reference: {reference_model}")
    log(f"Playing {games} games with {rollouts} rollouts")
    
    # Use JSON output for reliable parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        # Run test_models.py to compare the models
        cmd = f"""python3 test_models.py \
            --model1 {model_path} \
            --model2 {reference_model} \
            --games {games} \
            --rollouts {rollouts} \
            --threads {threads} \
            --alternate \
            --json {json_file}"""
        
        # Capture output to show progress
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        # Read JSON results
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        model1_wins = results['model1_wins']
        model2_wins = results['model2_wins']
        draws = results['draws']
        
        # Calculate win rate for the new model (model1)
        total_decisive = model1_wins + model2_wins
        if total_decisive == 0:
            # All draws, consider it slightly worse to be conservative
            win_rate = 0.45
        else:
            win_rate = model1_wins / total_decisive
        
        # Consider draws as half points
        score_rate = (model1_wins + 0.5 * draws) / games
        
        # New model is better if it wins more than it loses
        is_better = model1_wins > model2_wins
        
        log(f"Evaluation complete: Wins: {model1_wins}, Losses: {model2_wins}, Draws: {draws}")
        log(f"Win rate: {win_rate:.2%}, Score rate: {score_rate:.2%}")
        log(f"New model is {'BETTER' if is_better else 'WORSE'} than reference")
        
        return is_better, win_rate
        
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Model evaluation failed: {e}")
        log(f"STDERR: {e.stderr}")
        # On error, conservatively keep the old model
        return False, 0.0
    finally:
        # Clean up temporary JSON file
        if os.path.exists(json_file):
            os.remove(json_file)

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
    parser.add_argument('--supervised-epochs', type=int, default=100, help='Epochs for supervised learning')
    parser.add_argument('--supervised-lr', type=float, default=0.001, help='Learning rate for supervised')
    parser.add_argument('--resume-supervised', type=str, help='Resume supervised training from checkpoint')
    parser.add_argument('--skip-supervised', action='store_true', help='Skip supervised phase')
    
    # Phase 2: Self-play
    parser.add_argument('--games-per-iter', type=int, default=20000, help='Games per iteration')
    parser.add_argument('--rollouts', type=int, default=20, help='MCTS rollouts per thread')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection')
    parser.add_argument('--threads', type=int, default=40, help='Threads for MCTS')
    parser.add_argument('--selfplay-gpus', type=int, default=1, help='Number of GPUs for self-play generation')
    parser.add_argument('--selfplay-gpu-ids', type=str, help='Specific GPU IDs for self-play (e.g., "0,1,2")')
    parser.add_argument('--use-cuda-mcts', action='store_true', help='Use CUDA-optimized MCTS for game generation')
    
    # Phase 3: Reinforcement learning
    parser.add_argument('--rl-epochs', type=int, default=20, help='Epochs per RL iteration')
    parser.add_argument('--rl-lr', type=float, default=0.01, help='Learning rate for RL')
    parser.add_argument('--iterations', type=int, default=80, help='Number of RL iterations')
    
    # Training configuration
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for distributed training')
    parser.add_argument('--output-dir', type=str, default='curriculum_training', help='Output directory')
    parser.add_argument('--selfplay-dir', type=str, default='games_training_data/selfplay', help='Self-play data directory')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay for recent games (lambda in exp(-lambda * age))')
    
    # Resume training
    parser.add_argument('--resume-state', type=str, help='Resume from saved training state')
    
    # Model evaluation
    parser.add_argument('--eval-games', type=int, default=100, help='Number of games for model evaluation')
    parser.add_argument('--eval-rollouts', type=int, default=50, help='MCTS rollouts per move during evaluation')
    parser.add_argument('--skip-eval', action='store_true', help='Skip model evaluation (always accept new model)')
    
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
        if args.skip_eval:
            log("Skipping evaluation, accepting new model")
            current_model = new_model
        else:
            is_better, win_rate = evaluate_model(
                new_model, 
                current_model,
                games=args.eval_games,
                rollouts=args.eval_rollouts,
                threads=args.threads
            )
            
            # Update current model only if new model is better
            if is_better:
                log(f"New model is better (win rate: {win_rate:.2%}), updating current model")
                current_model = new_model
            else:
                log(f"New model is worse (win rate: {win_rate:.2%}), keeping current model")
                log(f"Current model remains: {current_model}")
        
        # Save training state
        save_training_state(args, iteration + 1, current_model)
        
        log(f"Iteration {iteration + 1} complete")
    
    log("\nCurriculum training complete!")
    log(f"Final model: {current_model}")

if __name__ == '__main__':
    main()