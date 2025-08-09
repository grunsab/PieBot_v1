"""
Profile MCTS_root_parallel.py with 100,000 rollouts to identify bottlenecks.
"""

import cProfile
import pstats
import io
import time
import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_root_parallel import RootParallelMCTS
import device_utils

def profile_mcts():
    # Load model
    device, device_str = device_utils.get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Load the model (using the 20x256 model for more realistic profiling)
    model_path = "weights/AlphaZeroNet_20x256.pt"
    try:
        model = AlphaZeroNet(num_blocks=20, num_filters=256)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        # Fallback to 10x128 if 20x256 not available
        model_path = "weights/AlphaZeroNet_10x128.pt"
        try:
            model = AlphaZeroNet(num_blocks=10, num_filters=128)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Loaded model from {model_path}")
        except:
            # Create a dummy model if no weights available
            print("No model weights found, using untrained model")
            model = AlphaZeroNet(num_blocks=10, num_filters=128).to(device)
            model.eval()
    
    # Initialize board
    board = chess.Board()
    
    # Create MCTS engine with reasonable parameters
    num_workers = 8  # Reasonable number of workers
    epsilon = 0.0  # No noise for deterministic profiling
    inference_batch_size = 256  # Larger batch size for efficiency
    inference_timeout_ms = 100  # Shorter timeout for faster batching
    
    print(f"\nInitializing MCTS with {num_workers} workers")
    print(f"Inference batch size: {inference_batch_size}")
    print(f"Inference timeout: {inference_timeout_ms}ms")
    
    mcts = RootParallelMCTS(
        model=model,
        num_workers=num_workers,
        epsilon=epsilon,
        alpha=0.3,
        inference_batch_size=inference_batch_size,
        inference_timeout_ms=inference_timeout_ms
    )
    
    # Total rollouts to perform
    total_rollouts = 100_000
    
    print(f"\nStarting profiling with {total_rollouts:,} rollouts...")
    print("=" * 60)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    start_time = time.time()
    profiler.enable()
    
    # Run the search
    try:
        stats = mcts.run_parallel_search(board, total_rollouts)
        
        # Get best move
        best_move = mcts.get_best_move(stats)
        
    finally:
        profiler.disable()
        end_time = time.time()
        
        # Clean up
        mcts.cleanup()
    
    # Calculate performance metrics
    total_time = end_time - start_time
    rollouts_per_second = total_rollouts / total_time
    
    print(f"\nPerformance Summary:")
    print(f"=" * 60)
    print(f"Total rollouts: {total_rollouts:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Rollouts per second: {rollouts_per_second:.0f}")
    print(f"Microseconds per rollout: {(total_time * 1_000_000) / total_rollouts:.2f}")
    
    if stats:
        print(f"\nMove statistics (top 5):")
        sorted_moves = sorted(stats.items(), key=lambda x: x[1]['visits'], reverse=True)[:5]
        for move, move_stats in sorted_moves:
            print(f"  {move}: {move_stats['visits']:,} visits, Q={move_stats['q_value']:.4f}")
    
    # Print profiling results
    print(f"\n{'=' * 60}")
    print("Profiling Results (Top 30 Functions by Cumulative Time):")
    print("=" * 60)
    
    # Sort by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Also show by total time
    print(f"\n{'=' * 60}")
    print("Profiling Results (Top 30 Functions by Total Time):")
    print("=" * 60)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Show callers of expensive functions
    print(f"\n{'=' * 60}")
    print("Callers of Key Functions:")
    print("=" * 60)
    
    # Focus on key functions we care about
    key_functions = [
        'run_parallel_search',
        'run_independent_search', 
        'run_parallel_rollouts',
        'prepare_rollout',
        'apply_evaluation_and_backpropagate',
        'backpropagate',
        'UCTSelect',
        'expand',
        'get',  # Queue operations
        'put',
    ]
    
    for func in key_functions:
        print(f"\nCallers of '{func}':")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.print_callers(func, 5)
        output = s.getvalue()
        if output.strip():
            print(output)

if __name__ == "__main__":
    profile_mcts()