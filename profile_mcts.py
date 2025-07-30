#!/usr/bin/env python3
"""
Profile MCTS and chess engine performance to identify CPU bottlenecks.

Usage:
    python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 100 --threads 1
    python profile_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 100 --threads 10 --output profile_results.txt
"""

import argparse
import cProfile
import pstats
import io
import sys
import os
from contextlib import contextmanager
import time

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import playchess
import chess
import MCTS
import torch
import AlphaZeroNetwork
from device_utils import get_optimal_device

@contextmanager
def profile_context(sort_by='cumulative', limit=50):
    """Context manager for profiling code sections."""
    pr = cProfile.Profile()
    pr.enable()
    yield pr
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats(limit)
    return s.getvalue()

def profile_single_move(model_file, num_rollouts, num_threads, verbose=False):
    """Profile a single move calculation to identify bottlenecks."""
    
    # Load model
    models, devices = playchess.load_model_multi_gpu(model_file, None)
    alphaZeroNet = models[0]
    device = devices[0]
    
    # Create board
    board = chess.Board()
    
    print(f"\nProfiling MCTS with {num_rollouts} rollouts and {num_threads} threads...")
    print(f"Model on device: {device}")
    print("-" * 80)
    
    # Profile the entire move calculation
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        root = MCTS.Root(board, alphaZeroNet)
        
        for i in range(num_rollouts):
            root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)
    
    end_time = time.perf_counter()
    
    pr.disable()
    
    # Calculate statistics
    elapsed = end_time - start_time
    total_nodes = root.getN()
    nps = total_nodes / elapsed
    
    print(f"\nPerformance Statistics:")
    print(f"Total rollouts: {int(total_nodes)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Nodes per second: {nps:.2f}")
    print(f"Duplicate paths: {root.same_paths}")
    
    # Print profile results
    print("\n" + "="*80)
    print("PROFILE RESULTS (sorted by cumulative time):")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    profile_output = s.getvalue()
    print(profile_output)
    
    # Additional analysis - breakdown by function
    print("\n" + "="*80)
    print("TOP TIME-CONSUMING FUNCTIONS (sorted by total time):")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Analyze specific MCTS functions
    print("\n" + "="*80)
    print("MCTS-SPECIFIC FUNCTIONS:")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats('MCTS.py:', 20)
    print(s.getvalue())
    
    # Analyze encoder functions
    print("\n" + "="*80)
    print("ENCODER-SPECIFIC FUNCTIONS:")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats('encoder.py:', 20)
    print(s.getvalue())
    
    return profile_output, {
        'total_nodes': total_nodes,
        'elapsed': elapsed,
        'nps': nps,
        'duplicate_paths': root.same_paths
    }

def profile_components(model_file):
    """Profile individual components of the engine."""
    
    print("\n" + "="*80)
    print("COMPONENT-WISE PROFILING")
    print("="*80)
    
    # Load model
    models, devices = playchess.load_model_multi_gpu(model_file, None)
    alphaZeroNet = models[0]
    
    # Create board
    board = chess.Board()
    
    # Profile position encoding
    print("\n1. Position Encoding Performance:")
    import encoder
    
    start = time.perf_counter()
    for _ in range(1000):
        position, mask = encoder.encodePositionForInference(board)
    encoding_time = time.perf_counter() - start
    print(f"   1000 position encodings: {encoding_time:.3f}s ({1000/encoding_time:.1f} encodings/sec)")
    
    # Profile neural network inference
    print("\n2. Neural Network Inference:")
    position, mask = encoder.encodePositionForInference(board)
    
    # Single inference
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            value, move_probs = encoder.callNeuralNetwork(board, alphaZeroNet)
    single_time = time.perf_counter() - start
    print(f"   100 single inferences: {single_time:.3f}s ({100/single_time:.1f} inferences/sec)")
    
    # Batch inference
    boards = [board.copy() for _ in range(10)]
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            values, move_probs = encoder.callNeuralNetworkBatched(boards, alphaZeroNet)
    batch_time = time.perf_counter() - start
    print(f"   10 batches of 10: {batch_time:.3f}s ({100/batch_time:.1f} inferences/sec)")
    
    # Profile UCT calculations
    print("\n3. UCT Calculation Performance:")
    root = MCTS.Root(board, alphaZeroNet)
    
    # Do one rollout to create some edges
    root.rollout(board.copy(), alphaZeroNet)
    
    if root.edges:
        edge = root.edges[0]
        start = time.perf_counter()
        for _ in range(100000):
            uct = MCTS.calcUCT(edge, root.N)
        uct_time = time.perf_counter() - start
        print(f"   100,000 UCT calculations: {uct_time:.3f}s ({100000/uct_time:.1f} calcs/sec)")
    
    # Profile tree traversal
    print("\n4. Tree Traversal (selection phase):")
    # Build a deeper tree
    for _ in range(10):
        root.parallelRollouts(board.copy(), alphaZeroNet, 1)
    
    start = time.perf_counter()
    for _ in range(1000):
        node_path = []
        edge_path = []
        test_board = board.copy()
        root.selectTask(test_board, node_path, edge_path)
    select_time = time.perf_counter() - start
    print(f"   1000 tree traversals: {select_time:.3f}s ({1000/select_time:.1f} traversals/sec)")

def main():
    parser = argparse.ArgumentParser(description='Profile MCTS performance')
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of rollouts')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads per rollout')
    parser.add_argument('--output', help='Output file for profile results')
    parser.add_argument('--components', action='store_true', help='Profile individual components')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run profiling
    if args.components:
        profile_components(args.model)
    
    profile_output, stats = profile_single_move(
        args.model, 
        args.rollouts, 
        args.threads, 
        args.verbose
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Profile Results\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Rollouts: {args.rollouts}\n")
            f.write(f"Threads: {args.threads}\n")
            f.write(f"Stats: {stats}\n\n")
            f.write(profile_output)
        print(f"\nProfile results saved to {args.output}")

if __name__ == '__main__':
    main()