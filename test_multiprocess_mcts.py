#!/usr/bin/env python3
"""
Test and benchmark script for multi-process MCTS implementation.
"""

import time
import chess
import torch
import multiprocessing as mp
import argparse
import os
import sys

import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
import MCTS_profiling_speedups_v2 as MCTS_threaded
import MCTS_multiprocess as MCTS_mp  # Fallback to threaded for now


def load_model(model_path):
    """Load the neural network model."""
    device, device_str = get_optimal_device()
    print(f"Loading model on {device_str}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    
    # Handle different model formats
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
        if weights.get('model_type') == 'fp16':
            model = model.half()
    else:
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
        
    return model, device


def benchmark_threaded(board, model, rollouts, threads):
    """Benchmark threaded MCTS."""
    print(f"\nBenchmarking threaded MCTS with {rollouts} rollouts on {threads} threads...")
    
    start_time = time.time()
    
    # Create root and run rollouts
    root = MCTS_threaded.Root(board, model)
    
    num_iterations = max(1, rollouts // threads)
    remainder = rollouts % threads
    
    for i in range(num_iterations):
        root.parallelRollouts(board.copy(), model, threads)
        
    if remainder > 0:
        root.parallelRollouts(board.copy(), model, remainder)
    
    elapsed = time.time() - start_time
    
    # Get best move
    edge = root.maxNSelect()
    if edge:
        best_move = edge.getMove()
        Q = root.getQ()
    else:
        best_move = None
        Q = 0.5
    
    # Cleanup
    if hasattr(root, 'cleanup'):
        root.cleanup()
    if hasattr(MCTS_threaded, 'clear_caches'):
        MCTS_threaded.clear_caches()
    if hasattr(MCTS_threaded, 'clear_pools'):
        MCTS_threaded.clear_pools()
    
    return elapsed, best_move, Q, root.same_paths


def benchmark_multiprocess(board, model, rollouts, processes=None):
    """Benchmark multi-process MCTS."""
    if processes is None:
        processes = max(1, mp.cpu_count() - 2)
        
    print(f"\nBenchmarking with {rollouts} rollouts...")
    
    start_time = time.time()
    
    # Create root and run rollouts
    root = MCTS_mp.Root(board, model)
    root.parallelRollouts(board.copy(), model, rollouts)
    
    elapsed = time.time() - start_time
    
    # Get best move
    edge = root.maxNSelect()
    if edge:
        best_move = edge.getMove()
        Q = root.getQ()
    else:
        best_move = None
        Q = 0.5
    
    # Cleanup
    root.cleanup()
    
    return elapsed, best_move, Q, root.same_paths


def run_benchmark(model_path, rollouts, threads, processes):
    """Run benchmark comparison."""
    # Load model
    model, device = load_model(model_path)
    
    # Create test position
    board = chess.Board()
    
    # Make some moves to get to a middle game position
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
    for move_str in moves:
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
        except:
            print(f"Failed to apply move: {move_str}")
    
    print(f"\nTest position after moves: {' '.join(moves)}")
    print(board)
    
    # Run benchmarks
    results = {}
    
    # Threaded version
    elapsed_t, move_t, q_t, same_t = benchmark_threaded(board, model, rollouts, threads)
    results['threaded'] = {
        'time': elapsed_t,
        'rollouts_per_sec': rollouts / elapsed_t,
        'move': move_t,
        'Q': q_t,
        'same_paths': same_t
    }
    
    # Multi-process version
    elapsed_m, move_m, q_m, same_m = benchmark_multiprocess(board, model, rollouts, processes)
    results['multiprocess'] = {
        'time': elapsed_m,
        'rollouts_per_sec': rollouts / elapsed_m,
        'move': move_m,
        'Q': q_m,
        'same_paths': same_m
    }
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nThreaded MCTS ({threads} threads):")
    print(f"  Time: {results['threaded']['time']:.2f}s")
    print(f"  Rollouts/sec: {results['threaded']['rollouts_per_sec']:.0f}")
    print(f"  Best move: {results['threaded']['move']}")
    print(f"  Q value: {results['threaded']['Q']:.4f}")
    print(f"  Same paths: {results['threaded']['same_paths']}")
    
    print(f"\nMulti-process MCTS ({processes} processes):")
    print(f"  Time: {results['multiprocess']['time']:.2f}s")
    print(f"  Rollouts/sec: {results['multiprocess']['rollouts_per_sec']:.0f}")
    print(f"  Best move: {results['multiprocess']['move']}")
    print(f"  Q value: {results['multiprocess']['Q']:.4f}")
    print(f"  Same paths: {results['multiprocess']['same_paths']}")
    
    print(f"\nSpeedup: {results['multiprocess']['rollouts_per_sec'] / results['threaded']['rollouts_per_sec']:.2f}x")
    
    # Check if moves agree
    if results['threaded']['move'] == results['multiprocess']['move']:
        print("\n✓ Both methods chose the same move")
    else:
        print("\n✗ Methods chose different moves!")
        print(f"  This may be due to different exploration patterns")
    
    # Check Q values
    q_diff = abs(results['threaded']['Q'] - results['multiprocess']['Q'])
    if q_diff < 0.05:
        print(f"✓ Q values are similar (diff: {q_diff:.4f})")
    else:
        print(f"✗ Q values differ significantly (diff: {q_diff:.4f})")


def main():
    """Main entry point."""
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Test and benchmark multi-process MCTS"
    )
    parser.add_argument("--model", help="Path to model file",
                       default="weights/AlphaZeroNet_20x256.pt")
    parser.add_argument("--rollouts", type=int, default=10000,
                       help="Number of rollouts")
    parser.add_argument("--threads", type=int, default=64,
                       help="Number of threads for threaded version")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes for multi-process version (default: auto)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    run_benchmark(args.model, args.rollouts, args.threads, args.processes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        import gc
        gc.collect()