#!/usr/bin/env python3
"""
Test and benchmark script for the persistent multi-process MCTS implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import chess
import torch
import multiprocessing as mp
import argparse

import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
import MCTS_profiling_speedups_v2 as MCTS_threaded
import MCTS_multiprocess as MCTS_mp

def load_model(model_path):
    """Load the neural network model."""
    device, device_str = get_optimal_device()
    print(f"Loading model on {device_str}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
        if weights.get('model_type') == 'fp16':
            model = model.half()
    else:
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, device

def benchmark_threaded(board, model, rollouts, threads):
    """Benchmark threaded MCTS."""
    print(f"\nBenchmarking threaded MCTS with {rollouts} rollouts on {threads} threads...")
    
    start_time = time.time()
    
    root = MCTS_threaded.Root(board, model)
    num_iterations = max(1, rollouts // threads)
    remainder = rollouts % threads
    
    for _ in range(num_iterations):
        root.parallelRollouts(board.copy(), model, threads)
    if remainder > 0:
        root.parallelRollouts(board.copy(), model, remainder)
    
    elapsed = time.time() - start_time
    
    edge = root.maxNSelect()
    best_move = edge.getMove() if edge else None
    q_value = root.getQ() if edge else 0.5
    
    MCTS_threaded.clear_caches()
    MCTS_threaded.clear_pools()
    
    return elapsed, best_move, q_value

def benchmark_multiprocess(board, model, rollouts, processes=None):
    """Benchmark the new persistent multi-process MCTS."""
    if processes is None:
        processes = max(1, mp.cpu_count() - 2)
        
    print(f"\nBenchmarking persistent multi-process MCTS with {rollouts} rollouts on {processes} processes...")
    
    # The new MCTS engine is persistent. We initialize it once.
    # The Root class now manages a singleton engine instance.
    mcts_root = MCTS_mp.Root(board, model)

    start_time = time.time()
    mcts_root.parallelRollouts(board.copy(), model, rollouts)
    elapsed = time.time() - start_time
    
    edge = mcts_root.maxNSelect()
    best_move = edge.getMove() if edge else None
    q_value = mcts_root.getQ() if edge else 0.5
    
    # The cleanup of the persistent engine is handled by atexit in the module itself.
    # We don't call cleanup() here because we might want to run more tests.
    
    return elapsed, best_move, q_value

def run_benchmark(model_path, rollouts, threads, processes):
    """Run benchmark comparison."""
    model, device = load_model(model_path)
    
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    for move_str in moves:
        board.push(chess.Move.from_uci(move_str))
    
    print(f"Test position FEN: {board.fen()}")

    # --- Threaded Benchmark ---
    elapsed_t, move_t, q_t = benchmark_threaded(board, model, rollouts, threads)
    rps_t = rollouts / elapsed_t if elapsed_t > 0 else 0

    # --- Multi-process Benchmark ---
    elapsed_m, move_m, q_m = benchmark_multiprocess(board, model, rollouts, processes)
    rps_m = rollouts / elapsed_m if elapsed_m > 0 else 0

    # --- Print Results ---
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nThreaded MCTS ({threads} threads):")
    print(f"  Time: {elapsed_t:.2f}s")
    print(f"  Rollouts/sec: {rps_t:.0f}")
    print(f"  Best move: {move_t}")
    print(f"  Q value: {q_t:.4f}")
    
    print(f"\nPersistent Multi-process MCTS ({processes} processes):")
    print(f"  Time: {elapsed_m:.2f}s")
    print(f"  Rollouts/sec: {rps_m:.0f}")
    print(f"  Best move: {move_m}")
    print(f"  Q value: {q_m:.4f}")
    
    if rps_t > 0:
        speedup = rps_m / rps_t
        print(f"\nSpeedup: {speedup:.2f}x")
    
    if move_t == move_m:
        print("\n✓ Both methods chose the same move.")
    else:
        print(f"\n✗ Methods chose different moves: {move_t} vs {move_m}")

    q_diff = abs(q_t - q_m)
    if q_diff < 0.05:
        print(f"✓ Q values are similar (diff: {q_diff:.4f})")
    else:
        print(f"✗ Q values differ significantly (diff: {q_diff:.4f})")

    # Explicitly clean up the multiprocess engine at the end of the script
    MCTS_mp.Root.cleanup_engine()

def main():
    """Main entry point."""
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Test and benchmark the persistent multi-process MCTS"
    )
    parser.add_argument("--model", help="Path to model file", default="weights/AlphaZeroNet_20x256.pt")
    parser.add_argument("--rollouts", type=int, default=2000, help="Number of rollouts")
    parser.add_argument("--threads", type=int, default=64, help="Number of threads for threaded version")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes for multi-process version (default: auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}", file=sys.stderr)
        sys.exit(1)
    
    run_benchmark(args.model, args.rollouts, args.threads, args.processes)

if __name__ == "__main__":
    main()
