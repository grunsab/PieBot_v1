#!/usr/bin/env python3
"""
Test script for root parallel MCTS implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import time
import AlphaZeroNetwork
from MCTS_root_parallel import Root as RootParallel
from MCTS_multiprocess import Root as RootMultiprocess
import argparse


def test_basic_functionality():
    """Test that root parallel MCTS works with basic operations."""
    print("Testing basic functionality...")
    
    # Load a model
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    try:
        model.load_state_dict(torch.load('weights/AlphaZeroNet_10x128.pt', map_location='cpu'))
        print("Loaded 10x128 model")
    except:
        print("Warning: Could not load weights, using random initialization")
    
    model.eval()
    
    # Create board
    board = chess.Board()
    
    # Test with Dirichlet noise (training mode)
    print("\nTesting with Dirichlet noise (epsilon=0.25)...")
    root = RootParallel(board, model, epsilon=0.25)
    root.parallelRollouts(board, model, 100)
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print("\nMove statistics:")
    print(root.getStatisticsString())
    
    # Test without noise (play mode)
    print("\nTesting without Dirichlet noise (epsilon=0.0)...")
    RootParallel.cleanup_engine()  # Clean up previous engine
    
    root2 = RootParallel(board, model, epsilon=0.0)
    root2.parallelRollouts(board, model, 100)
    
    best_edge2 = root2.maxNSelect()
    if best_edge2:
        print(f"Best move: {best_edge2.getMove()}")
        print(f"Visits: {best_edge2.getN()}")
        print(f"Q-value: {best_edge2.getQ():.4f}")
    
    print("\nMove statistics:")
    print(root2.getStatisticsString())
    
    # Clean up
    RootParallel.cleanup_engine()
    print("\nBasic functionality test passed!")


def compare_implementations(model_path, num_rollouts=400):
    """Compare root parallel vs tree parallel implementations."""
    print(f"\nComparing implementations with {num_rollouts} rollouts...")
    
    # Load model
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    except:
        print("Warning: Could not load weights, using random initialization")
    
    model.eval()
    
    # Test position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    
    print(f"\nTesting position after: e4 e5 Nf3 Nc6")
    print(board)
    
    # Test root parallel (without noise for fair comparison)
    print("\n--- Root Parallel MCTS ---")
    start_time = time.time()
    root_parallel = RootParallel(board, model, epsilon=0.0)
    root_parallel.parallelRollouts(board, model, num_rollouts)
    root_parallel_time = time.time() - start_time
    
    best_edge = root_parallel.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Time: {root_parallel_time:.2f}s")
        print(f"Rollouts/sec: {num_rollouts/root_parallel_time:.1f}")
    
    print("\nTop moves:")
    print(root_parallel.getStatisticsString())
    
    # Clean up root parallel
    RootParallel.cleanup_engine()
    
    # Test tree parallel (multiprocess)
    print("\n--- Tree Parallel MCTS (Multiprocess) ---")
    start_time = time.time()
    root_multiprocess = RootMultiprocess(board, model)
    root_multiprocess.parallelRollouts(board, model, num_rollouts)
    multiprocess_time = time.time() - start_time
    
    best_edge = root_multiprocess.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Time: {multiprocess_time:.2f}s")
        print(f"Rollouts/sec: {num_rollouts/multiprocess_time:.1f}")
    
    print("\nTop moves:")
    print(root_multiprocess.getStatisticsString())
    
    # Clean up multiprocess
    RootMultiprocess.cleanup_engine()
    
    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Root Parallel: {root_parallel_time:.2f}s ({num_rollouts/root_parallel_time:.1f} rollouts/sec)")
    print(f"Tree Parallel: {multiprocess_time:.2f}s ({num_rollouts/multiprocess_time:.1f} rollouts/sec)")
    speedup = multiprocess_time / root_parallel_time
    print(f"Speedup: {speedup:.2f}x")


def test_diversity_with_noise():
    """Test that Dirichlet noise creates diverse exploration."""
    print("\nTesting exploration diversity with Dirichlet noise...")
    
    # Load model
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    model.eval()
    
    board = chess.Board()
    
    # Run multiple searches with noise
    print("\nRunning 5 searches with epsilon=0.25...")
    move_counts = {}
    
    for i in range(5):
        RootParallel.cleanup_engine()
        root = RootParallel(board, model, epsilon=0.25)
        root.parallelRollouts(board, model, 100)
        
        # Count visit distribution
        if root.stats:
            for move, stats in root.stats.items():
                if move not in move_counts:
                    move_counts[move] = []
                move_counts[move].append(stats['visits'])
    
    print("\nMove visit distributions across 5 runs:")
    for move, visits_list in sorted(move_counts.items(), key=lambda x: sum(x[1]), reverse=True)[:5]:
        avg_visits = sum(visits_list) / len(visits_list)
        std_visits = np.std(visits_list) if len(visits_list) > 1 else 0
        print(f"{move}: avg={avg_visits:.1f}, std={std_visits:.1f}, runs={len(visits_list)}")
    
    # Clean up
    RootParallel.cleanup_engine()
    
    # Compare with deterministic (no noise)
    print("\nRunning 5 searches with epsilon=0.0 (no noise)...")
    move_counts_det = {}
    
    for i in range(5):
        RootParallel.cleanup_engine()
        root = RootParallel(board, model, epsilon=0.0)
        root.parallelRollouts(board, model, 100)
        
        if root.stats:
            for move, stats in root.stats.items():
                if move not in move_counts_det:
                    move_counts_det[move] = []
                move_counts_det[move].append(stats['visits'])
    
    print("\nMove visit distributions without noise:")
    for move, visits_list in sorted(move_counts_det.items(), key=lambda x: sum(x[1]), reverse=True)[:5]:
        avg_visits = sum(visits_list) / len(visits_list)
        std_visits = np.std(visits_list) if len(visits_list) > 1 else 0
        print(f"{move}: avg={avg_visits:.1f}, std={std_visits:.1f}, runs={len(visits_list)}")
    
    RootParallel.cleanup_engine()
    print("\nDiversity test completed!")


if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/AlphaZeroNet_10x128.pt", 
                       help="Model path")
    parser.add_argument("--rollouts", type=int, default=400,
                       help="Number of rollouts for comparison")
    parser.add_argument("--test", default="all",
                       choices=["basic", "compare", "diversity", "all"],
                       help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ["basic", "all"]:
        test_basic_functionality()
    
    if args.test in ["compare", "all"]:
        compare_implementations(args.model, args.rollouts)
    
    if args.test in ["diversity", "all"]:
        test_diversity_with_noise()
    
    print("\nAll tests completed!")