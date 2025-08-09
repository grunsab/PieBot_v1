#!/usr/bin/env python3
"""
Test script to compare performance of original vs optimized parallel inference.

This script measures:
1. Original MCTS_root_parallel with single inference server
2. Optimized MCTS_root_parallel with multiple parallel inference servers
3. Detailed profiling of inference server components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import time
import numpy as np
import argparse
import multiprocessing as mp

import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device


def test_original_implementation(model, board, num_rollouts):
    """Test original MCTS_root_parallel implementation."""
    from MCTS_root_parallel import Root as RootOriginal
    
    print("\n" + "="*80)
    print("TESTING ORIGINAL IMPLEMENTATION (Single Inference Server)")
    print("="*80)
    
    # Warm up
    root = RootOriginal(board, model, epsilon=0.0)
    root.parallelRollouts(board, model, 50)
    RootOriginal.cleanup_engine()
    
    # Actual test
    start_time = time.perf_counter()
    root = RootOriginal(board, model, epsilon=0.0)
    root.parallelRollouts(board, model, num_rollouts)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    throughput = num_rollouts / elapsed
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.0f} rollouts/second")
    
    # Cleanup
    RootOriginal.cleanup_engine()
    
    return elapsed, throughput, root.getStatisticsString()


def test_optimized_implementation(model, board, num_rollouts, num_servers=2):
    """Test optimized MCTS_root_parallel with parallel inference servers."""
    from MCTS_root_parallel_optimized import Root as RootOptimized
    
    print("\n" + "="*80)
    print(f"TESTING OPTIMIZED IMPLEMENTATION ({num_servers} Parallel Inference Servers)")
    print("="*80)
    
    # Warm up
    root = RootOptimized(board, model, epsilon=0.0, use_parallel_servers=True)
    root.parallelRollouts(board, model, 50)
    RootOptimized.cleanup_engine()
    
    # Actual test
    start_time = time.perf_counter()
    root = RootOptimized(board, model, epsilon=0.0, use_parallel_servers=True)
    root.parallelRollouts(board, model, num_rollouts)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    throughput = num_rollouts / elapsed
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.0f} rollouts/second")
    
    # Cleanup
    RootOptimized.cleanup_engine()
    
    return elapsed, throughput, root.getStatisticsString()


def profile_inference_bottleneck(model_path, num_workers=8, num_rollouts=200):
    """Profile the inference server to identify bottlenecks."""
    print("\n" + "="*80)
    print("PROFILING INFERENCE SERVER BOTTLENECKS")
    print("="*80)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "profile_inference_server.py", model_path],
        capture_output=True,
        text=True
    )
    
    # Extract key metrics from output
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if 'Phase' in line or 'encoding' in line or 'nn_inference' in line or 'throughput' in line:
            print(line)


def compare_scaling(model, board, base_rollouts=400):
    """Compare scaling behavior with different numbers of rollouts."""
    print("\n" + "="*80)
    print("SCALING COMPARISON")
    print("="*80)
    
    rollout_counts = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    
    results = {
        'original': [],
        'optimized_2': [],
        'optimized_4': []
    }
    
    for num_rollouts in rollout_counts:
        print(f"\n--- Testing with {num_rollouts} rollouts ---")
        
        # Test original
        elapsed, throughput, _ = test_original_implementation(model, board, num_rollouts)
        results['original'].append({
            'rollouts': num_rollouts,
            'time': elapsed,
            'throughput': throughput
        })
        
        # Test optimized with 2 servers
        from MCTS_root_parallel_optimized import Root as RootOptimized
        from MCTS_root_parallel_optimized import OptimizedRootParallelMCTS
        
        # Force 2 servers
        root = RootOptimized(board, model, epsilon=0.0, use_parallel_servers=True)
        if RootOptimized._mcts_engine:
            RootOptimized._mcts_engine.num_inference_servers = 2
        
        start_time = time.perf_counter()
        root.parallelRollouts(board, model, num_rollouts)
        elapsed = time.perf_counter() - start_time
        
        results['optimized_2'].append({
            'rollouts': num_rollouts,
            'time': elapsed,
            'throughput': num_rollouts / elapsed
        })
        RootOptimized.cleanup_engine()
        
        # Test optimized with 4 servers
        root = RootOptimized(board, model, epsilon=0.0, use_parallel_servers=True)
        if RootOptimized._mcts_engine:
            RootOptimized._mcts_engine.num_inference_servers = 4
        
        start_time = time.perf_counter()
        root.parallelRollouts(board, model, num_rollouts)
        elapsed = time.perf_counter() - start_time
        
        results['optimized_4'].append({
            'rollouts': num_rollouts,
            'time': elapsed,
            'throughput': num_rollouts / elapsed
        })
        RootOptimized.cleanup_engine()
    
    # Print summary table
    print("\n" + "="*80)
    print("SCALING SUMMARY")
    print("="*80)
    print(f"{'Rollouts':<10} | {'Original (1 server)':>20} | {'Optimized (2 servers)':>20} | {'Optimized (4 servers)':>20}")
    print(f"{'':10} | {'Time (s)':>10} {'R/s':>10} | {'Time (s)':>10} {'R/s':>10} | {'Time (s)':>10} {'R/s':>10}")
    print("-"*90)
    
    for i, num_rollouts in enumerate(rollout_counts):
        orig = results['original'][i]
        opt2 = results['optimized_2'][i]
        opt4 = results['optimized_4'][i]
        
        print(f"{num_rollouts:<10} | {orig['time']:>10.2f} {orig['throughput']:>10.0f} | "
              f"{opt2['time']:>10.2f} {opt2['throughput']:>10.0f} | "
              f"{opt4['time']:>10.2f} {opt4['throughput']:>10.0f}")
    
    # Calculate speedups
    print("\n" + "="*80)
    print("SPEEDUP SUMMARY")
    print("="*80)
    print(f"{'Rollouts':<10} | {'2 servers vs 1':>15} | {'4 servers vs 1':>15} | {'4 servers vs 2':>15}")
    print("-"*60)
    
    for i, num_rollouts in enumerate(rollout_counts):
        speedup_2v1 = results['original'][i]['time'] / results['optimized_2'][i]['time']
        speedup_4v1 = results['original'][i]['time'] / results['optimized_4'][i]['time']
        speedup_4v2 = results['optimized_2'][i]['time'] / results['optimized_4'][i]['time']
        
        print(f"{num_rollouts:<10} | {speedup_2v1:>15.2f}x | {speedup_4v1:>15.2f}x | {speedup_4v2:>15.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Test parallel inference optimization')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--rollouts', type=int, default=400, help='Number of rollouts')
    parser.add_argument('--mode', choices=['compare', 'profile', 'scaling'], 
                       default='compare', help='Test mode')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    device, device_str = get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Try different model sizes
    model = None
    for num_blocks, num_channels in [(20, 256), (10, 128)]:
        try:
            model = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_channels)
            weights = torch.load(args.model, map_location=device)
            model.load_state_dict(weights)
            print(f"Loaded {num_blocks}x{num_channels} model")
            break
        except:
            continue
    
    if model is None:
        print("Error: Could not load model")
        sys.exit(1)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    # Test position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    
    print(f"\nTest position after: e4 e5 Nf3 Nc6")
    print(board)
    
    if args.mode == 'compare':
        # Compare original vs optimized
        print(f"\nComparing implementations with {args.rollouts} rollouts...")
        
        orig_time, orig_throughput, orig_stats = test_original_implementation(
            model, board, args.rollouts
        )
        
        opt_time, opt_throughput, opt_stats = test_optimized_implementation(
            model, board, args.rollouts, num_servers=2
        )
        
        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Implementation':<30} {'Time (s)':>15} {'Throughput (r/s)':>20}")
        print("-"*65)
        print(f"{'Original (1 server)':<30} {orig_time:>15.2f} {orig_throughput:>20.0f}")
        print(f"{'Optimized (2 servers)':<30} {opt_time:>15.2f} {opt_throughput:>20.0f}")
        print("-"*65)
        print(f"{'Speedup':<30} {orig_time/opt_time:>15.2f}x")
        print(f"{'Throughput Improvement':<30} {opt_throughput/orig_throughput:>15.2f}x")
        
    elif args.mode == 'profile':
        # Profile inference server
        profile_inference_bottleneck(args.model)
        
    elif args.mode == 'scaling':
        # Test scaling behavior
        compare_scaling(model, board, args.rollouts)


if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows
    mp.set_start_method('spawn', force=True)
    main()