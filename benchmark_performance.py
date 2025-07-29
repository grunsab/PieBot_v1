#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and async MCTS implementations.

This script measures:
- Nodes per second (NPS)
- GPU utilization
- Batch size efficiency
- Response time distribution
"""

import argparse
import time
import torch
import chess
import numpy as np
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
import sys
import os

# Import all implementations
import MCTS
from MCTS_async import MCTSEngine
from MCTS_async_v2 import MCTSEngineV2


def benchmark_original_mcts(model, board, num_rollouts, num_threads, num_runs=5):
    """Benchmark the original MCTS implementation."""
    print("\n" + "="*60)
    print("Benchmarking Original MCTS Implementation")
    print("="*60)
    
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Warm up
        if run == 0:
            print("Warming up...")
            with torch.no_grad():
                root = MCTS.Root(board, model)
                for _ in range(10):
                    root.parallelRollouts(board.copy(), model, num_threads)
        
        # Actual benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            root = MCTS.Root(board, model)
            
            # Perform rollouts
            num_iterations = num_rollouts // num_threads
            remainder = num_rollouts % num_threads
            
            for i in range(num_iterations):
                root.parallelRollouts(board.copy(), model, num_threads)
                
                # Progress indicator
                if (i + 1) % (num_iterations // 10) == 0:
                    print(".", end="", flush=True)
            
            if remainder > 0:
                root.parallelRollouts(board.copy(), model, remainder)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        actual_rollouts = num_iterations * num_threads + remainder
        nps = actual_rollouts / elapsed
        
        results.append({
            'time': elapsed,
            'rollouts': actual_rollouts,
            'nps': nps
        })
        
        print(f"\nTime: {elapsed:.2f}s, NPS: {nps:.0f}")
        
        # Clean up
        root.cleanup()
    
    return results


def benchmark_async_mcts(model, board, num_rollouts, num_threads, batch_size=256, num_runs=5):
    """Benchmark the async MCTS implementation."""
    print("\n" + "="*60)
    print("Benchmarking Async MCTS Implementation")
    print("="*60)
    print(f"Batch size: {batch_size}, Workers: {num_threads}")
    
    results = []
    
    # Create MCTS engine
    device, _ = get_optimal_device()
    engine = MCTSEngine(
        model,
        device=device,
        max_batch_size=batch_size,
        num_workers=num_threads,
        verbose=True
    )
    
    # Start the engine
    engine.start()
    
    try:
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Warm up
            if run == 0:
                print("Warming up...")
                _ = engine.search(board, min(100, num_rollouts // 10))
            
            # Reset statistics
            engine.nn_server.total_requests = 0
            engine.nn_server.total_batches = 0
            engine.nn_server.total_batch_size = 0
            
            # Actual benchmark
            start_time = time.perf_counter()
            
            root = engine.search(board, num_rollouts, return_root=True)
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            nps = num_rollouts / elapsed
            
            # Collect batch statistics
            if engine.nn_server.total_batches > 0:
                avg_batch_size = engine.nn_server.total_batch_size / engine.nn_server.total_batches
                gpu_utilization = (avg_batch_size / batch_size) * 100
            else:
                avg_batch_size = 0
                gpu_utilization = 0
            
            results.append({
                'time': elapsed,
                'rollouts': num_rollouts,
                'nps': nps,
                'avg_batch_size': avg_batch_size,
                'gpu_utilization': gpu_utilization,
                'total_batches': engine.nn_server.total_batches
            })
            
            print(f"Time: {elapsed:.2f}s, NPS: {nps:.0f}")
            print(f"Avg batch size: {avg_batch_size:.1f}, GPU util: {gpu_utilization:.1f}%")
            
            # Clean up root
            root.cleanup()
            
    finally:
        # Stop the engine
        engine.stop()
    
    return results


def print_comparison(original_results, async_results):
    """Print comparison between original and async results."""
    print("\n" + "="*60)
    print("Performance Comparison Summary")
    print("="*60)
    
    # Calculate averages
    orig_avg_nps = np.mean([r['nps'] for r in original_results])
    orig_std_nps = np.std([r['nps'] for r in original_results])
    
    async_avg_nps = np.mean([r['nps'] for r in async_results])
    async_std_nps = np.std([r['nps'] for r in async_results])
    
    speedup = async_avg_nps / orig_avg_nps
    
    print(f"\nOriginal MCTS:")
    print(f"  Average NPS: {orig_avg_nps:.0f} ± {orig_std_nps:.0f}")
    print(f"  Best NPS: {max(r['nps'] for r in original_results):.0f}")
    
    print(f"\nAsync MCTS:")
    print(f"  Average NPS: {async_avg_nps:.0f} ± {async_std_nps:.0f}")
    print(f"  Best NPS: {max(r['nps'] for r in async_results):.0f}")
    
    if 'avg_batch_size' in async_results[0]:
        avg_batch = np.mean([r['avg_batch_size'] for r in async_results])
        avg_gpu_util = np.mean([r['gpu_utilization'] for r in async_results])
        print(f"  Average batch size: {avg_batch:.1f}")
        print(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
    
    print(f"\nSpeedup: {speedup:.1f}x")
    
    # Performance per thread
    if len(original_results) > 0 and len(async_results) > 0:
        orig_nps_per_thread = orig_avg_nps / original_results[0]['rollouts'] * 8  # Assuming 8 threads for original
        async_nps_per_thread = async_avg_nps / async_results[0]['rollouts'] * 32  # Assuming 32 workers for async
        
        print(f"\nEfficiency:")
        print(f"  Original: {orig_nps_per_thread:.1f} NPS per thread")
        print(f"  Async: {async_nps_per_thread:.1f} NPS per worker")


def benchmark_async_v2_mcts(model, board, num_rollouts, num_threads, batch_size=256, num_runs=5):
    """Benchmark the async v2 MCTS implementation."""
    print("\n" + "="*60)
    print("Benchmarking Async V2 MCTS Implementation")
    print("="*60)
    print(f"Batch size: {batch_size}, Workers: {num_threads}")
    
    results = []
    
    # Create MCTS engine
    device, _ = get_optimal_device()
    engine = MCTSEngineV2(
        model,
        device=device,
        max_batch_size=batch_size,
        num_workers=num_threads,
        verbose=True
    )
    
    # Start the engine
    engine.start()
    
    try:
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Warm up
            if run == 0:
                print("Warming up...")
                _ = engine.search(board, min(100, num_rollouts // 10))
            
            # Reset statistics
            engine.nn_server.total_requests = 0
            engine.nn_server.total_batches = 0
            engine.nn_server.total_batch_size = 0
            
            # Actual benchmark
            start_time = time.perf_counter()
            
            root = engine.search(board, num_rollouts, return_root=True)
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            nps = num_rollouts / elapsed
            
            # Collect batch statistics
            if engine.nn_server.total_batches > 0:
                avg_batch_size = engine.nn_server.total_batch_size / engine.nn_server.total_batches
                gpu_utilization = (avg_batch_size / batch_size) * 100
            else:
                avg_batch_size = 0
                gpu_utilization = 0
            
            results.append({
                'time': elapsed,
                'rollouts': num_rollouts,
                'nps': nps,
                'avg_batch_size': avg_batch_size,
                'gpu_utilization': gpu_utilization,
                'total_batches': engine.nn_server.total_batches
            })
            
            print(f"Time: {elapsed:.2f}s, NPS: {nps:.0f}")
            print(f"Avg batch size: {avg_batch_size:.1f}, GPU util: {gpu_utilization:.1f}%")
            
            # Clean up root
            root.cleanup()
            
    finally:
        # Stop the engine
        engine.stop()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark MCTS performance: original vs async implementation'
    )
    parser.add_argument('--model', help='Path to model file', required=True)
    parser.add_argument('--rollouts', type=int, default=1000, 
                       help='Number of rollouts per test')
    parser.add_argument('--original-threads', type=int, default=8,
                       help='Threads for original implementation')
    parser.add_argument('--async-threads', type=int, default=32,
                       help='Worker threads for async implementation')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for async implementation')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of benchmark runs')
    parser.add_argument('--fen', type=str, default=None,
                       help='Starting position in FEN notation')
    parser.add_argument('--skip-original', action='store_true',
                       help='Skip benchmarking original implementation')
    parser.add_argument('--skip-async', action='store_true',
                       help='Skip benchmarking async implementation')
    parser.add_argument('--test-v2', action='store_true',
                       help='Test the v2 async implementation')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device, device_str = get_optimal_device()
    print(f"Device: {device_str}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(args.model, map_location=device)
    model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Set up board
    if args.fen:
        board = chess.Board(args.fen)
    else:
        board = chess.Board()
    
    print(f"\nBenchmarking with {args.rollouts} rollouts, {args.runs} runs each")
    
    # Run benchmarks
    original_results = []
    async_results = []
    async_v2_results = []
    
    if not args.skip_original:
        original_results = benchmark_original_mcts(
            model, board, args.rollouts, args.original_threads, args.runs
        )
    
    if not args.skip_async and not args.test_v2:
        async_results = benchmark_async_mcts(
            model, board, args.rollouts, args.async_threads, 
            args.batch_size, args.runs
        )
    
    if args.test_v2:
        async_v2_results = benchmark_async_v2_mcts(
            model, board, args.rollouts, args.async_threads, 
            args.batch_size, args.runs
        )
    
    # Print comparison
    if original_results and async_results:
        print_comparison(original_results, async_results)
    elif original_results and async_v2_results:
        print("\n" + "="*60)
        print("Performance Comparison Summary (Original vs Async V2)")
        print("="*60)
        
        # Calculate averages
        orig_avg_nps = np.mean([r['nps'] for r in original_results])
        orig_std_nps = np.std([r['nps'] for r in original_results])
        
        v2_avg_nps = np.mean([r['nps'] for r in async_v2_results])
        v2_std_nps = np.std([r['nps'] for r in async_v2_results])
        
        speedup = v2_avg_nps / orig_avg_nps
        
        print(f"\nOriginal MCTS:")
        print(f"  Average NPS: {orig_avg_nps:.0f} ± {orig_std_nps:.0f}")
        print(f"  Best NPS: {max(r['nps'] for r in original_results):.0f}")
        
        print(f"\nAsync V2 MCTS:")
        print(f"  Average NPS: {v2_avg_nps:.0f} ± {v2_std_nps:.0f}")
        print(f"  Best NPS: {max(r['nps'] for r in async_v2_results):.0f}")
        
        if 'avg_batch_size' in async_v2_results[0]:
            avg_batch = np.mean([r['avg_batch_size'] for r in async_v2_results])
            avg_gpu_util = np.mean([r['gpu_utilization'] for r in async_v2_results])
            print(f"  Average batch size: {avg_batch:.1f}")
            print(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
        
        print(f"\nSpeedup: {speedup:.1f}x")
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()