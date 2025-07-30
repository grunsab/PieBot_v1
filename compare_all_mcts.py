#!/usr/bin/env python3
"""
Compare performance between original, optimized, advanced, and multiprocess MCTS implementations.

Usage:
    python compare_all_mcts.py --model AlphaZeroNet_20x256_distributed.pt
    
    # Skip multiprocessing benchmark (useful for quick tests)
    python compare_all_mcts.py --model AlphaZeroNet_20x256_distributed.pt --skip-multiprocess
    
    # Test with high rollout count where multiprocessing shines
    python compare_all_mcts.py --model AlphaZeroNet_20x256_distributed.pt --rollouts 1000 --processes 4
"""

import argparse
import time
import chess
import torch
import sys
import os
import gc
import multiprocessing as mp

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MCTS
import MCTS_profiling_speedups as MCTS_opt
import MCTS_advanced_optimizations as MCTS_adv
import MCTS_multiprocess
from playchess import load_model_multi_gpu

def benchmark_mcts(mcts_module, model_file, num_rollouts=100, num_threads=10, num_moves=3, name="MCTS", num_processes=1):
    """Benchmark a specific MCTS implementation."""
    
    print(f"\n{'='*80}")
    print(f"{name} PERFORMANCE")
    print(f"{'='*80}")
    
    # Load model
    models, devices = load_model_multi_gpu(model_file, None)
    alphaZeroNet = models[0]
    device = devices[0]
    
    print(f"Model on device: {device}")
    
    # Create board
    board = chess.Board()
    
    total_time = 0
    total_nodes = 0
    move_times = []
    
    # Simulate several moves
    for move_num in range(num_moves):
        if board.is_game_over():
            break
            
        print(f"\nMove {move_num + 1}...")
        
        # Force garbage collection before timing
        gc.collect()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if mcts_module == MCTS_multiprocess and num_processes > 1:
                # Use multiprocessing version
                root = MCTS_multiprocess.create_multiprocess_root(board, alphaZeroNet, model_file)
                root.multiprocess_rollouts(board, model_file, num_rollouts, num_processes, num_threads)
            else:
                # Use regular version
                root = mcts_module.Root(board, alphaZeroNet)
                
                for i in range(num_rollouts):
                    root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        move_times.append(elapsed)
        
        nodes = root.getN()
        nps = nodes / elapsed
        
        print(f"  Rollouts: {int(nodes)}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  NPS: {nps:.1f}")
        
        total_time += elapsed
        total_nodes += nodes
        
        # Make the best move
        edge = root.maxNSelect()
        if edge:
            board.push(edge.getMove())
        
        # Cleanup if available
        if hasattr(root, 'cleanup'):
            root.cleanup()
    
    # Calculate statistics
    avg_time = sum(move_times) / len(move_times)
    min_time = min(move_times)
    max_time = max(move_times)
    
    return {
        'total_nodes': total_nodes,
        'total_time': total_time,
        'avg_nps': total_nodes / total_time,
        'move_times': move_times,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time
    }

def main():
    parser = argparse.ArgumentParser(description='Compare all MCTS implementations')
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of rollouts per move')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads')
    parser.add_argument('--moves', type=int, default=3, help='Number of moves to simulate')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for multiprocessing')
    parser.add_argument('--skip-multiprocess', action='store_true', help='Skip multiprocessing benchmark')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = {}
    
    # Original MCTS
    results['original'] = benchmark_mcts(
        MCTS, args.model, args.rollouts, args.threads, args.moves,
        "ORIGINAL MCTS"
    )
    
    # Clear memory
    gc.collect()
    time.sleep(1)
    
    # Optimized MCTS
    MCTS_opt.clear_caches()
    MCTS_opt.clear_pools()
    results['optimized'] = benchmark_mcts(
        MCTS_opt, args.model, args.rollouts, args.threads, args.moves,
        "OPTIMIZED MCTS"
    )
    
    # Clear memory
    gc.collect()
    time.sleep(1)
    
    # Advanced MCTS
    MCTS_adv.clear_caches()
    MCTS_adv.clear_pools()
    results['advanced'] = benchmark_mcts(
        MCTS_adv, args.model, args.rollouts, args.threads, args.moves,
        "ADVANCED MCTS"
    )
    
    # Clear memory
    gc.collect()
    time.sleep(1)
    
    # Multiprocess MCTS
    if args.processes > 1 and not getattr(args, 'skip_multiprocess', False):
        results['multiprocess'] = benchmark_mcts(
            MCTS_multiprocess, args.model, args.rollouts, args.threads, args.moves,
            f"MULTIPROCESS MCTS ({args.processes} processes)",
            num_processes=args.processes
        )
        print(f"\nNote: Multiprocessing performance varies by system.")
        print(f"      Best for high rollout counts (1000+) on systems with many CPU cores.")
    
    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Implementation':<30} {'Avg NPS':>10} {'Total Time':>12} {'Avg Move Time':>15} {'Min Move Time':>15} {'Max Move Time':>15}")
    print("-"*105)
    
    implementations = [('original', 'Original'), ('optimized', 'Optimized'), ('advanced', 'Advanced')]
    if 'multiprocess' in results:
        implementations.append(('multiprocess', f'Multiprocess ({args.processes} procs)'))
    
    for name, label in implementations:
        r = results[name]
        print(f"{label:<30} {r['avg_nps']:>10.1f} {r['total_time']:>12.3f}s {r['avg_time']:>15.3f}s {r['min_time']:>15.3f}s {r['max_time']:>15.3f}s")
    
    # Calculate improvements
    print(f"\nImprovements over original:")
    print(f"  Optimized:    {results['optimized']['avg_nps']/results['original']['avg_nps']:.2f}x speedup ({(results['optimized']['avg_nps']/results['original']['avg_nps'] - 1)*100:.1f}% faster)")
    print(f"  Advanced:     {results['advanced']['avg_nps']/results['original']['avg_nps']:.2f}x speedup ({(results['advanced']['avg_nps']/results['original']['avg_nps'] - 1)*100:.1f}% faster)")
    if 'multiprocess' in results:
        print(f"  Multiprocess: {results['multiprocess']['avg_nps']/results['original']['avg_nps']:.2f}x speedup ({(results['multiprocess']['avg_nps']/results['original']['avg_nps'] - 1)*100:.1f}% faster)")
    
    # Cache statistics for optimized versions
    if hasattr(MCTS_opt, 'position_cache'):
        print(f"\nOptimized MCTS Cache Statistics:")
        print(f"  Position cache size: {len(MCTS_opt.position_cache)}")
        print(f"  Legal moves cache size: {len(MCTS_opt.legal_moves_cache)}")
        print(f"  Node pool size: {len(MCTS_opt.node_pool)}")
        print(f"  Edge pool size: {len(MCTS_opt.edge_pool)}")
    
    if hasattr(MCTS_adv, 'get_cache_stats'):
        print(f"\nAdvanced MCTS Cache Statistics:")
        stats = MCTS_adv.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    main()