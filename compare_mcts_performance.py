#!/usr/bin/env python3
"""
Compare performance between original MCTS and optimized version.

Usage:
    python compare_mcts_performance.py --model AlphaZeroNet_20x256_distributed.pt
"""

import argparse
import time
import chess
import torch
import sys
import os

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MCTS
import MCTS_profiling_speedups as MCTS_opt
from playchess import load_model_multi_gpu

def benchmark_mcts(mcts_module, model_file, num_rollouts=100, num_threads=10, num_moves=5):
    """Benchmark a specific MCTS implementation."""
    
    # Load model
    models, devices = load_model_multi_gpu(model_file, None)
    alphaZeroNet = models[0]
    device = devices[0]
    
    print(f"Model on device: {device}")
    
    # Create board
    board = chess.Board()
    
    total_time = 0
    total_nodes = 0
    
    # Simulate several moves
    for move_num in range(num_moves):
        if board.is_game_over():
            break
            
        print(f"\nMove {move_num + 1}...")
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            root = mcts_module.Root(board, alphaZeroNet)
            
            for i in range(num_rollouts):
                root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
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
    
    return total_nodes, total_time

def main():
    parser = argparse.ArgumentParser(description='Compare MCTS performance')
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of rollouts per move')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads')
    parser.add_argument('--moves', type=int, default=5, help='Number of moves to simulate')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ORIGINAL MCTS PERFORMANCE")
    print("="*80)
    
    orig_nodes, orig_time = benchmark_mcts(
        MCTS, 
        args.model, 
        args.rollouts, 
        args.threads, 
        args.moves
    )
    
    orig_nps = orig_nodes / orig_time
    
    print("\n" + "="*80)
    print("OPTIMIZED MCTS PERFORMANCE")
    print("="*80)
    
    # Clear caches before optimized run
    MCTS_opt.clear_caches()
    MCTS_opt.clear_pools()
    
    opt_nodes, opt_time = benchmark_mcts(
        MCTS_opt, 
        args.model, 
        args.rollouts, 
        args.threads, 
        args.moves
    )
    
    opt_nps = opt_nodes / opt_time
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Original MCTS:")
    print(f"  Total nodes: {int(orig_nodes)}")
    print(f"  Total time: {orig_time:.3f}s")
    print(f"  Average NPS: {orig_nps:.1f}")
    
    print(f"\nOptimized MCTS:")
    print(f"  Total nodes: {int(opt_nodes)}")
    print(f"  Total time: {opt_time:.3f}s")
    print(f"  Average NPS: {opt_nps:.1f}")
    
    print(f"\nImprovement:")
    print(f"  Speedup: {opt_nps/orig_nps:.2f}x")
    print(f"  Time saved: {orig_time - opt_time:.3f}s ({(1 - opt_time/orig_time)*100:.1f}%)")
    
    # Print cache statistics if available
    print(f"\nCache Statistics:")
    print(f"  Position cache size: {len(MCTS_opt.position_cache)}")
    print(f"  Legal moves cache size: {len(MCTS_opt.legal_moves_cache)}")
    print(f"  Node pool size: {len(MCTS_opt.node_pool)}")
    print(f"  Edge pool size: {len(MCTS_opt.edge_pool)}")

if __name__ == '__main__':
    main()