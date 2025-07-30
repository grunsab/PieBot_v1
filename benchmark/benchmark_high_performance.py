#!/usr/bin/env python3
"""
Benchmark for High-Performance MCTS Implementation
"""

import os
import sys
sys.path.append('..')
import time
import chess
import torch
import argparse
import AlphaZeroNetwork
from experiments.MCTS.MCTS_high_performance import HighPerformanceMCTSEngine

def benchmark_mcts(model_path, device_id=0, num_searches=100, rollouts_per_search=1000):
    """Benchmark the high-performance MCTS implementation"""
    
    print("="*60)
    print("High-Performance MCTS Benchmark")
    print("="*60)
    
    # Load model
    device = torch.device(f'cuda:{device_id}')
    print(f"Loading model: {model_path}")
    print(f"Device: {torch.cuda.get_device_name(device_id)}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    
    # Create engine
    engine = HighPerformanceMCTSEngine(
        model, 
        device=device, 
        batch_size=512,
        verbose=True
    )
    
    # Start engine
    print("\nStarting engine...")
    engine.start()
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # e4
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),  # Italian
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 8"),  # Middlegame
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),  # Endgame
    ]
    
    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        engine.search(positions[0], 100)
    
    # Benchmark different configurations
    configs = [
        (100, 512),    # Fast search, large batch
        (1000, 512),   # Medium search
        (5000, 512),   # Deep search
        (10000, 512),  # Very deep search
    ]
    
    results = []
    
    for rollouts, batch_size in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {rollouts} rollouts, batch size {batch_size}")
        print("-"*60)
        
        # Recreate engine with new batch size
        engine.stop()
        engine = HighPerformanceMCTSEngine(model, device=device, batch_size=batch_size, verbose=False)
        engine.start()
        
        total_time = 0
        total_nodes = 0
        
        # Run searches
        for i, pos in enumerate(positions * (num_searches // len(positions))):
            start = time.time()
            move = engine.search(pos, rollouts)
            elapsed = time.time() - start
            
            total_time += elapsed
            total_nodes += rollouts
            
            nps = rollouts / elapsed
            print(f"Position {i+1}: {move} in {elapsed:.3f}s ({nps:,.0f} NPS)")
            
        avg_nps = total_nodes / total_time
        results.append((rollouts, batch_size, avg_nps))
        
        print(f"\nAverage NPS: {avg_nps:,.0f}")
    
    # Stop engine
    engine.stop()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Rollouts':<10} {'Batch':<10} {'NPS':<15}")
    print("-"*35)
    
    for rollouts, batch_size, nps in results:
        print(f"{rollouts:<10} {batch_size:<10} {nps:>14,.0f}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x[2])
    print(f"\nBest configuration: {best_config[0]} rollouts, "
          f"batch size {best_config[1]} = {best_config[2]:,.0f} NPS")
    
    return best_config[2]

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark High-Performance MCTS Implementation'
    )
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--searches', type=int, default=20, 
                        help='Number of searches per configuration')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Run benchmark
    best_nps = benchmark_mcts(
        args.model,
        args.device,
        args.searches
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {best_nps:,.0f} nodes per second")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()