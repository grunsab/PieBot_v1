#!/usr/bin/env python3
"""
Benchmark script for Ultra-Performance MCTS implementation.
Tests the new leaf batching approach for maximum NPS.
"""

import argparse
import time
import torch
import chess
from MCTS_ultra_performance import UltraPerformanceMCTSEngine
from AlphaZeroNetwork import AlphaZeroNetwork
import device_utils

def benchmark_ultra_mcts(model_path, device, num_searches=20):
    """Benchmark the ultra-performance MCTS implementation"""
    
    # Load model
    print(f"Loading model: {model_path}")
    model = AlphaZeroNetwork()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # e4
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),  # Italian
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 8"),  # Middlegame
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),  # Endgame
    ]
    
    # Test different configurations
    configs = [
        # (rollouts, batch_size, num_workers)
        (100, 256, 8),      # Fast search
        (1000, 512, 16),    # Medium search
        (5000, 512, 16),    # Deep search
        (10000, 512, 32),   # Very deep search
    ]
    
    results = []
    
    for rollouts, batch_size, num_workers in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {rollouts} rollouts, batch {batch_size}, workers {num_workers}")
        print("-"*60)
        
        # Create engine
        engine = UltraPerformanceMCTSEngine(
            model, 
            device=device, 
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=False
        )
        
        # Start engine
        engine.start()
        
        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            engine.search(positions[0], 100)
        
        # Benchmark
        total_time = 0
        total_nodes = 0
        
        print("\nBenchmarking...")
        for i, pos in enumerate(positions * (num_searches // len(positions))):
            start = time.time()
            move = engine.search(pos, rollouts)
            elapsed = time.time() - start
            
            total_time += elapsed
            total_nodes += rollouts
            
            nps = rollouts / elapsed if elapsed > 0 else 0
            print(f"Position {i+1}: {move} in {elapsed:.3f}s ({nps:,.0f} NPS)")
        
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        results.append((rollouts, batch_size, num_workers, avg_nps))
        
        print(f"\nAverage NPS: {avg_nps:,.0f}")
        
        # Stop engine
        engine.stop()
    
    # Summary
    print("\n" + "="*60)
    print("ULTRA-PERFORMANCE MCTS BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Rollouts':<10} {'Batch':<10} {'Workers':<10} {'NPS':<15}")
    print("-"*45)
    
    for rollouts, batch_size, workers, nps in results:
        print(f"{rollouts:<10} {batch_size:<10} {workers:<10} {nps:>14,.0f}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x[3])
    print(f"\nBest configuration: {best_config[0]} rollouts, "
          f"batch {best_config[1]}, workers {best_config[2]} = {best_config[3]:,.0f} NPS")
    
    return best_config[3]

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Ultra-Performance MCTS Implementation'
    )
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--searches', type=int, default=20, 
                        help='Number of searches per configuration')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.device).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Configure device
    device_utils.set_device_options(device)
    
    print("="*60)
    print("Ultra-Performance MCTS Benchmark")
    print("="*60)
    
    best_nps = benchmark_ultra_mcts(
        args.model,
        device,
        args.searches
    )
    
    print(f"\nPeak performance: {best_nps:,.0f} NPS")

if __name__ == "__main__":
    main()