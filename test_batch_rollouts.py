"""
Test script for batched rollout implementation in MCTS_root_parallel.py
"""

import chess
import torch
import time
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_root_parallel import RootParallelMCTS

def test_batched_rollouts():
    """Test the new batched rollout implementation."""
    print("Testing batched rollout implementation...")
    
    # Create a simple test model
    model = AlphaZeroNet(num_blocks=10, num_filters=128)
    model.eval()
    
    # Initialize board
    board = chess.Board()
    
    # Test with different batch sizes
    batch_sizes = [32, 64, 128]
    num_rollouts = 256
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size={batch_size}, num_rollouts={num_rollouts}")
        
        # Create MCTS engine with batching
        mcts_engine = RootParallelMCTS(
            model, 
            num_workers=4,
            epsilon=0.25,
            alpha=0.3,
            inference_batch_size=batch_size,
            inference_timeout_ms=5
        )
        
        try:
            # Run search
            start_time = time.time()
            stats = mcts_engine.run_parallel_search(board, num_rollouts)
            end_time = time.time()
            
            # Print results
            print(f"  Time taken: {end_time - start_time:.2f} seconds")
            print(f"  Moves evaluated: {len(stats)}")
            
            # Show top 5 moves
            sorted_moves = sorted(
                stats.items(), 
                key=lambda x: x[1]['visits'], 
                reverse=True
            )[:5]
            
            print("  Top 5 moves:")
            for move, move_stats in sorted_moves:
                print(f"    {move}: visits={move_stats['visits']}, Q={move_stats['q_value']:.4f}")
            
        finally:
            # Clean up
            mcts_engine.cleanup()
            
    print("\nBatched rollout test completed successfully!")

if __name__ == "__main__":
    test_batched_rollouts()