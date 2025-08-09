#!/usr/bin/env python3
"""
Quick test to check for blunders in the first few moves.
"""

import chess
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AlphaZeroNetwork
import MCTS_root_parallel as MCTS
from device_utils import get_optimal_device, optimize_for_device

def load_model(model_path="weights/AlphaZeroNet_20x256.pt"):
    """Load the chess model."""
    device, device_str = get_optimal_device()
    print(f"Loading model on {device_str}...")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, device

def run_quick_test(model, num_moves=10, rollouts_per_move=1000):
    """Run a quick test game."""
    board = chess.Board()
    
    print(f"\nTesting {num_moves} moves with {rollouts_per_move} rollouts each...")
    print("="*60)
    
    for move_num in range(num_moves):
        turn = "White" if board.turn else "Black"
        print(f"\nMove {move_num + 1} ({turn}):")
        
        # Create fresh MCTS engine
        mcts_engine = MCTS.Root(board.copy(), model)
        
        start_time = time.time()
        mcts_engine.parallelRolloutsTotal(board.copy(), model, rollouts_per_move, 64)
        elapsed = time.time() - start_time
        
        edge = mcts_engine.maxNSelect()
        if edge:
            best_move = edge.getMove()
            q_value = edge.getQ()
            visits = edge.getN()
            
            print(f"  Move: {best_move}, Q: {q_value:.3f}, Visits: {visits}, Time: {elapsed:.2f}s")
            
            # Make the move
            board.push(best_move)
        else:
            print("  No move found!")
            break
    
    print("\n" + "="*60)
    print("Test completed!")
    print(f"Final position: {board.fen()}")
    
    # Cleanup
    MCTS.Root.cleanup_engine()

def main():
    """Main test function."""
    model, device = load_model()
    print(f"Model loaded successfully on {device}")
    
    try:
        run_quick_test(model, num_moves=10, rollouts_per_move=1000)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        MCTS.Root.cleanup_engine()
        print("\nCleanup completed.")

if __name__ == "__main__":
    main()