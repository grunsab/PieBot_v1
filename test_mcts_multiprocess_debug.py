#!/usr/bin/env python3
"""
Debug test script for the updated MCTS_multiprocess implementation.
"""

import chess
import torch
import time
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import Root

def test_basic_functionality():
    """Test basic MCTS functionality with debug output."""
    print("Testing MCTS_multiprocess implementation...")
    
    # Create a simple position
    board = chess.Board()
    
    # Load normal model for testing
    print("Creating model...")
    model = AlphaZeroNet(num_blocks=20, num_filters=256)
    
    # Check if we have a saved model
    try:
        model.load_state_dict(torch.load('weights/AlphaZeroNet_20x256.pt', map_location='cuda'))
        print("Loaded model weights")
    except:
        print("Using random model weights for testing")
    
    model.eval()
    
    # Test with very small number of rollouts
    print("\n1. Testing with 2 rollouts...")
    root = Root(board, model)
    
    print("Starting parallel rollouts...")
    start_time = time.time()
    
    try:
        root.parallelRollouts(board, model, 2)
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error during rollouts: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get best move
    print("Getting best move...")
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"   Best move: {best_edge.getMove()}")
        print(f"   Visit count: {best_edge.getN()}")
        print(f"   Q value: {best_edge.getQ():.3f}")
    else:
        print("   No best edge found")
    
    # Clean up
    print("\nCleaning up...")
    Root.cleanup_engine()
    print("   Cleanup completed")
    
    print("\n✓ Test completed!")

if __name__ == "__main__":
    test_basic_functionality()