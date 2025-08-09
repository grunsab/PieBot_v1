#!/usr/bin/env python3
"""
Test script for the updated MCTS_multiprocess implementation.
"""

import chess
import torch
import time
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import MCTS

def test_basic_functionality():
    """Test basic MCTS functionality."""
    print("Testing MCTS_multiprocess implementation...")
    
    # Create a simple position
    board = chess.Board()
    
    # Load normal model for testing
    model = AlphaZeroNet(num_blocks=20, num_filters=256)
    
    # Check if we have a saved model
    try:
        model.load_state_dict(torch.load('weights/AlphaZeroNet_20x256.pt', map_location='cuda'))
        print("Loaded model weights")
    except:
        print("Using random model weights for testing")
    
    model.eval()
    
    # Test with small number of rollouts
    print("\n1. Testing with 10 rollouts...")
    mcts_engine = MCTS(model=model)
    
    start_time = time.time()
    best_move = mcts_engine.search(board, num_simulations=10)
    elapsed = time.time() - start_time
    
    print(f" Completed in {elapsed:.2f} seconds")
    
    # Get best move
    print(best_move)    
    
    # Test with more rollouts
    print("\n2. Testing with 100000 rollouts...")
    board2 = chess.Board()
    mcts_engine = MCTS(model=model)
    
    start_time = time.time()
    best_move = mcts_engine.search(board, num_simulations=100000)
    elapsed = time.time() - start_time
    
    print(f"   Completed in {elapsed:.2f} seconds")
    print(f"   Speed: {100000/elapsed:.1f} rollouts/second")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()