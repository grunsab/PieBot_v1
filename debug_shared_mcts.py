#!/usr/bin/env python3
"""
Debug script for shared tree MCTS implementation.
"""

import chess
import sys
import os
import torch
import traceback
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AlphaZeroNetwork
import MCTS_shared_search as MCTS
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

def test_single_search(model, board, rollouts=10):
    """Test a single search operation."""
    print(f"\nTesting search with {rollouts} rollouts...")
    print(f"Position: {board.fen()}")
    
    mcts_engine = MCTS.Root(board, model)
    
    start_time = time.time()
    try:
        mcts_engine.parallelRolloutsTotal(board, model, rollouts, 4)
        edge = mcts_engine.maxNSelect()
        
        elapsed = time.time() - start_time
        
        if edge:
            best_move = edge.getMove()
            print(f"Best move: {best_move}")
            print(f"Search took {elapsed:.2f} seconds")
            return best_move
        else:
            print("No move found")
            return None
            
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("Debug test of shared tree MCTS...")
    
    # Load model
    model, device = load_model()
    print(f"Model loaded on {device}")
    
    # Test 1: Initial position with small rollouts
    board = chess.Board()
    print("\n" + "="*50)
    print("Test 1: Initial position with 5 rollouts")
    move = test_single_search(model, board, rollouts=5)
    
    if move:
        # Test 2: Make a move and search again
        board.push(move)
        print("\n" + "="*50)
        print("Test 2: After first move with 5 rollouts")
        move = test_single_search(model, board, rollouts=5)
    
    # Test 3: More rollouts
    print("\n" + "="*50)
    print("Test 3: Current position with 20 rollouts")
    move = test_single_search(model, board, rollouts=20)
    
    # Cleanup
    MCTS.Root.cleanup_engine()
    print("\nAll tests completed. Cleanup done.")

if __name__ == "__main__":
    main()