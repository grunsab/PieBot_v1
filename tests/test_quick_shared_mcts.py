#!/usr/bin/env python3
"""
Quick test for shared tree MCTS implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import traceback

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

def main():
    """Main test function."""
    print("Quick test of shared tree MCTS...")
    
    # Load model
    model, device = load_model()
    print(f"Model loaded on {device}")
    
    # Create initial position
    board = chess.Board()
    
    # Create MCTS engine
    print("\nCreating MCTS engine...")
    mcts_engine = MCTS.Root(board, model)
    
    # Run a small search
    print("Running search with 10 rollouts...")
    try:
        mcts_engine.parallelRolloutsTotal(board, model, 10, 4)
        
        # Get best move
        edge = mcts_engine.maxNSelect()
        if edge:
            best_move = edge.getMove()
            print(f"Best move found: {best_move}")
        else:
            print("No move found")
            
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        MCTS.Root.cleanup_engine()
        print("Cleanup completed")

if __name__ == "__main__":
    main()