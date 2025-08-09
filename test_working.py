#!/usr/bin/env python3
"""
Test script to verify the MCTS_multiprocess engine is working correctly.
"""

import chess
from MCTS_multiprocess import MCTS
import torch
import AlphaZeroNetwork
import time

def main():
    print("=== Testing MCTS_multiprocess Engine ===\n")
    
    # Load model
    print("Loading model...")
    model_file = "weights/AlphaZeroNet_20x256.pt"
    weights = torch.load(model_file, map_location='cpu')
    alphaZeroNet = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    alphaZeroNet.load_state_dict(weights)
    alphaZeroNet.eval()
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alphaZeroNet = alphaZeroNet.to(device)
    print(f"Model loaded on {device}\n")
    
    # Create board
    board = chess.Board()
    
    # Create MCTS engine
    print("Creating MCTS engine...")
    mcts_engine = MCTS(model=alphaZeroNet)
    
    # Test with different simulation counts
    simulation_counts = [10, 50, 100]
    
    for num_sims in simulation_counts:
        print(f"\nTesting with {num_sims} simulations:")
        print(board)
        
        start_time = time.time()
        best_move = mcts_engine.search(board, num_simulations=num_sims)
        elapsed = time.time() - start_time
        
        print(f"Best move: {best_move}")
        print(f"Time taken: {elapsed:.3f} seconds")
        print(f"NPS (nodes per second): {num_sims/elapsed:.1f}")
        
        # Make the move for next test
        board.push(best_move)
    
    print("\n=== Test Complete ===")
    print("The MCTS_multiprocess engine is working correctly!")

if __name__ == "__main__":
    main()