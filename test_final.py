#!/usr/bin/env python3
"""
Final test to verify playchess.py works
"""

import chess
import torch
import AlphaZeroNetwork
from MCTS_multiprocess import MCTS
import time

def play_game():
    # Load the model
    print("Loading model...")
    model_file = "weights/AlphaZeroNet_10x128.pt"
    weights = torch.load(model_file, map_location='cpu')
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    model.load_state_dict(weights)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Create board and engine
    board = chess.Board()
    mcts_engine = MCTS(model=model)
    
    # Play 10 moves
    print("\nPlaying 10 moves...")
    for i in range(10):
        print(f"\n--- Move {i+1} ---")
        print(board)
        
        # Search for best move
        start = time.time()
        best_move = mcts_engine.search(board, num_simulations=20)
        elapsed = time.time() - start
        
        print(f"Best move: {best_move}")
        print(f"Time: {elapsed:.2f}s, NPS: {20/elapsed:.1f}")
        
        # Make the move
        board.push(best_move)
        
        # Check if game is over
        if board.is_game_over():
            print("\nGame Over!")
            print(f"Result: {board.result()}")
            break
    
    print("\n=== SUCCESS ===")
    print("playchess.py components are working correctly!")
    return True

if __name__ == "__main__":
    try:
        play_game()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()