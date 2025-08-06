#!/usr/bin/env python3
"""
Simple test for root parallel MCTS to debug issues.
"""

import chess
import torch
import AlphaZeroNetwork
import MCTS_root_parallel as MCTS
import time

def test_simple():
    print("Loading model...")
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    model.load_state_dict(torch.load('weights/AlphaZeroNet_20x256.pt', map_location='cpu'))
    model.eval()
    
    print("Creating board...")
    board = chess.Board()
    
    print("Creating root node...")
    root = MCTS.Root(board, model, epsilon=0.0)
    
    print("Running rollouts...")
    start = time.time()
    root.parallelRollouts(board, model, 10)  # Just 10 rollouts
    elapsed = time.time() - start
    
    print(f"Rollouts completed in {elapsed:.2f}s")
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print(f"Total visits: {root.getN()}")
    print(f"Root Q: {root.getQ():.4f}")
    
    # Clean up
    MCTS.Root.cleanup_engine()
    print("Test completed!")

if __name__ == "__main__":
    test_simple()