#!/usr/bin/env python3
"""Simple test for MCTS cuda optimized."""

import chess
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    import MCTS_cuda_optimized as MCTS
    from playchess import load_model_multi_gpu
    
    # Load model
    model_file = "weights/AlphaZeroNet_20x256.pt"
    if not os.path.exists(model_file):
        model_file = "AlphaZeroNet_20x256_distributed.pt"
    
    print(f"Loading model: {model_file}")
    models, devices = load_model_multi_gpu(model_file, None)
    model = models[0]
    print("Model loaded successfully")
    
    # Create board
    board = chess.Board()
    
    # Create root
    print("\nCreating Root...")
    root = MCTS.Root(board, model)
    print(f"Root created with {root.num_edges} edges")
    
    # Try parallel rollouts directly
    print("\nTesting parallelRollouts...")
    try:
        root.parallelRollouts(board.copy(), model, 1)
        print("✓ 1 rollout successful")
        
        root.parallelRollouts(board.copy(), model, 10)
        print("✓ 10 rollouts successful")
        
        root.parallelRollouts(board.copy(), model, 100)
        print("✓ 100 rollouts successful")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()