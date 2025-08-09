#!/usr/bin/env python3
"""Test MCTS creation step by step."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey patch to add debugging
original_init = None

def debug_init(self, board, new_Q, move_probabilities, node_id=0):
    print(f"\n=== CudaNode.__init__ called ===")
    print(f"board: {type(board)}")
    print(f"new_Q: {new_Q}")
    print(f"move_probabilities type: {type(move_probabilities)}")
    print(f"move_probabilities shape: {getattr(move_probabilities, 'shape', 'N/A')}")
    print(f"move_probabilities size: {getattr(move_probabilities, 'size', 'N/A')}")
    if hasattr(move_probabilities, '__len__'):
        print(f"move_probabilities length: {len(move_probabilities)}")
        if len(move_probabilities) > 0:
            print(f"First element: {move_probabilities[0]}")
    print(f"node_id: {node_id}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    # Call original
    try:
        original_init(self, board, new_Q, move_probabilities, node_id)
        print("CudaNode created successfully!")
    except Exception as e:
        print(f"Error in original init: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    import MCTS_cuda_optimized
    import encoder
    from playchess import load_model_multi_gpu
    
    # Patch the init method
    global original_init
    original_init = MCTS_cuda_optimized.CudaNode.__init__
    MCTS_cuda_optimized.CudaNode.__init__ = debug_init
    
    try:
        # Load model
        model_file = "weights/AlphaZeroNet_20x256.pt"
        if not os.path.exists(model_file):
            model_file = "AlphaZeroNet_20x256_distributed.pt"
        
        print(f"Loading model: {model_file}")
        models, devices = load_model_multi_gpu(model_file, None)
        model = models[0]
        
        # Create board
        board = chess.Board()
        
        # Test Root creation
        print("\n=== Creating MCTS Root ===")
        root = MCTS_cuda_optimized.Root(board, model)
        print(f"Root created with {root.num_edges} edges")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original
        if original_init:
            MCTS_cuda_optimized.CudaNode.__init__ = original_init

if __name__ == "__main__":
    main()