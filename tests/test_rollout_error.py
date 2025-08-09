#!/usr/bin/env python3
"""Test rollout to find where the index error occurs."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global flag to track detailed debugging
DEBUG = True

# Store original methods
original_getitem = None
original_decode = None

def debug_array_access(self, index):
    """Debug wrapper for array access."""
    if DEBUG:
        print(f"\n[ARRAY ACCESS] shape={self.shape}, index={index}, type(index)={type(index)}")
        if isinstance(index, (int, np.integer)) and index >= self.shape[0]:
            print(f"[ERROR] Index {index} out of bounds for array of size {self.shape[0]}")
            traceback.print_stack()
    return original_getitem(self, index)

def debug_decode(board, policy):
    """Debug wrapper for decodePolicyOutput."""
    if DEBUG:
        print(f"\n[DECODE] board has {len(list(board.legal_moves))} moves, policy shape={policy.shape}")
    try:
        result = original_decode(board, policy)
        if DEBUG:
            print(f"[DECODE] returned shape={result.shape}")
        return result
    except Exception as e:
        print(f"[DECODE ERROR] {e}")
        print(f"[DECODE] policy type: {type(policy)}")
        print(f"[DECODE] policy shape: {getattr(policy, 'shape', 'N/A')}")
        raise

def test_rollouts():
    import MCTS_cuda_optimized
    import encoder
    from playchess import load_model_multi_gpu
    
    # Patch numpy array access
    global original_getitem, original_decode
    original_getitem = np.ndarray.__getitem__
    original_decode = encoder.decodePolicyOutput
    
    # Don't patch yet, let's see where the error happens first
    
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
        
        # Create root
        print("\n=== Creating Root ===")
        root = MCTS_cuda_optimized.Root(board, model)
        print(f"Root created successfully")
        
        # Try a single rollout
        print("\n=== Testing Single Rollout ===")
        try:
            root.rollout(board.copy())
            print("Single rollout completed successfully")
        except Exception as e:
            print(f"Error in single rollout: {e}")
            traceback.print_exc()
            
            # Now enable detailed debugging
            np.ndarray.__getitem__ = debug_array_access
            encoder.decodePolicyOutput = debug_decode
            
            print("\n=== Retrying with debug enabled ===")
            root.rollout(board.copy())
        
        # Try parallel rollouts
        print("\n=== Testing Parallel Rollouts ===")
        try:
            root.parallelRollouts(board.copy(), model, 10)
            print("Parallel rollouts completed successfully")
        except Exception as e:
            print(f"Error in parallel rollouts: {e}")
            traceback.print_exc()
            
            # Enable debugging if not already
            if np.ndarray.__getitem__ != debug_array_access:
                np.ndarray.__getitem__ = debug_array_access
                encoder.decodePolicyOutput = debug_decode
            
            print("\n=== Retrying parallel with debug enabled ===")
            root.parallelRollouts(board.copy(), model, 10)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        # Restore original methods
        if original_getitem:
            np.ndarray.__getitem__ = original_getitem
        if original_decode:
            encoder.decodePolicyOutput = original_decode

if __name__ == "__main__":
    test_rollouts()