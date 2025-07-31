#!/usr/bin/env python3
"""Debug script to find the exact error location in CUDA MCTS."""

import chess
import torch
import numpy as np
import sys
import os
import traceback

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cuda_mcts():
    """Test CUDA MCTS with detailed error reporting."""
    print("Testing CUDA MCTS implementation...")
    
    # Import modules
    import encoder
    from playchess import load_model_multi_gpu
    
    # Load model
    model_file = "weights/AlphaZeroNet_20x256.pt"
    if not os.path.exists(model_file):
        print(f"Error: Model file not found: {model_file}")
        return
    
    print(f"Loading model from {model_file}...")
    models, devices = load_model_multi_gpu(model_file, None)
    model = models[0]
    device = devices[0]
    print(f"Model loaded on {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Create board
    board = chess.Board()
    print(f"\nBoard has {len(list(board.legal_moves))} legal moves")
    
    # Test encoder
    print("\nTesting encoder...")
    position, mask = encoder.encodePositionForInference(board)
    print(f"Position shape: {position.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test neural network
    print("\nTesting neural network inference...")
    with torch.no_grad():
        pos_tensor = torch.from_numpy(position).unsqueeze(0).to(device, dtype=next(model.parameters()).dtype)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device, dtype=next(model.parameters()).dtype)
        mask_flat = mask_tensor.view(1, -1)
        
        value, policy = model(pos_tensor, policyMask=mask_flat)
        print(f"Value shape: {value.shape}")
        print(f"Policy shape: {policy.shape}")
        
        policy_np = policy[0].cpu().numpy()
        print(f"Policy numpy shape: {policy_np.shape}")
        print(f"Policy numpy size: {policy_np.size}")
        print(f"Policy max index would be: {policy_np.size - 1}")
    
    # Test policy decoding
    print("\nTesting policy decoding...")
    move_probs = encoder.decodePolicyOutput(board, policy_np)
    print(f"Move probabilities shape: {move_probs.shape}")
    print(f"Move probabilities size: {move_probs.size}")
    
    # Now test MCTS
    print("\n\nTesting MCTS initialization...")
    try:
        import MCTS_cuda_optimized as MCTS
        
        # Test Root creation
        print("Creating MCTS Root...")
        root = MCTS.Root(board, model)
        print(f"Root created successfully")
        print(f"Root has {root.num_edges} edges")
        
        # Test rollout
        print("\nTesting rollout...")
        root.rollout(board.copy())
        print("Single rollout successful")
        
        # Test parallel rollouts
        print("\nTesting parallel rollouts...")
        root.parallelRollouts(board.copy(), model, 1)
        print("Parallel rollouts successful")
        
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to get more details
        if hasattr(e, 'args') and len(e.args) > 0:
            print(f"\nError args: {e.args}")

if __name__ == "__main__":
    test_cuda_mcts()