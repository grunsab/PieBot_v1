#!/usr/bin/env python3
"""Debug script to find the exact location of the index error."""

import chess
import torch
import numpy as np
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_policy_decoding():
    """Test the policy decoding with actual values."""
    import encoder
    
    board = chess.Board()
    print(f"Board has {len(list(board.legal_moves))} legal moves")
    
    # Create a dummy policy output
    policy = np.random.rand(4608)
    
    # Try to decode it
    try:
        move_probs = encoder.decodePolicyOutput(board, policy)
        print(f"Successfully decoded policy: shape {move_probs.shape}")
    except Exception as e:
        print(f"Error in decodePolicyOutput: {e}")
        traceback.print_exc()

def test_cuda_node_creation():
    """Test CudaNode creation with move probabilities."""
    import MCTS_cuda_optimized
    import encoder
    from playchess import load_model_multi_gpu
    
    # Load model
    model_file = "weights/AlphaZeroNet_20x256.pt"
    if not os.path.exists(model_file):
        model_file = "AlphaZeroNet_20x256_distributed.pt"
    
    models, devices = load_model_multi_gpu(model_file, None)
    model = models[0]
    
    board = chess.Board()
    print(f"\nTesting with board having {len(list(board.legal_moves))} legal moves")
    
    # Get neural network output
    position, mask = encoder.encodePositionForInference(board)
    pos_tensor = torch.from_numpy(position).unsqueeze(0).to(devices[0], dtype=next(model.parameters()).dtype)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(devices[0], dtype=next(model.parameters()).dtype)
    mask_flat = mask_tensor.view(1, -1)
    
    with torch.no_grad():
        value, policy = model(pos_tensor, policyMask=mask_flat)
        policy_np = policy[0].cpu().numpy()
    
    print(f"Policy shape from NN: {policy_np.shape}")
    
    # Decode policy
    move_probs = encoder.decodePolicyOutput(board, policy_np)
    print(f"Move probabilities shape: {move_probs.shape}")
    print(f"Move probabilities type: {type(move_probs)}")
    print(f"First few values: {move_probs[:5]}")
    
    # Try creating a CudaNode
    try:
        print("\nCreating CudaNode...")
        node = MCTS_cuda_optimized.CudaNode(board, 0.5, move_probs)
        print(f"CudaNode created successfully with {node.num_edges} edges")
    except Exception as e:
        print(f"Error creating CudaNode: {e}")
        traceback.print_exc()
        
        # Try to find the exact line
        import linecache
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        filename = tb.tb_frame.f_code.co_filename
        lineno = tb.tb_lineno
        line = linecache.getline(filename, lineno).strip()
        print(f"\nError occurred at {filename}:{lineno}")
        print(f"Line: {line}")

def test_indexing_issue():
    """Test the specific indexing that might be causing the error."""
    # Simulate the scenario
    move_probabilities = np.random.rand(20)  # 20 legal moves
    print(f"\nTesting indexing with move_probabilities shape: {move_probabilities.shape}")
    
    # This should work
    try:
        for i in range(20):
            val = move_probabilities[i]
        print("Direct indexing works fine")
    except Exception as e:
        print(f"Direct indexing failed: {e}")
    
    # Test what happens if we accidentally pass the wrong thing
    wrong_probs = 4102  # This might be a move index instead of probabilities
    try:
        val = move_probabilities[wrong_probs]
    except IndexError as e:
        print(f"\nThis reproduces our error: {e}")
        print("It seems like a move index (4102) is being used instead of move probabilities array")

if __name__ == "__main__":
    print("=== Testing Policy Decoding ===")
    test_policy_decoding()
    
    print("\n=== Testing CudaNode Creation ===")
    test_cuda_node_creation()
    
    print("\n=== Testing Indexing Issue ===")
    test_indexing_issue()