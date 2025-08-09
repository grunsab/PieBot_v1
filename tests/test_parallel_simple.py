#!/usr/bin/env python3
"""
Simple test script to verify parallel inference servers work on Windows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import time
import multiprocessing as mp

def test_optimized():
    """Test the optimized parallel implementation."""
    print("Testing optimized parallel inference...")
    
    # Import after setting spawn
    import AlphaZeroNetwork
    from MCTS_root_parallel_optimized import Root as RootOptimized
    
    # Load model
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    try:
        model.load_state_dict(torch.load('weights/AlphaZeroNet_10x128.pt', map_location='cpu'))
        print("Loaded 10x128 model")
    except Exception as e:
        print(f"Warning: Could not load weights ({e}), using random initialization")
    
    model.eval()
    
    # Create board
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    
    print(f"\nTest position after: e4 e5")
    print(board)
    
    # Test with parallel servers
    print("\nRunning MCTS with parallel inference servers...")
    start_time = time.perf_counter()
    
    root = RootOptimized(board, model, epsilon=0.0, use_parallel_servers=True)
    root.parallelRollouts(board, model, 100)
    
    elapsed = time.perf_counter() - start_time
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Throughput: {100/elapsed:.0f} rollouts/second")
    
    # Cleanup
    RootOptimized.cleanup_engine()
    
    print("\nTest completed successfully!")
    return True

def test_original():
    """Test the original implementation for comparison."""
    print("Testing original implementation...")
    
    # Import after setting spawn
    import AlphaZeroNetwork
    from MCTS_root_parallel import Root as RootOriginal
    
    # Load model
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    try:
        model.load_state_dict(torch.load('weights/AlphaZeroNet_10x128.pt', map_location='cpu'))
        print("Loaded 10x128 model")
    except:
        print("Warning: Could not load weights, using random initialization")
    
    model.eval()
    
    # Create board
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    
    print(f"\nTest position after: e4 e5")
    print(board)
    
    # Test
    print("\nRunning MCTS with single inference server...")
    start_time = time.perf_counter()
    
    root = RootOriginal(board, model, epsilon=0.0)
    root.parallelRollouts(board, model, 100)
    
    elapsed = time.perf_counter() - start_time
    
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()}")
        print(f"Visits: {best_edge.getN()}")
        print(f"Q-value: {best_edge.getQ():.4f}")
    
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Throughput: {100/elapsed:.0f} rollouts/second")
    
    # Cleanup
    RootOriginal.cleanup_engine()
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("PARALLEL INFERENCE SERVER TEST")
    print("="*60)
    
    # Test based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--original':
        success = test_original()
    else:
        success = test_optimized()
        
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)