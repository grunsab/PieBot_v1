#!/usr/bin/env python3
"""
Minimal test for parallel inference - uses CPU only to avoid memory issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to avoid DLL loading issues

import chess
import torch
import time
import multiprocessing as mp

def test_simple():
    """Simple test with minimal resources."""
    print("Testing optimized parallel inference (CPU only)...")
    
    # Force CPU
    torch.set_default_device('cpu')
    
    # Import after setting device
    import AlphaZeroNetwork
    from MCTS_root_parallel_optimized import OptimizedRootParallelMCTS
    
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
    
    # Test with minimal configuration
    print("\nRunning MCTS with parallel inference servers (CPU)...")
    print("Using only 2 workers and 2 inference servers to minimize memory usage")
    
    # Create engine with minimal resources
    engine = OptimizedRootParallelMCTS(
        model, 
        num_workers=2,  # Only 2 MCTS workers
        epsilon=0.0,
        use_parallel_servers=True,
        num_inference_servers=2,  # Only 2 inference servers
        inference_batch_size=32,
        inference_timeout_ms=100
    )
    
    start_time = time.perf_counter()
    
    # Run search
    stats = engine.run_parallel_search(board, 50)  # Only 50 rollouts for quick test
    
    elapsed = time.perf_counter() - start_time
    
    # Get best move
    best_move = engine.get_best_move(stats)
    if best_move and best_move in stats:
        move_stats = stats[best_move]
        print(f"\nBest move: {best_move}")
        print(f"Visits: {move_stats['visits']}")
        print(f"Q-value: {move_stats['q_value']:.4f}")
    
    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Throughput: {50/elapsed:.0f} rollouts/second")
    
    # Cleanup
    engine.cleanup()
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("MINIMAL PARALLEL INFERENCE TEST (CPU ONLY)")
    print("="*60)
    
    try:
        success = test_simple()
        if success:
            print("\n✓ Test passed!")
        else:
            print("\n✗ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)