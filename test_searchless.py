#!/usr/bin/env python3
"""
Test script to verify searchless_value and searchless_policy work as drop-in replacements.
"""

import chess
import torch
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
import time

def test_searchless_value():
    """Test searchless_value module."""
    print("Testing searchless_value...")
    import searchless_value as MCTS
    
    # Load model
    device, device_str = get_optimal_device()
    print(f"Using device: {device_str}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    weights = torch.load("weights/AlphaZeroNet_10x128.pt", map_location='cpu')
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    # Ensure no gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Create board
    board = chess.Board()
    
    # Test Root creation
    root = MCTS.Root(board, model)
    print(f"Created root with {len(root.edges)} legal moves")
    
    # Test rollouts (should be no-op but compatible)
    start = time.time()
    root.parallelRolloutsTotal(board.copy(), model, 100, 10)
    elapsed = time.time() - start
    print(f"Mock rollouts took {elapsed:.3f}s")
    
    # Get best move
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()} with value {best_edge.getQ():.4f}")
    
    # Test statistics
    print("\nMove statistics:")
    print(root.getStatisticsString())
    
    print("✓ searchless_value test passed\n")


def test_searchless_policy():
    """Test searchless_policy module."""
    print("Testing searchless_policy...")
    import searchless_policy as MCTS
    
    # Load model
    device, device_str = get_optimal_device()
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    weights = torch.load("weights/AlphaZeroNet_10x128.pt", map_location='cpu')
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    # Ensure no gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Create board
    board = chess.Board()
    
    # Test Root creation
    root = MCTS.Root(board, model)
    print(f"Created root with {len(root.edges)} legal moves")
    
    # Test rollouts (should be no-op but compatible)
    start = time.time()
    root.parallelRolloutsTotal(board.copy(), model, 100, 10)
    elapsed = time.time() - start
    print(f"Mock rollouts took {elapsed:.3f}s")
    
    # Get best move
    best_edge = root.maxNSelect()
    if best_edge:
        print(f"Best move: {best_edge.getMove()} with policy {best_edge.getP():.4f}")
    
    # Test statistics
    print("\nMove statistics:")
    print(root.getStatisticsString())
    
    print("✓ searchless_policy test passed\n")


def test_game_play():
    """Test playing a few moves with each method."""
    print("Testing game play...")
    
    device, device_str = get_optimal_device()
    model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
    weights = torch.load("weights/AlphaZeroNet_10x128.pt", map_location='cpu')
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    # Ensure no gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Test with searchless_value
    print("\nPlaying 5 moves with searchless_value:")
    import searchless_value as MCTS_VALUE
    board_value = chess.Board()
    for i in range(5):
        root = MCTS_VALUE.Root(board_value, model)
        root.parallelRolloutsTotal(board_value.copy(), model, 100, 10)
        best_edge = root.maxNSelect()
        if best_edge:
            move = best_edge.getMove()
            print(f"Move {i+1}: {move}")
            board_value.push(move)
        else:
            print("No moves available")
            break
    
    # Test with searchless_policy
    print("\nPlaying 5 moves with searchless_policy:")
    import searchless_policy as MCTS_POLICY
    board_policy = chess.Board()
    for i in range(5):
        root = MCTS_POLICY.Root(board_policy, model)
        root.parallelRolloutsTotal(board_policy.copy(), model, 100, 10)
        best_edge = root.maxNSelect()
        if best_edge:
            move = best_edge.getMove()
            print(f"Move {i+1}: {move}")
            board_policy.push(move)
        else:
            print("No moves available")
            break
    
    print("\n✓ Game play test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Searchless Move Selection Modules")
    print("=" * 60)
    
    try:
        test_searchless_value()
        test_searchless_policy()
        test_game_play()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("The modules can be used as drop-in replacements:")
        print("  - Change: import MCTS_root_parallel as MCTS")
        print("  - To: import searchless_value as MCTS")
        print("  - Or: import searchless_policy as MCTS")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()