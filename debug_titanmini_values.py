#!/usr/bin/env python3
"""Debug TitanMini's value outputs and MCTS Q conversion."""

import torch
import chess
import numpy as np
import TitanMiniNetwork
import encoder
import os

def test_value_range():
    """Test TitanMini's value output range."""
    
    # Create and load model
    model = TitanMiniNetwork.TitanMini()
    
    weight_file = "weights/titan_mini_best_weights_only.pt"
    if os.path.exists(weight_file):
        weights = torch.load(weight_file, map_location='cpu', weights_only=False)
        if any(k.startswith('_orig_mod.') for k in weights.keys()):
            new_weights = {}
            for k, v in weights.items():
                if k.startswith('_orig_mod.'):
                    new_weights[k[10:]] = v
                else:
                    new_weights[k] = v
            weights = new_weights
        model.load_state_dict(weights)
        print("✓ Weights loaded")
    
    model.eval()
    
    # Test various positions
    positions = [
        ("Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("White winning", "4k3/8/4K3/8/8/8/8/4R3 w - - 0 1"),
        ("Black winning", "4K3/8/8/8/8/8/8/4kr2 b - - 0 1"),
        ("Equal endgame", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ]
    
    print("\n" + "="*60)
    print("Testing value outputs and Q conversion:")
    print("="*60)
    
    for name, fen in positions:
        board = chess.Board(fen)
        
        # Get raw value from neural network
        with torch.no_grad():
            value, policy = encoder.callNeuralNetwork(board, model)
        
        # MCTS Q conversion
        Q = value / 2.0 + 0.5
        
        print(f"\n{name}:")
        print(f"  FEN: {fen}")
        print(f"  Turn: {'White' if board.turn else 'Black'}")
        print(f"  Raw value: {value:.4f} (from NN, should be in [-1, 1])")
        print(f"  Q value: {Q:.4f} (for MCTS, should be in [0, 1])")
        
        # Check if value is in expected range
        if abs(value) > 1.0:
            print(f"  ⚠️ WARNING: Value {value} outside [-1, 1] range!")
        
        # Check if Q is in expected range
        if Q < 0 or Q > 1:
            print(f"  ⚠️ WARNING: Q value {Q} outside [0, 1] range!")
        
        # Check policy sum
        policy_sum = np.sum(policy)
        print(f"  Policy sum: {policy_sum:.4f} (should be ~1.0)")
        if abs(policy_sum - 1.0) > 0.01:
            print(f"  ⚠️ WARNING: Policy doesn't sum to 1!")

def test_value_consistency():
    """Test if values are consistent (white winning = -black winning from black's view)."""
    
    model = TitanMiniNetwork.TitanMini()
    weight_file = "weights/titan_mini_best_weights_only.pt"
    if os.path.exists(weight_file):
        weights = torch.load(weight_file, map_location='cpu', weights_only=False)
        if any(k.startswith('_orig_mod.') for k in weights.keys()):
            new_weights = {}
            for k, v in weights.items():
                if k.startswith('_orig_mod.'):
                    new_weights[k[10:]] = v
                else:
                    new_weights[k] = v
            weights = new_weights
        model.load_state_dict(weights)
    
    model.eval()
    
    print("\n" + "="*60)
    print("Testing value consistency:")
    print("="*60)
    
    # Test the same position from both perspectives
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    
    with torch.no_grad():
        value_white, _ = encoder.callNeuralNetwork(board, model)
    
    board.turn = chess.BLACK
    with torch.no_grad():
        value_black, _ = encoder.callNeuralNetwork(board, model)
    
    print(f"\nPosition after 1.e4:")
    print(f"  Value from White's perspective: {value_white:.4f}")
    print(f"  Value from Black's perspective: {value_black:.4f}")
    print(f"  Sum: {value_white + value_black:.4f} (should be close to 0)")
    
    if abs(value_white + value_black) > 0.2:
        print(f"  ⚠️ WARNING: Values not consistent! They should sum to ~0")

if __name__ == "__main__":
    test_value_range()
    test_value_consistency()