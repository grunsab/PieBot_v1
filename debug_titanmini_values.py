#!/usr/bin/env python3
"""Debug TitanMini's value outputs and MCTS Q conversion."""

import torch
import chess
import numpy as np
import TitanMiniNetwork
import encoder
import os

def test_value_range():
    """Test TitanMini's value output range after fix."""
    
    # Try to load weights and determine model configuration
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
        
        # Detect model configuration from weights
        # Check if this uses WDL (3 outputs) or single value (1 output)
        use_wdl = False
        has_legacy_value_head = False
        
        if 'value_head.value_proj.6.weight' in weights:
            # Has layer 6 - this is WDL model
            use_wdl = True
            print("✓ Detected WDL (Win-Draw-Loss) value head")
        elif 'value_head.value_proj.3.weight' in weights:
            shape = weights['value_head.value_proj.3.weight'].shape
            if shape[0] == 1:  # Old 2-layer model - last layer outputs 1 value
                has_legacy_value_head = True
                use_wdl = False
                print("✓ Detected legacy 2-layer value head model (non-WDL)")
            elif shape[0] == 3:  # WDL model with 3 outputs
                use_wdl = True
                print("✓ Detected WDL model")
            else:  # shape[0] == 128 - new 3-layer non-WDL model
                use_wdl = False
                has_legacy_value_head = False
                print("✓ Detected new 3-layer value head model (non-WDL)")
        
        # Create model with appropriate configuration
        model = TitanMiniNetwork.TitanMini(use_wdl=use_wdl, legacy_value_head=has_legacy_value_head)
        model.load_state_dict(weights)
        print("✓ Weights loaded")
    else:
        print("⚠️ No weights file found, using random weights for testing")
        model = TitanMiniNetwork.TitanMini()
    
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
    print("Testing value outputs WITH FIX applied:")
    print("="*60)
    
    for name, fen in positions:
        board = chess.Board(fen)
        
        # Get raw output from model (before conversion)
        import encoder_enhanced
        try:
            position = encoder_enhanced.encode_enhanced_position(board)
        except:
            position, _ = encoder.encodePositionForInference(board)
        position_tensor = torch.from_numpy(position)[None, ...]
        _, mask = encoder.encodePositionForInference(board)
        mask_tensor = torch.from_numpy(mask)[None, ...]
        mask_flat = mask_tensor.view(mask_tensor.shape[0], -1)
        
        with torch.no_grad():
            raw_value, _ = model(position_tensor, policy_mask=mask_flat)
            raw_value = raw_value.item()
        
        # Get value through encoder (after conversion fix)
        with torch.no_grad():
            converted_value, policy = encoder.callNeuralNetwork(board, model)
        
        # MCTS Q conversion
        Q = converted_value / 2.0 + 0.5
        
        print(f"\n{name}:")
        print(f"  FEN: {fen}")
        print(f"  Turn: {'White' if board.turn else 'Black'}")
        print(f"  Raw model output: {raw_value:.4f} (direct from NN, should be in [0, 1] due to Sigmoid)")
        print(f"  Converted value: {converted_value:.4f} (after fix, should be in [-1, 1])")
        print(f"  Q value: {Q:.4f} (for MCTS, should be in [0, 1])")
        
        # Check if raw value is in expected [0, 1] range (Sigmoid output)
        if raw_value < 0 or raw_value > 1:
            print(f"  ⚠️ WARNING: Raw value {raw_value} outside [0, 1] range!")
        
        # Check if converted value is in expected [-1, 1] range
        if abs(converted_value) > 1.0:
            print(f"  ⚠️ WARNING: Converted value {converted_value} outside [-1, 1] range!")
        
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
        
        # Detect model configuration
        use_wdl = False
        has_legacy_value_head = False
        
        if 'value_head.value_proj.6.weight' in weights:
            use_wdl = True
        elif 'value_head.value_proj.3.weight' in weights:
            shape = weights['value_head.value_proj.3.weight'].shape
            if shape[0] == 1:
                has_legacy_value_head = True
                use_wdl = False
            elif shape[0] == 3:
                use_wdl = True
            else:
                use_wdl = False
                has_legacy_value_head = False
        
        model = TitanMiniNetwork.TitanMini(use_wdl=use_wdl, legacy_value_head=has_legacy_value_head)
        model.load_state_dict(weights)
    else:
        model = TitanMiniNetwork.TitanMini()
    
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

def test_mcts_integration():
    """Test MCTS integration with fixed TitanMini values."""
    
    print("\n" + "="*60)
    print("Testing MCTS integration with fixed values:")
    print("="*60)
    
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
        
        # Detect model configuration
        use_wdl = False
        has_legacy_value_head = False
        
        if 'value_head.value_proj.6.weight' in weights:
            use_wdl = True
        elif 'value_head.value_proj.3.weight' in weights:
            shape = weights['value_head.value_proj.3.weight'].shape
            if shape[0] == 1:
                has_legacy_value_head = True
                use_wdl = False
            elif shape[0] == 3:
                use_wdl = True
            else:
                use_wdl = False
                has_legacy_value_head = False
        
        model = TitanMiniNetwork.TitanMini(use_wdl=use_wdl, legacy_value_head=has_legacy_value_head)
        model.load_state_dict(weights)
    else:
        model = TitanMiniNetwork.TitanMini()
    
    model.eval()
    
    # Simulate what happens in MCTS rollout
    board = chess.Board()
    
    # Get value from encoder (with fix)
    value, move_probs = encoder.callNeuralNetwork(board, model)
    
    # Simulate MCTS Q conversion (from MCTS.py line ~224)
    new_Q = value / 2.0 + 0.5
    
    print(f"\nSimulating MCTS rollout at starting position:")
    print(f"  NN value output (after fix): {value:.4f}")
    print(f"  MCTS Q value: {new_Q:.4f}")
    print(f"  Expected range check:")
    print(f"    - NN value in [-1, 1]: {'✓' if -1 <= value <= 1 else '✗'}")
    print(f"    - Q value in [0, 1]: {'✓' if 0 <= new_Q <= 1 else '✗'}")
    
    # Test a few moves deep
    board.push_san("e4")
    board.push_san("e5")
    
    value2, _ = encoder.callNeuralNetwork(board, model)
    new_Q2 = value2 / 2.0 + 0.5
    
    print(f"\nAfter 1.e4 e5:")
    print(f"  NN value output (after fix): {value2:.4f}")
    print(f"  MCTS Q value: {new_Q2:.4f}")
    print(f"  Q from opponent's view: {1 - new_Q2:.4f}")
    
    if abs(new_Q - 0.5) < 0.1 and abs(new_Q2 - 0.5) < 0.1:
        print("\n✓ Values look reasonable for balanced positions!")
    else:
        print("\n⚠️ Values might still need calibration")

if __name__ == "__main__":
    print("="*60)
    print("TitanMini Value Range Fix Verification")
    print("="*60)
    
    test_value_range()
    test_value_consistency()
    test_mcts_integration()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("The fix converts TitanMini's [0,1] Sigmoid output to [-1,1] range")
    print("This makes it compatible with MCTS which expects [-1,1] values")
    print("Run 'python playchess.py --model weights/titan_mini_best.pt' to test in practice")