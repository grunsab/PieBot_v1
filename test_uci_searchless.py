#!/usr/bin/env python3
"""
Test UCI engine compatibility with searchless modules.
"""

import sys
import chess

def test_uci_compatibility():
    """Test that searchless modules work with UCI_engine.py"""
    
    print("Testing UCI engine compatibility with searchless modules...")
    print("=" * 60)
    
    # Test importing as drop-in replacement
    print("\n1. Testing import compatibility...")
    
    # Test searchless_value
    try:
        # Temporarily modify sys.modules to test import compatibility
        import searchless_value
        sys.modules['MCTS_root_parallel'] = searchless_value
        print("✓ searchless_value can replace MCTS_root_parallel")
    except Exception as e:
        print(f"✗ searchless_value import failed: {e}")
        return False
    
    # Test searchless_policy
    try:
        import searchless_policy
        sys.modules['MCTS_root_parallel'] = searchless_policy
        print("✓ searchless_policy can replace MCTS_root_parallel")
    except Exception as e:
        print(f"✗ searchless_policy import failed: {e}")
        return False
    
    print("\n2. Testing method compatibility...")
    
    # Check that all required methods exist
    required_methods = ['Root', 'cleanup_engine']
    
    for module_name, module in [('searchless_value', searchless_value), 
                                 ('searchless_policy', searchless_policy)]:
        print(f"\n   Checking {module_name}:")
        for method in required_methods:
            if hasattr(module, method):
                print(f"   ✓ {method} exists")
            else:
                print(f"   ✗ {method} missing")
                return False
        
        # Check Root class methods
        root_methods = ['__init__', 'parallelRolloutsTotal', 'parallelRollouts', 
                       'maxNSelect', 'getN', 'cleanup']
        test_board = chess.Board()
        
        # We can't actually create a Root without a model, so check the class
        root_class = module.Root
        for method in root_methods:
            if hasattr(root_class, method):
                print(f"   ✓ Root.{method} exists")
            else:
                print(f"   ✗ Root.{method} missing")
                return False
    
    print("\n" + "=" * 60)
    print("✓ All compatibility tests passed!")
    print("\nTo use with UCI_engine.py, replace the import:")
    print("  Change: import MCTS_root_parallel as MCTS")
    print("  To:     import searchless_value as MCTS")
    print("  Or:     import searchless_policy as MCTS")
    
    return True


def test_playchess_run():
    """Test running playchess with searchless modules."""
    
    print("\n" + "=" * 60)
    print("Testing playchess.py compatibility...")
    print("=" * 60)
    
    # Create a test script that imports searchless instead of MCTS
    test_script = '''
import chess
import torch
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device

# Test with searchless_value
print("Testing with searchless_value...")
import searchless_value as MCTS

device, _ = get_optimal_device()
model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
weights = torch.load("weights/AlphaZeroNet_10x128.pt", map_location='cpu')
if isinstance(weights, dict) and 'model_state_dict' in weights:
    model.load_state_dict(weights['model_state_dict'])
else:
    model.load_state_dict(weights)
model = optimize_for_device(model, device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

board = chess.Board()
root = MCTS.Root(board, model)
root.parallelRolloutsTotal(board.copy(), model, 100, 10)
edge = root.maxNSelect()
if edge:
    print(f"  Selected move: {edge.getMove()}")
    print("  ✓ searchless_value works with playchess structure")

# Test with searchless_policy
print("\\nTesting with searchless_policy...")
import searchless_policy as MCTS

root = MCTS.Root(board, model)
root.parallelRolloutsTotal(board.copy(), model, 100, 10)
edge = root.maxNSelect()
if edge:
    print(f"  Selected move: {edge.getMove()}")
    print("  ✓ searchless_policy works with playchess structure")
'''
    
    # Write and run the test script
    with open('_test_playchess_compat.py', 'w') as f:
        f.write(test_script)
    
    import subprocess
    result = subprocess.run([sys.executable, '_test_playchess_compat.py'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Clean up
    import os
    os.remove('_test_playchess_compat.py')
    
    return result.returncode == 0


if __name__ == "__main__":
    success = test_uci_compatibility()
    
    if success:
        success = test_playchess_run()
    
    if success:
        print("\n" + "=" * 60)
        print("All compatibility tests passed!")
        print("The searchless modules are ready to use as drop-in replacements.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some tests failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)