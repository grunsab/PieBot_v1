#!/usr/bin/env python3
"""Test script to verify titan_piece_value_monitor works with both encoders."""

import torch
import sys
import os

# Add parent directory to path to import modules from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan_piece_value_monitor import TitanPieceValueMonitor
import chess

# Create a dummy model for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, policyMask=None):
        # Return dummy values
        batch_size = x.shape[0]
        value = torch.randn(batch_size, 1)
        policy = torch.randn(batch_size, 73 * 8 * 8)  # 73 move types * 64 squares
        return value, policy
    
    def eval(self):
        return self

def test_regular_encoder():
    """Test with regular encoder."""
    print("Testing with regular encoder...")
    model = DummyModel()
    monitor = TitanPieceValueMonitor(model, device='cpu', enhanced_encoder=False)
    
    # Test creating positions
    positions = monitor.create_comprehensive_test_positions()
    print(f"Created {len(positions)} test positions")
    
    # Test a single position pair
    board_with = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board_without = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1")
    
    # This should not raise an error
    try:
        metrics = monitor.get_convergence_metrics()
        print("Regular encoder test passed!")
        return True
    except Exception as e:
        print(f"Regular encoder test failed: {e}")
        return False

def test_enhanced_encoder():
    """Test with enhanced encoder if available."""
    try:
        import encoder_enhanced
        print("Testing with enhanced encoder...")
        model = DummyModel()
        monitor = TitanPieceValueMonitor(model, device='cpu', enhanced_encoder=True)
        
        # Test creating positions
        positions = monitor.create_comprehensive_test_positions()
        print(f"Created {len(positions)} test positions")
        
        # This should not raise an error
        try:
            metrics = monitor.get_convergence_metrics()
            print("Enhanced encoder test passed!")
            return True
        except Exception as e:
            print(f"Enhanced encoder test failed: {e}")
            return False
    except ImportError:
        print("Enhanced encoder not available, skipping test")
        return True

def main():
    print("="*60)
    print("Testing Titan Piece Value Monitor")
    print("="*60)
    
    # Test both encoders
    regular_ok = test_regular_encoder()
    print()
    enhanced_ok = test_enhanced_encoder()
    
    print("\n" + "="*60)
    if regular_ok and enhanced_ok:
        print("All tests passed!")
    else:
        print("Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()