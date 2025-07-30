#!/usr/bin/env python3
"""
Helper script to configure and use CUDA-optimized MCTS in existing code.

This script shows how to use the CUDA-optimized MCTS as a drop-in replacement
in your existing code without modifying the original files.
"""

import sys
import os

def setup_cuda_mcts():
    """
    Configure Python to use CUDA-optimized MCTS by default.
    Call this at the beginning of your script.
    """
    # Try to import and check if CUDA MCTS is available
    try:
        import MCTS_cuda_optimized
        
        # Check if extensions are available
        cpp_available = MCTS_cuda_optimized.CPP_AVAILABLE
        cuda_available = MCTS_cuda_optimized.CUDA_AVAILABLE
        
        print("CUDA MCTS Status:")
        print(f"  C++ Extension: {'✓ Available' if cpp_available else '✗ Not available'}")
        print(f"  CUDA Extension: {'✓ Available' if cuda_available else '✗ Not available'}")
        
        if not cpp_available and not cuda_available:
            print("\nNo extensions available. Building...")
            if MCTS_cuda_optimized.build_extensions():
                print("Extensions built successfully!")
            else:
                print("Failed to build extensions. Using Python fallback.")
        
        # Monkey patch the MCTS module
        sys.modules['MCTS'] = MCTS_cuda_optimized
        print("\n✓ CUDA-optimized MCTS is now the default MCTS implementation")
        
        return True
        
    except ImportError as e:
        print(f"Failed to import CUDA MCTS: {e}")
        print("Using original MCTS implementation")
        return False

def example_usage():
    """Show example usage in existing code."""
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    
    print("""
1. In your existing code, add at the beginning:
   
   from use_cuda_mcts import setup_cuda_mcts
   setup_cuda_mcts()
   
   # Now any import of MCTS will use the CUDA version
   import MCTS  # This will be MCTS_cuda_optimized
   
2. Or use the CUDA versions directly:
   
   # For playchess
   python playchess_cuda.py --model weights/model.pt --rollouts 1000 --threads 16
   
   # For UCI engine
   python UCI_engine_cuda.py --model weights/model.pt --threads 32
   
3. To use in a Python script:
   
   import MCTS_cuda_optimized as MCTS
   
   # Use exactly like regular MCTS
   root = MCTS.Root(board, neural_network)
   root.parallelRollouts(board.copy(), neural_network, num_threads)
   
4. Configuration (edit MCTS_cuda_optimized.py):
   
   BATCH_SIZE = 256  # Increase for better GPU utilization
   MAX_BATCH_WAIT_TIME = 0.001  # Decrease for lower latency
   USE_GPU_TREE = True  # Use GPU for tree operations
""")

def check_performance():
    """Quick performance check."""
    print("\n" + "="*60)
    print("PERFORMANCE CHECK")
    print("="*60)
    
    try:
        import torch
        import chess
        import time
        
        # Setup CUDA MCTS
        if not setup_cuda_mcts():
            return
        
        import MCTS
        
        print("\nRunning quick benchmark...")
        
        # Create a simple test
        board = chess.Board()
        
        # Create dummy model
        class DummyModel(torch.nn.Module):
            def forward(self, x, policyMask=None):
                batch_size = x.shape[0]
                value = torch.zeros((batch_size, 1))
                policy = torch.zeros((batch_size, 4608))
                return value, policy
        
        model = DummyModel()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Time the creation and a few rollouts
        start = time.time()
        root = MCTS.Root(board, model)
        for _ in range(10):
            root.parallelRollouts(board.copy(), model, 1)
        elapsed = time.time() - start
        
        nodes = root.getN()
        nps = nodes / elapsed
        
        print(f"\nResults:")
        print(f"  Nodes: {nodes}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  NPS: {nps:.1f}")
        
        if hasattr(MCTS, 'CUDA_AVAILABLE') and MCTS.CUDA_AVAILABLE:
            print(f"\n✓ CUDA acceleration is working!")
        else:
            print(f"\n⚠ CUDA not available, using CPU optimizations only")
            
    except Exception as e:
        print(f"Performance check failed: {e}")

def main():
    """Main function to show usage and check setup."""
    print("CUDA MCTS Setup Utility")
    print("="*60)
    
    # Check system
    import platform
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA: Not available")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Show usage
    example_usage()
    
    # Optionally run performance check
    response = input("\nRun performance check? (y/n): ")
    if response.lower() == 'y':
        check_performance()

if __name__ == "__main__":
    main()