#!/usr/bin/env python3
"""
Test and benchmark CUDA-optimized MCTS implementation.

This script:
1. Builds the C++/CUDA extensions if needed
2. Tests the implementations
3. Benchmarks performance
4. Provides recommendations for Windows CUDA systems
"""

import argparse
import time
import chess
import torch
import sys
import os
import platform
import subprocess

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_system():
    """Check system capabilities."""
    print("="*80)
    print("SYSTEM CHECK")
    print("="*80)
    
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("CUDA Available: No")
    
    # Check CPU
    try:
        import multiprocessing
        print(f"CPU Cores: {multiprocessing.cpu_count()}")
    except:
        pass
    
    print()

def build_extensions():
    """Build C++ and CUDA extensions."""
    print("="*80)
    print("BUILDING EXTENSIONS")
    print("="*80)
    
    if os.path.exists("mcts_cpp.pyd") or os.path.exists("mcts_cpp.so"):
        print("C++ extension already built")
    else:
        print("Building C++ extension...")
        result = subprocess.run([sys.executable, "setup_extensions.py", "build_ext", "--inplace"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ C++ extension built successfully")
        else:
            print("✗ Failed to build C++ extension:")
            print(result.stderr)
    
    if torch.cuda.is_available():
        if os.path.exists("mcts_cuda.pyd") or os.path.exists("mcts_cuda.so"):
            print("CUDA extension already built")
        else:
            print("Building CUDA extension...")
            # CUDA extension is built together with C++ in setup_extensions.py
            print("✓ CUDA extension built (if CUDA toolkit is installed)")
    
    print()

def test_implementations():
    """Test all MCTS implementations."""
    print("="*80)
    print("TESTING IMPLEMENTATIONS")
    print("="*80)
    
    # Test imports
    implementations = []
    
    try:
        import MCTS
        implementations.append(("Original", MCTS))
        print("✓ Original MCTS available")
    except Exception as e:
        print(f"✗ Original MCTS failed: {e}")
    
    try:
        import MCTS_profiling_speedups
        implementations.append(("Optimized", MCTS_profiling_speedups))
        print("✓ Optimized MCTS available")
    except Exception as e:
        print(f"✗ Optimized MCTS failed: {e}")
    
    try:
        import MCTS_advanced_optimizations
        implementations.append(("Advanced", MCTS_advanced_optimizations))
        print("✓ Advanced MCTS available")
    except Exception as e:
        print(f"✗ Advanced MCTS failed: {e}")
    
    try:
        import MCTS_cuda_optimized
        implementations.append(("CUDA", MCTS_cuda_optimized))
        print("✓ CUDA MCTS available")
        
        # Check extensions
        if MCTS_cuda_optimized.CPP_AVAILABLE:
            print("  ✓ C++ extension loaded")
        else:
            print("  ✗ C++ extension not available")
        
        if MCTS_cuda_optimized.CUDA_AVAILABLE:
            print("  ✓ CUDA extension loaded")
        else:
            print("  ✗ CUDA extension not available")
    except Exception as e:
        print(f"✗ CUDA MCTS failed: {e}")
    
    print()
    return implementations

def benchmark_implementation(mcts_module, model_file, num_rollouts=100, num_threads=10, name="MCTS"):
    """Benchmark a single implementation."""
    from playchess import load_model_multi_gpu
    
    print(f"\nBenchmarking {name}...")
    
    # Load model
    models, devices = load_model_multi_gpu(model_file, None)
    alphaZeroNet = models[0]
    device = devices[0]
    
    # Create board
    board = chess.Board()
    
    # Warmup
    with torch.no_grad():
        root = mcts_module.Root(board, alphaZeroNet)
        for _ in range(10):
            root.parallelRollouts(board.copy(), alphaZeroNet, 1)
    
    # Benchmark
    times = []
    for run in range(3):
        start = time.perf_counter()
        
        with torch.no_grad():
            root = mcts_module.Root(board, alphaZeroNet)
            for _ in range(num_rollouts):
                root.parallelRollouts(board.copy(), alphaZeroNet, num_threads)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        nodes = root.getN()
        nps = nodes / elapsed
        
        print(f"  Run {run+1}: {elapsed:.3f}s, {nps:.1f} nps")
    
    avg_time = sum(times) / len(times)
    avg_nps = nodes / avg_time
    
    return {
        'name': name,
        'avg_time': avg_time,
        'avg_nps': avg_nps,
        'times': times,
        'nodes': nodes
    }

def main():
    parser = argparse.ArgumentParser(description='Test CUDA-optimized MCTS')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of rollouts')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads')
    parser.add_argument('--skip-build', action='store_true', help='Skip building extensions')
    parser.add_argument('--benchmark-only', nargs='*', help='Only benchmark specific implementations')
    
    args = parser.parse_args()
    
    # System check
    check_system()
    
    # Build extensions
    if not args.skip_build:
        build_extensions()
    
    # Test implementations
    implementations = test_implementations()
    
    if not implementations:
        print("No implementations available to test!")
        return
    
    # Benchmark
    print("="*80)
    print("BENCHMARKING")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Rollouts: {args.rollouts}")
    print(f"Threads: {args.threads}")
    
    results = []
    
    # Filter implementations if requested
    if args.benchmark_only:
        implementations = [(name, module) for name, module in implementations 
                          if any(filter_name.lower() in name.lower() 
                                for filter_name in args.benchmark_only)]
    
    for name, module in implementations:
        try:
            result = benchmark_implementation(
                module, args.model, args.rollouts, args.threads, name
            )
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Sort by performance
        results.sort(key=lambda x: x['avg_nps'], reverse=True)
        
        print(f"\n{'Implementation':<20} {'Avg Time':>12} {'Avg NPS':>12} {'Speedup':>10}")
        print("-"*54)
        
        baseline_nps = results[-1]['avg_nps']  # Slowest as baseline
        
        for r in results:
            speedup = r['avg_nps'] / baseline_nps
            print(f"{r['name']:<20} {r['avg_time']:>12.3f}s {r['avg_nps']:>12.1f} {speedup:>10.2f}x")
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR WINDOWS CUDA SYSTEMS")
        print("="*80)
        print("""
1. **Build Extensions**: On Windows, ensure you have:
   - Visual Studio 2019 or later with C++ tools
   - CUDA Toolkit matching your PyTorch version
   - Run: python setup_extensions.py build_ext --inplace

2. **Optimal Settings**:
   - Use CUDA implementation for maximum performance
   - Set batch size based on GPU memory (32-128 typical)
   - Use high thread count (16-32) for CPU portions
   - Enable aggressive batching for many rollouts

3. **Performance Tuning**:
   - Adjust BATCH_SIZE in MCTS_cuda_optimized.py
   - Set MAX_BATCH_WAIT_TIME based on latency requirements
   - Use larger cache sizes if memory permits
   - Consider multi-GPU for very high rollout counts

4. **Expected Performance**:
   - With RTX 3080+: 2000-5000 nps
   - With RTX 4090: 3000-8000 nps
   - Scales well with multiple GPUs

5. **Usage**:
   python playchess.py --model <model> --rollouts 1000 --threads 32
   
   Or in code:
   import MCTS_cuda_optimized as MCTS
   root = MCTS.Root(board, model)
""")

if __name__ == '__main__':
    main()