#!/usr/bin/env python3
"""
Line-by-line profiling for MCTS functions.

This script requires line_profiler to be installed:
    pip install line_profiler

Usage:
    python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function calcUCT
    python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function UCTSelect
    python profile_line.py --model weights/AlphaZeroNet_20x256.pt --function selectTask
"""

import argparse
import sys
import os
import tempfile
import subprocess

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_script(model_file, function_name, num_rollouts=100):
    """Create a test script that exercises the target function."""
    
    test_code = f"""
import chess
import torch
import MCTS
import AlphaZeroNetwork
from playchess import load_model_multi_gpu

# Load model
models, devices = load_model_multi_gpu("{model_file}", None)
alphaZeroNet = models[0]

# Create board
board = chess.Board()

# Create root
with torch.no_grad():
    root = MCTS.Root(board, alphaZeroNet)
    
    # Run rollouts to exercise the functions
    for i in range({num_rollouts}):
        root.parallelRollouts(board.copy(), alphaZeroNet, 1)

print("Profiling complete")
"""
    
    return test_code

def profile_function(model_file, function_name, num_rollouts=100):
    """Profile a specific function using line_profiler."""
    
    # Map function names to their modules and full names
    function_map = {
        'calcUCT': ('MCTS', 'calcUCT'),
        'UCTSelect': ('MCTS', 'Node.UCTSelect'),
        'selectTask': ('MCTS', 'Root.selectTask'),
        'encodePosition': ('encoder', 'encodePosition'),
        'callNeuralNetwork': ('encoder', 'callNeuralNetwork'),
        'parallelRollouts': ('MCTS', 'Root.parallelRolloutsOptimized'),
        'expand': ('MCTS', 'Edge.expand'),
        'updateStats': ('MCTS', 'Node.updateStats')
    }
    
    if function_name not in function_map:
        print(f"Unknown function: {function_name}")
        print(f"Available functions: {', '.join(function_map.keys())}")
        return
    
    module_name, full_function_name = function_map[function_name]
    
    # Create temporary test script
    test_code = create_test_script(model_file, function_name, num_rollouts)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        test_file = f.name
    
    try:
        # Run line_profiler
        cmd = [
            'kernprof', '-l', '-v',
            '-f', f'{module_name}:{full_function_name}',
            test_file
        ]
        
        # Alternative approach using line_profiler directly
        print(f"Profiling {full_function_name} from {module_name}...")
        print("-" * 80)
        
        # Import the modules
        import MCTS
        import encoder
        import chess
        import torch
        from playchess import load_model_multi_gpu
        
        # Get the function to profile
        if module_name == 'MCTS':
            if '.' in full_function_name:
                # It's a method
                parts = full_function_name.split('.')
                if parts[0] == 'Node':
                    func = MCTS.Node.UCTSelect
                elif parts[0] == 'Root':
                    if parts[1] == 'selectTask':
                        func = MCTS.Root.selectTask
                    else:
                        func = MCTS.Root.parallelRolloutsOptimized
                elif parts[0] == 'Edge':
                    func = MCTS.Edge.expand
            else:
                func = getattr(MCTS, full_function_name)
        else:
            func = getattr(encoder, full_function_name)
        
        # Use line_profiler if available
        try:
            from line_profiler import LineProfiler
            
            lp = LineProfiler()
            lp.add_function(func)
            
            # Additional functions to profile together
            if function_name == 'UCTSelect':
                lp.add_function(MCTS.calcUCT)
            elif function_name == 'selectTask':
                lp.add_function(MCTS.Node.UCTSelect)
                lp.add_function(MCTS.calcUCT)
            
            # Load model
            models, devices = load_model_multi_gpu(model_file, None)
            alphaZeroNet = models[0]
            
            # Create board
            board = chess.Board()
            
            # Profile the execution
            with torch.no_grad():
                root = MCTS.Root(board, alphaZeroNet)
                
                # Wrap the rollout in profiler
                lp_wrapper = lp(lambda: [root.parallelRollouts(board.copy(), alphaZeroNet, 1) for _ in range(num_rollouts)])
                lp_wrapper()
            
            # Print results
            lp.print_stats()
            
        except ImportError:
            print("line_profiler not installed. Install with: pip install line_profiler")
            print("\nFalling back to basic timing analysis...")
            
            # Basic timing analysis
            import time
            
            # Load model
            models, devices = load_model_multi_gpu(model_file, None)
            alphaZeroNet = models[0]
            
            # Create board
            board = chess.Board()
            
            # Time the execution
            with torch.no_grad():
                root = MCTS.Root(board, alphaZeroNet)
                
                start = time.perf_counter()
                for _ in range(num_rollouts):
                    root.parallelRollouts(board.copy(), alphaZeroNet, 1)
                elapsed = time.perf_counter() - start
            
            print(f"\nCompleted {num_rollouts} rollouts in {elapsed:.3f} seconds")
            print(f"Average time per rollout: {elapsed/num_rollouts*1000:.3f} ms")
            
            # Get some statistics
            if function_name == 'calcUCT':
                # Count UCT calculations
                total_uct_calls = sum(len(node.edges) for node in get_all_nodes(root))
                print(f"Estimated UCT calculations: {total_uct_calls * num_rollouts}")
                print(f"UCT calculations per second: {total_uct_calls * num_rollouts / elapsed:.0f}")
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def get_all_nodes(root):
    """Get all nodes in the tree for statistics."""
    nodes = [root]
    visited = set()
    
    def traverse(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        nodes.append(node)
        
        for edge in node.edges:
            if edge.has_child():
                traverse(edge.getChild())
    
    traverse(root)
    return nodes

def main():
    parser = argparse.ArgumentParser(description='Line-by-line profiling for MCTS functions')
    parser.add_argument('--model', required=True, help='Path to model (.pt) file')
    parser.add_argument('--function', default='UCTSelect', 
                       choices=['calcUCT', 'UCTSelect', 'selectTask', 'encodePosition', 
                               'callNeuralNetwork', 'parallelRollouts', 'expand', 'updateStats'],
                       help='Function to profile')
    parser.add_argument('--rollouts', type=int, default=100, help='Number of rollouts to perform')
    
    args = parser.parse_args()
    
    profile_function(args.model, args.function, args.rollouts)

if __name__ == '__main__':
    main()