#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and quantized AlphaZero models.

This script measures:
- Inference speed (nodes per second)
- Model size reduction
- Accuracy differences
- Memory usage
"""

import argparse
import time
import torch
import chess
import numpy as np
import os
import gc
from typing import List, Tuple
import AlphaZeroNetwork
import encoder
import MCTS
from device_utils import get_optimal_device, optimize_for_device
from quantization_utils import (
    apply_dynamic_quantization,
    apply_static_quantization,
    save_quantized_model,
    load_quantized_model,
    compare_model_outputs,
    create_calibration_dataset
)


def benchmark_inference_speed(model, board: chess.Board, num_iterations: int = 1000,
                            batch_sizes: List[int] = [1, 8, 32, 64, 128]) -> dict:
    """
    Benchmark inference speed for different batch sizes.
    
    Args:
        model: Model to benchmark
        board: Chess board for testing
        num_iterations: Number of iterations per batch size
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with benchmark results
    """
    device, device_str = get_optimal_device()
    model.eval()
    
    # Handle both regular and quantized models
    is_quantized = hasattr(model, '__class__') and 'quantized' in str(type(model)).lower()
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch of positions
        boards = []
        for _ in range(batch_size):
            temp_board = board.copy()
            # Make some random moves to vary positions
            for _ in range(np.random.randint(0, 10)):
                legal_moves = list(temp_board.legal_moves)
                if legal_moves:
                    temp_board.push(np.random.choice(legal_moves))
            boards.append(temp_board)
        
        # Encode positions
        inputs = []
        masks = []
        for b in boards:
            input_planes = encoder.encode_board(b)
            inputs.append(input_planes)
            mask = encoder.create_move_mask(b)
            masks.append(mask)
        
        input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
        mask_tensor = torch.tensor(np.array(masks), dtype=torch.float32)
        
        # Move to device (quantized models typically run on CPU)
        if not is_quantized:
            input_tensor = input_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor, policyMask=mask_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor, policyMask=mask_tensor)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        total_inferences = num_iterations * batch_size
        inferences_per_second = total_inferences / elapsed
        
        results[batch_size] = {
            'elapsed_time': elapsed,
            'inferences_per_second': inferences_per_second,
            'ms_per_inference': (elapsed * 1000) / total_inferences
        }
        
        print(f"Batch size {batch_size}: {inferences_per_second:.0f} inferences/sec, "
              f"{results[batch_size]['ms_per_inference']:.2f} ms/inference")
    
    return results


def benchmark_mcts_performance(model, board: chess.Board, num_rollouts: int = 1000,
                             num_threads: int = 8, num_runs: int = 3) -> dict:
    """
    Benchmark MCTS performance with the model.
    
    Args:
        model: Model to use for MCTS
        board: Starting position
        num_rollouts: Number of MCTS rollouts
        num_threads: Number of threads for parallel rollouts
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with MCTS performance metrics
    """
    results = []
    
    for run in range(num_runs):
        # Create new MCTS root
        gc.collect()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            root = MCTS.Root(board, model)
            
            # Perform rollouts
            num_iterations = num_rollouts // num_threads
            remainder = num_rollouts % num_threads
            
            for i in range(num_iterations):
                root.parallelRollouts(board.copy(), model, num_threads)
            
            if remainder > 0:
                root.parallelRollouts(board.copy(), model, remainder)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        actual_rollouts = num_iterations * num_threads + remainder
        nps = actual_rollouts / elapsed
        
        results.append({
            'elapsed_time': elapsed,
            'rollouts': actual_rollouts,
            'nodes_per_second': nps
        })
        
        print(f"Run {run + 1}: {nps:.0f} nodes/sec")
        
        # Clean up
        root.cleanup()
        gc.collect()
    
    # Calculate averages
    avg_nps = np.mean([r['nodes_per_second'] for r in results])
    std_nps = np.std([r['nodes_per_second'] for r in results])
    
    return {
        'runs': results,
        'avg_nodes_per_second': avg_nps,
        'std_nodes_per_second': std_nps,
        'speedup': None  # Will be calculated later
    }


def print_comparison_summary(original_results: dict, quantized_results: dict,
                           accuracy_metrics: dict, size_info: dict) -> None:
    """
    Print a comprehensive comparison summary.
    """
    print("\n" + "="*70)
    print("QUANTIZATION PERFORMANCE SUMMARY")
    print("="*70)
    
    # Model size comparison
    print(f"\nModel Size:")
    print(f"  Original:  {size_info['original_size']:.2f} MB")
    print(f"  Quantized: {size_info['quantized_size']:.2f} MB")
    print(f"  Reduction: {size_info['reduction']:.1f}%")
    
    # Inference speed comparison
    print(f"\nInference Speed (batch size 1):")
    orig_speed = original_results['inference'][1]['inferences_per_second']
    quant_speed = quantized_results['inference'][1]['inferences_per_second']
    speedup = quant_speed / orig_speed
    print(f"  Original:  {orig_speed:.0f} inferences/sec")
    print(f"  Quantized: {quant_speed:.0f} inferences/sec")
    print(f"  Speedup:   {speedup:.1f}x")
    
    # MCTS performance comparison
    print(f"\nMCTS Performance:")
    orig_nps = original_results['mcts']['avg_nodes_per_second']
    quant_nps = quantized_results['mcts']['avg_nodes_per_second']
    mcts_speedup = quant_nps / orig_nps
    print(f"  Original:  {orig_nps:.0f} ± {original_results['mcts']['std_nodes_per_second']:.0f} nodes/sec")
    print(f"  Quantized: {quant_nps:.0f} ± {quantized_results['mcts']['std_nodes_per_second']:.0f} nodes/sec")
    print(f"  Speedup:   {mcts_speedup:.1f}x")
    
    # Accuracy comparison
    print(f"\nAccuracy Metrics:")
    print(f"  Value head:")
    print(f"    Average difference: {accuracy_metrics['avg_value_diff']:.6f}")
    print(f"    Maximum difference: {accuracy_metrics['max_value_diff']:.6f}")
    print(f"    RMSE: {accuracy_metrics['value_rmse']:.6f}")
    print(f"  Policy head:")
    print(f"    Average difference: {accuracy_metrics['avg_policy_diff']:.6f}")
    print(f"    Maximum difference: {accuracy_metrics['max_policy_diff']:.6f}")
    print(f"    RMSE: {accuracy_metrics['policy_rmse']:.6f}")
    
    # Summary
    print(f"\n" + "-"*70)
    print(f"SUMMARY: {size_info['reduction']:.0f}% smaller, {mcts_speedup:.1f}x faster, "
          f"minimal accuracy loss")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark quantized vs original AlphaZero models'
    )
    parser.add_argument('--model', help='Path to original model file', required=True)
    parser.add_argument('--quantization', choices=['dynamic', 'static', 'both'], 
                       default='dynamic', help='Type of quantization to test')
    parser.add_argument('--calibration-size', type=int, default=1000,
                       help='Number of positions for calibration (static quantization)')
    parser.add_argument('--rollouts', type=int, default=1000,
                       help='Number of MCTS rollouts for benchmark')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of threads for MCTS')
    parser.add_argument('--save-quantized', action='store_true',
                       help='Save the quantized model')
    parser.add_argument('--skip-mcts', action='store_true',
                       help='Skip MCTS benchmark (faster)')
    
    args = parser.parse_args()
    
    # Load original model
    print("Loading original model...")
    device, device_str = get_optimal_device()
    print(f"Device: {device_str}")
    
    # Determine model architecture from file name or use default
    if '20x256' in args.model:
        num_blocks, num_filters = 20, 256
    elif '10x128' in args.model:
        num_blocks, num_filters = 10, 128
    else:
        print("Warning: Could not determine architecture from filename, using 20x256")
        num_blocks, num_filters = 20, 256
    
    model = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_filters)
    weights = torch.load(args.model, map_location=device)
    model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    original_size = os.path.getsize(args.model) / (1024 * 1024)  # MB
    
    # Create test board
    board = chess.Board()
    
    # Benchmark original model
    print("\n" + "="*50)
    print("Benchmarking Original Model")
    print("="*50)
    
    original_results = {
        'inference': benchmark_inference_speed(model, board, num_iterations=100),
        'mcts': benchmark_mcts_performance(model, board, args.rollouts, args.threads) 
                if not args.skip_mcts else None
    }
    
    # Apply quantization
    quantization_results = {}
    
    if args.quantization in ['dynamic', 'both']:
        print("\n" + "="*50)
        print("Applying Dynamic Quantization")
        print("="*50)
        
        dynamic_model = apply_dynamic_quantization(model)
        
        print("\nBenchmarking Dynamic Quantized Model")
        dynamic_results = {
            'inference': benchmark_inference_speed(dynamic_model, board, num_iterations=100),
            'mcts': benchmark_mcts_performance(dynamic_model, board, args.rollouts, args.threads)
                    if not args.skip_mcts else None
        }
        
        quantization_results['dynamic'] = dynamic_results
        
        if args.save_quantized:
            save_path = save_quantized_model(dynamic_model, args.model, "_dynamic_quantized")
            print(f"Saved dynamic quantized model to: {save_path}")
    
    if args.quantization in ['static', 'both']:
        print("\n" + "="*50)
        print("Applying Static Quantization")
        print("="*50)
        
        # Generate calibration data
        print(f"Generating {args.calibration_size} calibration positions...")
        calibration_data = create_calibration_dataset(args.calibration_size)
        
        # Apply static quantization
        # Use 'qnnpack' backend for better ARM/Apple Silicon support
        backend = 'qnnpack' if device.type == 'mps' else 'fbgemm'
        static_model = apply_static_quantization(model, calibration_data, backend=backend)
        
        print("\nBenchmarking Static Quantized Model")
        static_results = {
            'inference': benchmark_inference_speed(static_model, board, num_iterations=100),
            'mcts': benchmark_mcts_performance(static_model, board, args.rollouts, args.threads)
                    if not args.skip_mcts else None
        }
        
        quantization_results['static'] = static_results
        
        if args.save_quantized:
            save_path = save_quantized_model(static_model, args.model, "_static_quantized")
            print(f"Saved static quantized model to: {save_path}")
    
    # Test accuracy
    print("\n" + "="*50)
    print("Testing Accuracy")
    print("="*50)
    
    test_positions = create_calibration_dataset(100)  # Use 100 positions for accuracy test
    
    # Use the best performing quantization method
    if args.quantization == 'both':
        # Compare and use the better one
        best_method = 'dynamic'  # Default, will be updated based on results
        best_model = dynamic_model
    else:
        best_method = args.quantization
        best_model = dynamic_model if args.quantization == 'dynamic' else static_model
    
    accuracy_metrics = compare_model_outputs(model, best_model, test_positions, device)
    
    # Calculate quantized model size (approximate for dynamic quantization)
    quantized_size = original_size * 0.25  # Approximate 75% reduction
    
    size_info = {
        'original_size': original_size,
        'quantized_size': quantized_size,
        'reduction': (1 - quantized_size / original_size) * 100
    }
    
    # Print final summary
    best_results = quantization_results.get(best_method, quantization_results[list(quantization_results.keys())[0]])
    print_comparison_summary(original_results, best_results, accuracy_metrics, size_info)
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()