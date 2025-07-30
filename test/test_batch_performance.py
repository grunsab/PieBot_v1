#!/usr/bin/env python3
"""
Test script to measure the performance difference between single and batched inference.
"""

import sys
sys.path.append('..')
import torch
import time
import numpy as np
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
import encoder
import chess


def test_single_inference(model, device, num_tests=1000):
    """Test single inference performance."""
    board = chess.Board()
    position, mask = encoder.encodePositionForInference(board)
    
    position_tensor = torch.from_numpy(position).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(position_tensor, policyMask=mask_tensor)
    
    # Test
    start_time = time.perf_counter()
    
    for _ in range(num_tests):
        with torch.no_grad():
            value, policy = model(position_tensor, policyMask=mask_tensor)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    return num_tests / elapsed, elapsed


def test_batch_inference(model, device, batch_size=32, num_tests=1000):
    """Test batched inference performance."""
    board = chess.Board()
    position, mask = encoder.encodePositionForInference(board)
    
    # Create batch
    position_batch = torch.from_numpy(position).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    mask_batch = torch.from_numpy(mask).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(position_batch, policyMask=mask_batch)
    
    # Test
    num_batches = num_tests // batch_size
    
    start_time = time.perf_counter()
    
    for _ in range(num_batches):
        with torch.no_grad():
            value, policy = model(position_batch, policyMask=mask_batch)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    total_inferences = num_batches * batch_size
    return total_inferences / elapsed, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test batch vs single inference performance')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--num-tests', type=int, default=1000, help='Number of test inferences')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device, device_str = get_optimal_device()
    print(f"Device: {device_str}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(args.model, map_location=device)
    model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"\nTesting with {args.num_tests} inferences...\n")
    
    # Test single inference
    single_ips, single_time = test_single_inference(model, device, args.num_tests)
    print(f"Single inference: {single_ips:.0f} inferences/sec ({single_time:.2f}s total)")
    
    # Test various batch sizes
    batch_sizes = [8, 16, 32, 64, 128, 256]
    
    print("\nBatch inference performance:")
    print(f"{'Batch Size':>10} | {'Inferences/sec':>15} | {'Speedup':>10} | {'Time/batch (ms)':>15}")
    print("-" * 65)
    
    for batch_size in batch_sizes:
        if batch_size > args.num_tests:
            continue
            
        batch_ips, batch_time = test_batch_inference(model, device, batch_size, args.num_tests)
        speedup = batch_ips / single_ips
        time_per_batch = (batch_time / (args.num_tests // batch_size)) * 1000
        
        print(f"{batch_size:>10} | {batch_ips:>15.0f} | {speedup:>10.1f}x | {time_per_batch:>15.1f}")
    
    print("\nConclusion:")
    print(f"Best batch size for {device_str}: {batch_sizes[-1]} (on this hardware)")


if __name__ == '__main__':
    main()