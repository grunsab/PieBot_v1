#!/usr/bin/env python3
"""
Simple benchmark to test raw neural network throughput for MCTS
"""

import os
import sys
sys.path.append('..')
import time
import chess
import torch
import numpy as np
import argparse
import AlphaZeroNetwork
import encoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

class SimpleBatchedMCTS:
    """Simple batched MCTS for benchmarking actual achievable NPS"""
    
    def __init__(self, model, device, batch_size=256):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
        
        # Request queue
        self.request_queue = queue.Queue()
        self.running = False
        self.server_thread = None
        
    def start(self):
        """Start the batch server"""
        self.running = True
        self.server_thread = threading.Thread(target=self._batch_server)
        self.server_thread.start()
        
    def stop(self):
        """Stop the batch server"""
        self.running = False
        if self.server_thread:
            self.server_thread.join()
            
    def _batch_server(self):
        """Server that batches requests"""
        while self.running:
            batch = []
            
            # Collect batch
            deadline = time.time() + 0.005  # 5ms
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = deadline - time.time()
                    if timeout > 0:
                        request = self.request_queue.get(timeout=timeout)
                        batch.append(request)
                except queue.Empty:
                    break
                    
            if not batch:
                continue
                
            # Process batch
            batch_size = len(batch)
            positions = torch.zeros((batch_size, 16, 8, 8), device=self.device)
            masks = torch.zeros((batch_size, 72 * 8 * 8), device=self.device)
            
            # Fill batch
            for i, (board, future) in enumerate(batch):
                pos, mask = encoder.encodePositionForInference(board)
                positions[i] = torch.from_numpy(pos)
                masks[i] = torch.from_numpy(mask).flatten()
                
            # Evaluate
            with torch.no_grad():
                values, policies = self.model(positions, policyMask=masks)
                
            # Return results
            values_np = values.cpu().numpy()
            policies_np = policies.cpu().numpy()
            
            for i, (board, future) in enumerate(batch):
                value = values_np[i, 0]
                policy = policies_np[i]
                move_probs = encoder.decodePolicyOutput(board, policy)
                future.set_result((value, move_probs))
                
    def evaluate_async(self, board):
        """Request async evaluation"""
        future = SimpleFuture()
        self.request_queue.put((board, future))
        return future

class SimpleFuture:
    """Simple future for async results"""
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        
    def set_result(self, result):
        self.result = result
        self.event.set()
        
    def get_result(self, timeout=None):
        if self.event.wait(timeout):
            return self.result
        raise TimeoutError("Evaluation timed out")

def benchmark_simple_mcts(model_path, device_id=0, num_positions=1000, num_workers=64, batch_size=256):
    """Benchmark simple MCTS to find actual achievable NPS"""
    
    print(f"\nSimple MCTS Benchmark")
    print(f"Positions: {num_positions}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print("-" * 40)
    
    # Load model
    device = torch.device(f'cuda:{device_id}')
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    
    # Create MCTS server
    mcts = SimpleBatchedMCTS(model, device, batch_size)
    mcts.start()
    
    # Test positions
    board = chess.Board()
    
    # Warmup
    print("Warming up...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _ in range(100):
            future = executor.submit(lambda: mcts.evaluate_async(board).get_result(5.0))
            futures.append(future)
        for f in futures:
            f.result()
    
    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    
    def evaluate_position():
        return mcts.evaluate_async(board).get_result(5.0)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _ in range(num_positions):
            future = executor.submit(evaluate_position)
            futures.append(future)
            
        # Wait for all to complete
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as e:
                print(f"Error: {e}")
                
    end_time = time.time()
    elapsed = end_time - start_time
    
    nps = completed / elapsed
    print(f"\nResults:")
    print(f"  Completed: {completed}/{num_positions}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  NPS: {nps:,.0f}")
    
    # Stop server
    mcts.stop()
    
    return nps

def main():
    parser = argparse.ArgumentParser(description='Simple MCTS benchmark')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--positions', type=int, default=10000, help='Number of positions to evaluate')
    parser.add_argument('--workers', type=int, default=64, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    
    args = parser.parse_args()
    
    # Run benchmark
    nps = benchmark_simple_mcts(
        args.model,
        args.device,
        args.positions,
        args.workers,
        args.batch_size
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL NPS: {nps:,.0f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()