#!/usr/bin/env python3
"""
Profiling tool for inference server to identify bottlenecks in batch processing.

This script measures the performance of different components in the inference pipeline:
- Board encoding time
- Tensor preparation and device transfer
- Neural network inference
- Result decoding and distribution
"""

import time
import torch
import chess
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
import uuid
from collections import defaultdict
import argparse

import AlphaZeroNetwork
import encoder
from inference_server import InferenceRequest, InferenceResult, InferenceServer
from MCTS_root_parallel import RootParallelMCTS
from device_utils import get_optimal_device, optimize_for_device


class ProfiledInferenceServer(InferenceServer):
    """Extended inference server with detailed profiling."""
    
    def __init__(self, model, device, batch_size=64, timeout_ms=150):
        super().__init__(model, device, batch_size, timeout_ms)
        
        # Detailed timing statistics
        self.timing_stats = defaultdict(list)
        self.batch_sizes = []
        
    def process_batch(self, request_tuples):
        """Process a batch with detailed timing measurements."""
        if not request_tuples:
            return
            
        total_start = time.perf_counter()
        batch_size = len(request_tuples)
        self.batch_sizes.append(batch_size)
        
        # Phase 1: Deduplication
        dedup_start = time.perf_counter()
        unique_boards = {}
        request_mapping = {}
        
        for req, result_queue in request_tuples:
            board = req.to_board()
            board_hash = self.get_position_hash(board)
            
            if board_hash not in unique_boards:
                unique_boards[board_hash] = board
                request_mapping[board_hash] = []
            request_mapping[board_hash].append((req, result_queue))
        
        dedup_time = time.perf_counter() - dedup_start
        self.timing_stats['deduplication'].append(dedup_time)
        
        # Phase 2: Encoding
        encoding_start = time.perf_counter()
        unique_count = len(unique_boards)
        inputs = torch.zeros((unique_count, 16, 8, 8), dtype=torch.float32)
        masks = torch.zeros((unique_count, 72, 8, 8), dtype=torch.float32)
        board_list = []
        
        for i, (board_hash, board) in enumerate(unique_boards.items()):
            position, mask = self.encode_board(board)
            inputs[i] = position
            masks[i] = mask
            board_list.append(board)
        
        encoding_time = time.perf_counter() - encoding_start
        self.timing_stats['encoding'].append(encoding_time)
        
        # Phase 3: Device transfer
        transfer_start = time.perf_counter()
        inputs = inputs.to(self.device)
        masks = masks.to(self.device)
        
        if next(self.model.parameters()).dtype == torch.float16:
            inputs = inputs.half()
            masks = masks.half()
        
        masks_flat = masks.view(masks.shape[0], -1)
        transfer_time = time.perf_counter() - transfer_start
        self.timing_stats['device_transfer'].append(transfer_time)
        
        # Phase 4: Neural network inference
        inference_start = time.perf_counter()
        with torch.no_grad():
            values, policies = self.model(inputs, policyMask=masks_flat)
        
        # Ensure GPU operations complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = time.perf_counter() - inference_start
        self.timing_stats['nn_inference'].append(inference_time)
        
        # Phase 5: Decoding and distribution
        decode_start = time.perf_counter()
        values = values.cpu().numpy().reshape((unique_count,))
        policies = policies.cpu().numpy()
        
        for i, (board_hash, board) in enumerate(unique_boards.items()):
            value = values[i]
            move_probs = encoder.decodePolicyOutput(board, policies[i])
            
            for req, result_queue in request_mapping[board_hash]:
                result = InferenceResult(req.request_id, value, move_probs)
                result_queue.put(result)
        
        decode_time = time.perf_counter() - decode_start
        self.timing_stats['decoding'].append(decode_time)
        
        # Total time
        total_time = time.perf_counter() - total_start
        self.timing_stats['total'].append(total_time)
        
        # Update base statistics
        self.total_requests += batch_size
        self.total_batches += 1
        self.total_time += total_time
        
    def print_profiling_report(self):
        """Print detailed profiling statistics."""
        print("\n" + "="*80)
        print("INFERENCE SERVER PROFILING REPORT")
        print("="*80)
        
        print(f"\nTotal batches processed: {self.total_batches}")
        print(f"Total requests processed: {self.total_requests}")
        
        if self.batch_sizes:
            print(f"Average batch size: {np.mean(self.batch_sizes):.1f}")
            print(f"Max batch size: {max(self.batch_sizes)}")
            print(f"Min batch size: {min(self.batch_sizes)}")
        
        print("\nTiming breakdown (milliseconds):")
        print("-"*60)
        print(f"{'Phase':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'%':>8}")
        print("-"*60)
        
        total_mean = np.mean(self.timing_stats['total']) * 1000 if self.timing_stats['total'] else 0
        
        for phase in ['deduplication', 'encoding', 'device_transfer', 'nn_inference', 'decoding', 'total']:
            if phase in self.timing_stats and self.timing_stats[phase]:
                times_ms = np.array(self.timing_stats[phase]) * 1000
                mean_time = np.mean(times_ms)
                std_time = np.std(times_ms)
                min_time = np.min(times_ms)
                max_time = np.max(times_ms)
                
                if phase != 'total' and total_mean > 0:
                    percentage = (mean_time / total_mean) * 100
                else:
                    percentage = 100 if phase == 'total' else 0
                
                print(f"{phase:<20} {mean_time:>10.2f} {std_time:>10.2f} {min_time:>10.2f} {max_time:>10.2f} {percentage:>7.1f}%")
        
        print("-"*60)
        
        # Calculate throughput
        if self.total_time > 0:
            throughput = self.total_requests / self.total_time
            print(f"\nOverall throughput: {throughput:.0f} requests/second")
            print(f"Average batch latency: {(self.total_time / self.total_batches) * 1000:.1f} ms")


def simulate_worker_requests(worker_id, num_rollouts, request_queue, result_queue, board_fen):
    """Simulate a worker sending inference requests."""
    board = chess.Board(board_fen)
    
    for i in range(num_rollouts):
        # Simulate MCTS exploration by making random moves
        temp_board = board.copy()
        for _ in range(np.random.randint(0, 5)):
            legal_moves = list(temp_board.legal_moves)
            if legal_moves:
                move = np.random.choice(legal_moves)
                temp_board.push(move)
        
        request_id = f"worker_{worker_id}_req_{i}"
        request = InferenceRequest(request_id, temp_board.fen(), worker_id)
        request_queue.put((request, result_queue))
        
        # Simulate processing time between requests
        time.sleep(np.random.uniform(0.0001, 0.001))


def profile_inference_server(model_path, num_workers=8, rollouts_per_worker=100, 
                            batch_size=64, timeout_ms=150):
    """Profile the inference server with simulated MCTS workers."""
    
    print(f"Loading model from {model_path}...")
    device, device_str = get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Load model
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    try:
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights)
    except:
        model = AlphaZeroNetwork.AlphaZeroNet(10, 128)
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    # Create profiled server
    server = ProfiledInferenceServer(model, device, batch_size, timeout_ms)
    
    # Setup multiprocessing
    manager = Manager()
    request_queue = manager.Queue()
    stop_event = manager.Event()
    
    # Start inference server in separate thread
    import threading
    server_thread = threading.Thread(target=server.run, args=(request_queue, stop_event))
    server_thread.start()
    
    print(f"\nStarting profiling with {num_workers} workers, {rollouts_per_worker} rollouts each...")
    print(f"Batch size: {batch_size}, Timeout: {timeout_ms}ms")
    print("-"*60)
    
    # Create worker processes
    start_time = time.perf_counter()
    worker_processes = []
    result_queues = []
    
    # Starting position for all workers
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board_fen = board.fen()
    
    for i in range(num_workers):
        result_queue = manager.Queue()
        result_queues.append(result_queue)
        
        p = Process(target=simulate_worker_requests, 
                   args=(i, rollouts_per_worker, request_queue, result_queue, board_fen))
        p.start()
        worker_processes.append(p)
    
    # Wait for all workers to complete
    for p in worker_processes:
        p.join()
    
    # Give server time to process remaining requests
    time.sleep(0.5)
    
    # Stop server
    stop_event.set()
    server_thread.join()
    
    total_time = time.perf_counter() - start_time
    
    # Print profiling report
    server.print_profiling_report()
    
    print(f"\nTotal wall-clock time: {total_time:.2f} seconds")
    print(f"Effective throughput: {(num_workers * rollouts_per_worker) / total_time:.0f} requests/second")
    
    return server.timing_stats


def compare_configurations():
    """Compare different batch sizes and timeout configurations."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python profile_inference_server.py <model_path> [--compare]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    configurations = [
        # (batch_size, timeout_ms, num_workers)
        (20, 50, 8),
        (32, 50, 8),
        (64, 100, 8),
        (128, 150, 8),
        (256, 200, 8),
        (64, 100, 4),
        (64, 100, 16),
    ]
    
    print("Comparing different configurations...")
    print("="*80)
    
    results = []
    
    for batch_size, timeout_ms, num_workers in configurations:
        print(f"\nConfiguration: batch_size={batch_size}, timeout={timeout_ms}ms, workers={num_workers}")
        print("-"*60)
        
        stats = profile_inference_server(
            model_path, 
            num_workers=num_workers,
            rollouts_per_worker=1000,
            batch_size=batch_size,
            timeout_ms=timeout_ms
        )
        
        # Calculate key metrics
        if stats['total']:
            avg_latency = np.mean(stats['total']) * 1000
            avg_batch_size = np.mean(stats['batch_sizes']) if 'batch_sizes' in stats else 0
            results.append({
                'config': f"BS={batch_size}, T={timeout_ms}ms, W={num_workers}",
                'latency': avg_latency,
                'batch_size': avg_batch_size
            })
    
    # Summary
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Avg Latency (ms)':>20} {'Avg Batch Size':>20}")
    print("-"*70)
    
    for result in results:
        print(f"{result['config']:<30} {result['latency']:>20.2f} {result['batch_size']:>20.1f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python profile_inference_server.py <model_path> [--compare]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if '--compare' in sys.argv:
        compare_configurations()
    else:
        # Default profiling
        profile_inference_server(
            model_path,
            num_workers=8,
            rollouts_per_worker=100,
            batch_size=64,
            timeout_ms=150
        )