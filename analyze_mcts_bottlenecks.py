"""
Analyze MCTS_root_parallel.py bottlenecks with detailed profiling.
"""

import cProfile
import pstats
import io
import time
import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_root_parallel import RootParallelMCTS
import device_utils
import sys

def detailed_profile():
    # Load model
    device, device_str = device_utils.get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Create a simple test model
    model = AlphaZeroNet(num_blocks=10, num_filters=128).to(device)
    model.eval()
    
    # Initialize board
    board = chess.Board()
    
    # Create MCTS engine with minimal workers for clearer profiling
    num_workers = 4  # Fewer workers to reduce noise
    epsilon = 0.0
    inference_batch_size = 128
    inference_timeout_ms = 50
    
    print(f"\nTesting with {num_workers} workers")
    print(f"Inference batch size: {inference_batch_size}")
    print(f"Inference timeout: {inference_timeout_ms}ms")
    
    mcts = RootParallelMCTS(
        model=model,
        num_workers=num_workers,
        epsilon=epsilon,
        alpha=0.3,
        inference_batch_size=inference_batch_size,
        inference_timeout_ms=inference_timeout_ms
    )
    
    # Smaller test for detailed analysis
    total_rollouts = 10_000
    
    print(f"\n{'=' * 60}")
    print("Running detailed timing analysis...")
    print(f"{'=' * 60}")
    
    # Time different phases
    phases = {}
    
    # 1. Time initialization
    start = time.perf_counter()
    mcts2 = RootParallelMCTS(
        model=model,
        num_workers=num_workers,
        epsilon=epsilon,
        alpha=0.3,
        inference_batch_size=inference_batch_size,
        inference_timeout_ms=inference_timeout_ms
    )
    phases['initialization'] = time.perf_counter() - start
    mcts2.cleanup()
    
    # 2. Time task distribution
    import uuid
    from MCTS_root_parallel import WorkerTask
    import numpy as np
    
    start = time.perf_counter()
    task_id = uuid.uuid4().hex
    rollouts_per_worker = total_rollouts // num_workers
    for i in range(num_workers):
        noise_seed = np.random.randint(0, 2**16)
        task = WorkerTask(
            task_id, board, rollouts_per_worker, 
            noise_seed, epsilon, 0.3
        )
        mcts.task_queue.put(task)
    phases['task_distribution'] = time.perf_counter() - start
    
    # 3. Time result collection (run actual search)
    start = time.perf_counter()
    results = []
    for _ in range(num_workers):
        result = mcts.result_queue.get()
        results.append(result)
    phases['result_collection'] = time.perf_counter() - start
    
    # 4. Time aggregation
    start = time.perf_counter()
    stats = mcts.aggregate_statistics(results)
    phases['aggregation'] = time.perf_counter() - start
    
    # Clean up
    mcts.cleanup()
    
    # Print timing breakdown
    print("\nPhase Timing Breakdown:")
    print("-" * 40)
    total_time = sum(phases.values())
    for phase, duration in phases.items():
        percentage = (duration / total_time) * 100
        print(f"{phase:20s}: {duration:8.4f}s ({percentage:5.1f}%)")
    print(f"{'Total':20s}: {total_time:8.4f}s")
    
    # Calculate throughput
    actual_throughput = total_rollouts / phases['result_collection']
    print(f"\nActual search throughput: {actual_throughput:.0f} rollouts/sec")
    
    # Analyze queue operations
    print(f"\n{'=' * 60}")
    print("Analyzing synchronization overhead...")
    print(f"{'=' * 60}")
    
    from multiprocessing import Manager
    import queue as q
    
    manager = Manager()
    test_queue = manager.Queue()
    
    # Test queue latency
    items = 1000
    start = time.perf_counter()
    for i in range(items):
        test_queue.put(i)
    put_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(items):
        _ = test_queue.get()
    get_time = time.perf_counter() - start
    
    print(f"Queue operations (multiprocessing.Manager):")
    print(f"  Put latency: {(put_time/items)*1000:.3f} ms per item")
    print(f"  Get latency: {(get_time/items)*1000:.3f} ms per item")
    print(f"  Total overhead for {total_rollouts} items: {((put_time+get_time)/items)*total_rollouts:.2f}s")
    
    # Memory analysis
    print(f"\n{'=' * 60}")
    print("Memory usage analysis...")
    print(f"{'=' * 60}")
    
    import tracemalloc
    tracemalloc.start()
    
    # Run a small search to measure memory
    mcts3 = RootParallelMCTS(
        model=model,
        num_workers=2,
        epsilon=epsilon,
        alpha=0.3,
        inference_batch_size=64,
        inference_timeout_ms=50
    )
    
    stats = mcts3.run_parallel_search(board, 1000)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Memory usage for 1000 rollouts:")
    print(f"  Current: {current / 1024 / 1024:.2f} MB")
    print(f"  Peak: {peak / 1024 / 1024:.2f} MB")
    print(f"  Estimated for 100k rollouts: {(peak / 1024 / 1024) * 100:.2f} MB")
    
    mcts3.cleanup()
    
    # Identify bottlenecks
    print(f"\n{'=' * 60}")
    print("KEY BOTTLENECKS IDENTIFIED:")
    print(f"{'=' * 60}")
    
    bottlenecks = []
    
    # Check if inference is the bottleneck
    inference_percentage = (phases['result_collection'] / total_time) * 100
    if inference_percentage > 70:
        bottlenecks.append(f"1. Neural network inference ({inference_percentage:.1f}% of time)")
        bottlenecks.append("   - Consider larger batch sizes")
        bottlenecks.append("   - Use mixed precision (FP16)")
        bottlenecks.append("   - Optimize model architecture")
    
    # Check queue overhead
    queue_overhead = ((put_time+get_time)/items)*total_rollouts
    queue_percentage = (queue_overhead / total_time) * 100
    if queue_percentage > 10:
        bottlenecks.append(f"2. Queue communication overhead ({queue_percentage:.1f}% estimated)")
        bottlenecks.append("   - Consider shared memory instead of queues")
        bottlenecks.append("   - Batch more operations together")
        bottlenecks.append("   - Use faster IPC mechanisms")
    
    # Check aggregation overhead
    agg_percentage = (phases['aggregation'] / total_time) * 100
    if agg_percentage > 5:
        bottlenecks.append(f"3. Result aggregation ({agg_percentage:.1f}% of time)")
        bottlenecks.append("   - Use numpy arrays for faster aggregation")
        bottlenecks.append("   - Pre-allocate data structures")
    
    # Check initialization overhead
    init_percentage = (phases['initialization'] / total_time) * 100
    if init_percentage > 5:
        bottlenecks.append(f"4. Process initialization ({init_percentage:.1f}% of time)")
        bottlenecks.append("   - Reuse worker processes across searches")
        bottlenecks.append("   - Use process pools")
    
    if not bottlenecks:
        bottlenecks.append("System appears well-balanced")
    
    for bottleneck in bottlenecks:
        print(bottleneck)
    
    # Optimization recommendations
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION RECOMMENDATIONS:")
    print(f"{'=' * 60}")
    
    recommendations = [
        "1. Implement C++ extension for tree operations",
        "   - UCT selection, expansion, backpropagation in C++",
        "   - 10-100x speedup potential for tree operations",
        "",
        "2. Use shared memory for tree structures",
        "   - Eliminate queue serialization overhead",
        "   - Direct memory access between processes",
        "",
        "3. Optimize batch inference",
        f"   - Current batch size: {inference_batch_size}",
        "   - Try larger batches (512-1024) if GPU memory allows",
        "   - Implement dynamic batching based on queue size",
        "",
        "4. Consider virtual loss for better parallelization",
        "   - Reduce waiting time between rollouts",
        "   - Better GPU utilization",
        "",
        "5. Profile with NVIDIA Nsight for GPU optimization",
        "   - Identify kernel bottlenecks",
        "   - Optimize memory transfers"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    detailed_profile()