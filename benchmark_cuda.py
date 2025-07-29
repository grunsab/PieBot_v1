#!/usr/bin/env python3
"""
Performance Benchmark for AlphaZero Chess Engine - Windows/CUDA Optimized

Comprehensive benchmarking tool that measures:
- Nodes per second (NPS) for MCTS search
- Neural network inference throughput
- Batch processing efficiency
- GPU utilization and memory usage
- Comparison with baseline performance
"""

import os
import sys
import time
import chess
import torch
import numpy as np
import argparse
from datetime import datetime
import json
import threading
import queue
import psutil
import AlphaZeroNetwork
from MCTS_cuda import MCTSEngineCUDA
import encoder

# Windows-specific imports
if sys.platform == 'win32':
    import pynvml
    try:
        pynvml.nvmlInit()
    except:
        print("Warning: NVIDIA Management Library not available")
        pynvml = None

class GPUMonitor:
    """Monitor GPU utilization during benchmarks"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.handle = None
        
        if sys.platform == 'win32' and pynvml:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except:
                pass
    
    def start(self):
        """Start monitoring GPU"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.samples:
            return None
        
        # Calculate statistics
        gpu_utils = [s['gpu_util'] for s in self.samples if 'gpu_util' in s]
        mem_utils = [s['mem_util'] for s in self.samples if 'mem_util' in s]
        temps = [s['temperature'] for s in self.samples if 'temperature' in s]
        
        stats = {
            'gpu_util_avg': np.mean(gpu_utils) if gpu_utils else 0,
            'gpu_util_max': np.max(gpu_utils) if gpu_utils else 0,
            'mem_util_avg': np.mean(mem_utils) if mem_utils else 0,
            'mem_util_max': np.max(mem_utils) if mem_utils else 0,
            'temperature_avg': np.mean(temps) if temps else 0,
            'temperature_max': np.max(temps) if temps else 0,
            'samples': len(self.samples)
        }
        
        return stats
    
    def _monitor_loop(self):
        """Monitor GPU in a loop"""
        while self.monitoring:
            sample = {'timestamp': time.time()}
            
            if self.handle and pynvml:
                try:
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    sample['gpu_util'] = util.gpu
                    sample['mem_util'] = util.memory
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    sample['temperature'] = temp
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    sample['mem_used'] = mem_info.used / 1024**3  # GB
                    sample['mem_total'] = mem_info.total / 1024**3  # GB
                except:
                    pass
            
            self.samples.append(sample)
            time.sleep(0.1)  # Sample every 100ms

class BenchmarkRunner:
    """Run various benchmarks on the chess engine"""
    
    def __init__(self, model_path, device_id=0, verbose=False):
        self.model_path = model_path
        self.device_id = device_id
        self.verbose = verbose
        self.device = torch.device(f'cuda:{device_id}')
        self.model = None
        self.mcts_engine = None
        self.gpu_monitor = GPUMonitor(device_id)
        
    def load_model(self):
        """Load the neural network model"""
        print(f"Loading model from: {self.model_path}")
        self.model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)  # Move model to GPU
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
    def benchmark_neural_network(self, batch_sizes=[1, 8, 32, 64, 128, 256, 512], num_iterations=100):
        """Benchmark neural network inference at various batch sizes"""
        print("\n" + "="*60)
        print("Neural Network Inference Benchmark")
        print("="*60)
        
        results = []
        
        for batch_size in batch_sizes:
            # Create dummy input
            positions = torch.randn(batch_size, 16, 8, 8).to(self.device)
            masks = torch.ones(batch_size, 72 * 8 * 8).to(self.device)  # Flattened mask
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(positions, policyMask=masks)
            
            # Synchronize before timing
            torch.cuda.synchronize(self.device)
            
            # Time iterations
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    values, policies = self.model(positions, policyMask=masks)
            
            torch.cuda.synchronize(self.device)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            positions_per_sec = (batch_size * num_iterations) / elapsed
            ms_per_batch = (elapsed / num_iterations) * 1000
            
            result = {
                'batch_size': batch_size,
                'positions_per_sec': positions_per_sec,
                'ms_per_batch': ms_per_batch,
                'efficiency': positions_per_sec / batch_size  # Normalized
            }
            results.append(result)
            
            print(f"Batch size {batch_size:4d}: {positions_per_sec:8.0f} pos/sec, "
                  f"{ms_per_batch:6.2f} ms/batch, efficiency: {result['efficiency']:.1f}")
        
        return results
    
    def benchmark_mcts(self, positions, rollout_configs):
        """Benchmark MCTS performance with different configurations"""
        print("\n" + "="*60)
        print("MCTS Search Benchmark")
        print("="*60)
        
        results = []
        
        for config in rollout_configs:
            rollouts = config['rollouts']
            threads = config['threads']
            batch_size = config['batch_size']
            
            print(f"\nConfiguration: {rollouts} rollouts, {threads} threads, batch size {batch_size}")
            
            # Create MCTS engine
            if self.mcts_engine:
                self.mcts_engine.stop()
            
            self.mcts_engine = MCTSEngineCUDA(
                self.model,
                device=self.device,
                max_batch_size=batch_size,
                num_workers=threads,
                verbose=False
            )
            self.mcts_engine.start()
            
            # Allow engine to initialize
            time.sleep(0.5)
            
            # Run searches on all positions
            position_results = []
            total_nodes = 0
            
            # Start GPU monitoring
            self.gpu_monitor.start()
            
            start_time = time.perf_counter()
            
            for i, board in enumerate(positions):
                pos_start = time.perf_counter()
                
                # Run MCTS search
                best_move = self.mcts_engine.search(board, rollouts)
                
                pos_end = time.perf_counter()
                pos_time = pos_end - pos_start
                
                nodes_per_sec = rollouts / pos_time if pos_time > 0 else 0
                position_results.append({
                    'position': i,
                    'time': pos_time,
                    'nodes_per_sec': nodes_per_sec
                })
                total_nodes += rollouts
                
                if self.verbose:
                    print(f"  Position {i+1}: {nodes_per_sec:,.0f} NPS, move: {best_move}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Stop GPU monitoring
            gpu_stats = self.gpu_monitor.stop()
            
            # Calculate aggregate statistics
            avg_nps = total_nodes / total_time
            position_nps = [r['nodes_per_sec'] for r in position_results]
            
            result = {
                'rollouts': rollouts,
                'threads': threads,
                'batch_size': batch_size,
                'total_time': total_time,
                'total_nodes': total_nodes,
                'avg_nps': avg_nps,
                'min_nps': min(position_nps),
                'max_nps': max(position_nps),
                'positions': len(positions),
                'gpu_stats': gpu_stats
            }
            results.append(result)
            
            print(f"Average: {avg_nps:,.0f} NPS (min: {result['min_nps']:,.0f}, max: {result['max_nps']:,.0f})")
            if gpu_stats:
                print(f"GPU utilization: {gpu_stats['gpu_util_avg']:.1f}% avg, {gpu_stats['gpu_util_max']:.1f}% max")
                print(f"GPU memory: {gpu_stats['mem_util_avg']:.1f}% avg")
                print(f"Temperature: {gpu_stats['temperature_avg']:.1f}°C avg, {gpu_stats['temperature_max']:.1f}°C max")
        
        return results
    
    def get_test_positions(self):
        """Get a diverse set of test positions"""
        positions = []
        
        # Starting position
        positions.append(chess.Board())
        
        # Common opening positions
        openings = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # King's Knight
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",  # Italian
        ]
        
        for fen in openings:
            positions.append(chess.Board(fen))
        
        # Middle game positions
        middle_games = [
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 8",
            "r2q1rk1/1b2bppp/p1n1pn2/1p6/3PP3/1BP2N2/PP3PPP/RNBQR1K1 w - - 0 12",
        ]
        
        for fen in middle_games:
            positions.append(chess.Board(fen))
        
        # Endgame positions
        endgames = [
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Rook endgame
            "8/8/3kpp2/3p4/3P4/3KPP2/8/8 w - - 0 1",  # King and pawn
        ]
        
        for fen in endgames:
            positions.append(chess.Board(fen))
        
        return positions
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks and generate report"""
        print("\n" + "="*60)
        print("AlphaZero Chess Engine Benchmark - Windows/CUDA")
        print("="*60)
        
        # System info
        print("\nSystem Information:")
        print(f"Platform: {sys.platform}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        
        # GPU info
        device_name = torch.cuda.get_device_name(self.device_id)
        device_memory = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
        print(f"\nGPU Information:")
        print(f"Device: {device_name}")
        print(f"Memory: {device_memory:.1f} GB")
        print(f"Compute Capability: {torch.cuda.get_device_capability(self.device_id)}")
        
        # CPU info
        print(f"\nCPU Information:")
        print(f"Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"Frequency: {psutil.cpu_freq().current:.0f} MHz")
        
        # Load model
        self.load_model()
        
        # Benchmark neural network
        nn_results = self.benchmark_neural_network()
        
        # Get test positions
        positions = self.get_test_positions()
        print(f"\nUsing {len(positions)} test positions")
        
        # MCTS configurations to test
        mcts_configs = [
            {'rollouts': 100, 'threads': 16, 'batch_size': 128},
            {'rollouts': 1000, 'threads': 32, 'batch_size': 256},
            {'rollouts': 5000, 'threads': 64, 'batch_size': 512},
            {'rollouts': 10000, 'threads': 64, 'batch_size': 512},
            {'rollouts': 50000, 'threads': 64, 'batch_size': 512},
        ]
        
        # Run MCTS benchmarks
        mcts_results = self.benchmark_mcts(positions, mcts_configs)
        
        # Generate summary
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        
        # Find optimal batch size for NN
        best_nn = max(nn_results, key=lambda x: x['positions_per_sec'])
        print(f"\nOptimal NN batch size: {best_nn['batch_size']} "
              f"({best_nn['positions_per_sec']:,.0f} positions/sec)")
        
        # Best MCTS configuration
        best_mcts = max(mcts_results, key=lambda x: x['avg_nps'])
        print(f"\nBest MCTS performance: {best_mcts['avg_nps']:,.0f} NPS")
        print(f"  Configuration: {best_mcts['rollouts']} rollouts, "
              f"{best_mcts['threads']} threads, batch size {best_mcts['batch_size']}")
        
        # Comparison with baseline
        baseline_nps = 800  # Original performance
        speedup = best_mcts['avg_nps'] / baseline_nps
        print(f"\nSpeedup vs baseline: {speedup:.1f}x ({baseline_nps} → {best_mcts['avg_nps']:,.0f} NPS)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"benchmark_results_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'system': {
                'platform': sys.platform,
                'python': sys.version,
                'pytorch': torch.__version__,
                'cuda': torch.version.cuda,
                'gpu': device_name,
                'gpu_memory_gb': device_memory
            },
            'neural_network_results': nn_results,
            'mcts_results': mcts_results,
            'summary': {
                'best_nn_batch_size': best_nn['batch_size'],
                'best_nn_throughput': best_nn['positions_per_sec'],
                'best_mcts_nps': best_mcts['avg_nps'],
                'best_mcts_config': {
                    'rollouts': best_mcts['rollouts'],
                    'threads': best_mcts['threads'],
                    'batch_size': best_mcts['batch_size']
                },
                'speedup_vs_baseline': speedup
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results saved to: {report_file}")
        
        # Cleanup
        if self.mcts_engine:
            self.mcts_engine.stop()
        
        return report

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark AlphaZero Chess Engine - Windows/CUDA Optimized'
    )
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick benchmark with fewer iterations')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
    
    # Set up CUDA optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create and run benchmark
    benchmark = BenchmarkRunner(args.model, args.device, args.verbose)
    
    try:
        report = benchmark.run_comprehensive_benchmark()
        
        # Print key metric
        print(f"\n{'='*60}")
        print(f"KEY METRIC: {report['summary']['best_mcts_nps']:,.0f} nodes per second")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if benchmark.mcts_engine:
            benchmark.mcts_engine.stop()
        
        # Shutdown NVML
        if sys.platform == 'win32' and pynvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

if __name__ == '__main__':
    main()