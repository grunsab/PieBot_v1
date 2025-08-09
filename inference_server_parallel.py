"""
Parallel neural network inference server with multiple server processes.

This implementation addresses the single-process bottleneck by running multiple
inference servers in parallel, each handling a subset of requests. It includes:
- Multiple inference server processes with their own model copies
- Load balancing across servers (round-robin or worker affinity)
- Parallel preprocessing pipeline for CPU-bound operations
- Optimized batch accumulation and processing
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
import torch
import numpy as np
import time
import chess
import encoder
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread
import os

from inference_server import InferenceRequest, InferenceResult


class ParallelInferenceServer:
    """Single inference server process with optimized batch processing."""
    
    def __init__(self, server_id, model, device, batch_size=64, timeout_ms=150, 
                 use_parallel_encoding=True, encoding_threads=2):
        """
        Initialize parallel inference server.
        
        Args:
            server_id: Unique identifier for this server
            model: Neural network model
            device: Torch device (cuda/MPS/cpu)
            batch_size: Maximum batch size
            timeout_ms: Timeout in milliseconds to wait for batch
            use_parallel_encoding: Use parallel encoding for CPU operations
            encoding_threads: Number of threads for parallel encoding
        """
        self.server_id = server_id
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.use_parallel_encoding = use_parallel_encoding
        self.encoding_threads = encoding_threads
        
        # Move model to device and optimize
        self.model = self.model.to(device)
        self.model.eval()
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Position cache
        self.position_cache = {}
        self.CACHE_MAX_SIZE = 50000  # Smaller cache per server
        
        # Thread pool for parallel encoding
        if self.use_parallel_encoding:
            self.encoding_executor = ThreadPoolExecutor(max_workers=encoding_threads)
        
        # Pre-allocate tensors for better memory management
        self.preallocated_inputs = torch.zeros((batch_size, 16, 8, 8), 
                                              dtype=torch.float32, device=device)
        self.preallocated_masks = torch.zeros((batch_size, 72, 8, 8), 
                                             dtype=torch.float32, device=device)
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_time = 0.0
        
    def get_position_hash(self, board):
        """Get hash for board position."""
        return hashlib.md5(board.fen().encode()).hexdigest()
        
    def encode_board_cached(self, board):
        """Encode board with caching."""
        board_hash = self.get_position_hash(board)
        
        if board_hash in self.position_cache:
            return self.position_cache[board_hash]
        else:
            position, mask = encoder.encodePositionForInference(board)
            position = torch.from_numpy(position)
            mask = torch.from_numpy(mask)
            
            # Cache if not full
            if len(self.position_cache) < self.CACHE_MAX_SIZE:
                self.position_cache[board_hash] = (position, mask)
                
            return position, mask
    
    def parallel_encode_boards(self, boards):
        """Encode multiple boards in parallel."""
        if self.use_parallel_encoding:
            futures = [self.encoding_executor.submit(self.encode_board_cached, board) 
                      for board in boards]
            return [future.result() for future in futures]
        else:
            return [self.encode_board_cached(board) for board in boards]
    
    def process_batch_optimized(self, request_tuples):
        """Process a batch with optimizations."""
        if not request_tuples:
            return
            
        start_time = time.perf_counter()
        batch_size = len(request_tuples)
        
        # Deduplicate requests by board position
        unique_boards = {}
        request_mapping = defaultdict(list)
        
        for req, result_queue in request_tuples:
            board = req.to_board()
            board_hash = self.get_position_hash(board)
            
            if board_hash not in unique_boards:
                unique_boards[board_hash] = board
            request_mapping[board_hash].append((req, result_queue))
        
        # Parallel encoding
        unique_count = len(unique_boards)
        board_list = list(unique_boards.values())
        encoded_data = self.parallel_encode_boards(board_list)
        
        # Use pre-allocated tensors
        actual_batch_size = min(unique_count, self.batch_size)
        inputs = self.preallocated_inputs[:actual_batch_size]
        masks = self.preallocated_masks[:actual_batch_size]
        
        # Fill tensors
        for i, (position, mask) in enumerate(encoded_data[:actual_batch_size]):
            inputs[i].copy_(position)
            masks[i].copy_(mask)
        
        # Convert to half precision if needed
        if next(self.model.parameters()).dtype == torch.float16:
            inputs = inputs.half()
            masks = masks.half()
        
        # Flatten masks
        masks_flat = masks.view(masks.shape[0], -1)
        
        # Run inference
        with torch.no_grad():
            values, policies = self.model(inputs, policyMask=masks_flat)
        
        # Process results
        values = values.cpu().numpy().reshape((actual_batch_size,))
        policies = policies.cpu().numpy()
        
        # Distribute results
        for i, (board_hash, board) in enumerate(list(unique_boards.items())[:actual_batch_size]):
            value = values[i]
            move_probs = encoder.decodePolicyOutput(board, policies[i])
            
            # Send to all requests for this position
            for req, result_queue in request_mapping[board_hash]:
                result = InferenceResult(req.request_id, value, move_probs)
                result_queue.put(result)
        
        # Update statistics
        elapsed = time.perf_counter() - start_time
        self.total_requests += batch_size
        self.total_batches += 1
        self.total_time += elapsed
        
    def run(self, request_queue, stop_event):
        """
        Main server loop.
        
        Args:
            request_queue: Queue for incoming requests
            stop_event: Event to signal shutdown
        """
        print(f"Inference server {self.server_id} started on {self.device}")
        
        while not stop_event.is_set():
            request_tuples = []
            deadline = time.time() + self.timeout_ms / 1000.0
            
            # Collect requests up to batch size or timeout
            while len(request_tuples) < self.batch_size and time.time() < deadline:
                timeout_remaining = max(0, deadline - time.time())
                
                try:
                    if timeout_remaining > 0:
                        req_tuple = request_queue.get(timeout=timeout_remaining)
                        request_tuples.append(req_tuple)
                    else:
                        break
                except:
                    break
            
            # Process batch if we have requests
            if request_tuples:
                self.process_batch_optimized(request_tuples)
                
            # Print statistics periodically
            if self.total_batches > 0 and self.total_batches % 500 == 0:
                avg_batch_size = self.total_requests / self.total_batches
                avg_time = self.total_time / self.total_batches
                throughput = self.total_requests / self.total_time if self.total_time > 0 else 0
                print(f"Server {self.server_id} stats: {self.total_requests} requests, "
                      f"avg batch: {avg_batch_size:.1f}, "
                      f"avg time: {avg_time*1000:.1f}ms, "
                      f"throughput: {throughput:.0f} req/s")
        
        # Cleanup
        if self.use_parallel_encoding:
            self.encoding_executor.shutdown(wait=False)
        
        print(f"Inference server {self.server_id} stopped")


class ParallelInferenceCoordinator:
    """Coordinator for multiple parallel inference servers."""
    
    def __init__(self, model, num_servers=2, batch_size=64, timeout_ms=150,
                 use_parallel_encoding=True, encoding_threads=2,
                 load_balance_mode='round_robin'):
        """
        Initialize parallel inference coordinator.
        
        Args:
            model: Neural network model
            num_servers: Number of parallel inference servers
            batch_size: Maximum batch size per server
            timeout_ms: Timeout for batch accumulation
            use_parallel_encoding: Use parallel encoding in each server
            encoding_threads: Threads per server for encoding
            load_balance_mode: 'round_robin' or 'worker_affinity'
        """
        self.model = model
        self.num_servers = num_servers
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.use_parallel_encoding = use_parallel_encoding
        self.encoding_threads = encoding_threads
        self.load_balance_mode = load_balance_mode
        
        # Setup multiprocessing
        self.manager = Manager()
        self.server_queues = [self.manager.Queue() for _ in range(num_servers)]
        self.stop_event = self.manager.Event()
        
        # For round-robin load balancing
        self.next_server_idx = 0
        self.server_idx_lock = mp.Lock()
        
        # Server processes
        self.server_processes = []
        
    def get_next_server_queue(self, worker_id=None):
        """Get the queue for the next server based on load balancing mode."""
        if self.load_balance_mode == 'worker_affinity' and worker_id is not None:
            # Assign workers to servers based on worker ID
            return self.server_queues[worker_id % self.num_servers]
        else:
            # Round-robin distribution
            with self.server_idx_lock:
                queue = self.server_queues[self.next_server_idx]
                self.next_server_idx = (self.next_server_idx + 1) % self.num_servers
                return queue
    
    def start_servers(self):
        """Start all inference server processes."""
        original_device = next(self.model.parameters()).device
        
        # Move model to CPU for serialization to avoid CUDA tensor issues
        model_state = self.model.cpu().state_dict()
        
        try:
            conv1_out_channels = self.model.conv1.out_channels
            num_blocks = len([m for m in self.model.modules() 
                            if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
            model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
        except:
            model_config = {'num_blocks': 20, 'num_channels': 256}
        
        # Determine target device type
        if original_device.type == 'cuda':
            device_type = 'cuda'
        elif original_device.type == 'mps':
            device_type = 'mps'
        else:
            device_type = 'cpu'
        
        for i in range(self.num_servers):
            process = Process(
                target=ParallelInferenceCoordinator._start_server_from_state,
                args=(i, model_state, model_config, device_type, 
                      self.server_queues[i], self.stop_event,
                      self.batch_size, self.timeout_ms,
                      self.use_parallel_encoding, self.encoding_threads)
            )
            
            process.start()
            self.server_processes.append(process)
        
        # Move model back to original device
        self.model = self.model.to(original_device)
        
        print(f"Started {self.num_servers} parallel inference servers")
    
    
    @staticmethod
    def _start_server_from_state(server_id, model_state, model_config, 
                                device_type, request_queue, stop_event,
                                batch_size, timeout_ms, use_parallel_encoding,
                                encoding_threads):
        """Start a single inference server from state dict."""
        import AlphaZeroNetwork
        
        # Create device
        if device_type == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # Create and load model
        model = AlphaZeroNetwork.AlphaZeroNet(
            model_config['num_blocks'], 
            model_config['num_channels']
        )
        model.load_state_dict(model_state)
        
        # Start server
        server = ParallelInferenceServer(
            server_id, model, device,
            batch_size, timeout_ms,
            use_parallel_encoding, encoding_threads
        )
        server.run(request_queue, stop_event)
    
    def process_request(self, request, result_queue, worker_id=None):
        """
        Process an inference request.
        
        Args:
            request: InferenceRequest object
            result_queue: Queue to put result in
            worker_id: Optional worker ID for affinity-based load balancing
        """
        server_queue = self.get_next_server_queue(worker_id)
        server_queue.put((request, result_queue))
    
    def cleanup(self):
        """Clean up all server processes."""
        self.stop_event.set()
        time.sleep(0.1)
        
        # Empty queues
        for queue in self.server_queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
        
        # Terminate processes
        for process in self.server_processes:
            if process.is_alive():
                process.terminate()
            process.join()
        
        print("All inference servers stopped")


# Compatibility functions for drop-in replacement
def start_parallel_inference_servers(model, device, coordinator_queue, stop_event,
                                    num_servers=2, batch_size=64, timeout_ms=150):
    """
    Start parallel inference servers (drop-in replacement for start_inference_server).
    
    This function creates a coordinator that manages multiple inference servers
    and routes requests from the coordinator_queue to the appropriate server.
    """
    coordinator = ParallelInferenceCoordinator(
        model, num_servers=num_servers, 
        batch_size=batch_size, timeout_ms=timeout_ms,
        use_parallel_encoding=True, encoding_threads=2,
        load_balance_mode='round_robin'
    )
    
    # Start server processes
    coordinator.start_servers()
    
    # Route requests from coordinator queue to server queues
    def route_requests():
        while not stop_event.is_set():
            try:
                req_tuple = coordinator_queue.get(timeout=0.1)
                if req_tuple:
                    request, result_queue = req_tuple
                    worker_id = request.worker_id if hasattr(request, 'worker_id') else None
                    coordinator.process_request(request, result_queue, worker_id)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error routing request: {e}")
    
    # Run router in a thread
    router_thread = Thread(target=route_requests)
    router_thread.start()
    
    # Wait for stop signal
    while not stop_event.is_set():
        time.sleep(0.1)
    
    # Cleanup
    coordinator.cleanup()
    router_thread.join()


def start_parallel_inference_servers_from_state(model_state, model_config, device_type,
                                               coordinator_queue, stop_event,
                                               num_servers=2, batch_size=64, timeout_ms=150):
    """
    Start parallel inference servers from state dict.
    
    This function creates a coordinator with the model state dict and starts
    multiple inference server processes, each loading the model independently.
    """
    import AlphaZeroNetwork
    
    # Create a dummy model for the coordinator (will be serialized anyway)
    model = AlphaZeroNetwork.AlphaZeroNet(
        model_config['num_blocks'], 
        model_config['num_channels']
    )
    model.load_state_dict(model_state)
    
    # Determine device for the coordinator (CPU is safe for serialization)
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Create and start coordinator
    coordinator = ParallelInferenceCoordinator(
        model, num_servers=num_servers,
        batch_size=batch_size, timeout_ms=timeout_ms,
        use_parallel_encoding=True, encoding_threads=2,
        load_balance_mode='round_robin'
    )
    
    # Start server processes
    coordinator.start_servers()
    
    # Route requests from coordinator queue to server queues
    def route_requests():
        while not stop_event.is_set():
            try:
                req_tuple = coordinator_queue.get(timeout=0.1)
                if req_tuple:
                    request, result_queue = req_tuple
                    worker_id = request.worker_id if hasattr(request, 'worker_id') else None
                    coordinator.process_request(request, result_queue, worker_id)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error routing request: {e}")
    
    # Run router in a thread
    from threading import Thread
    router_thread = Thread(target=route_requests)
    router_thread.start()
    
    # Wait for stop signal
    while not stop_event.is_set():
        time.sleep(0.1)
    
    # Cleanup
    coordinator.cleanup()
    router_thread.join()