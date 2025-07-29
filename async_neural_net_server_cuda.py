import torch
import numpy as np
import threading
import queue
import time
import sys
from collections import namedtuple
from typing import List, Tuple, Optional
import encoder
from device_utils import get_optimal_device, optimize_for_device
import os

# Request and Response types
NNRequest = namedtuple('NNRequest', ['request_id', 'board', 'future'])
NNResponse = namedtuple('NNResponse', ['request_id', 'value', 'move_probabilities'])

class AsyncNeuralNetworkServerCUDA:
    """
    High-performance async neural network server optimized for NVIDIA GPUs.
    Specifically tuned for RTX 4080 and similar high-end consumer GPUs.
    """
    
    def __init__(self, neural_network, device=None, max_batch_size=512, 
                 max_wait_time=0.005, verbose=False):
        """
        Initialize the async neural network server for CUDA.
        
        Args:
            neural_network: The neural network model
            device: Device to run on (auto-detect if None)
            max_batch_size: Maximum batch size for GPU processing (512 optimal for RTX 4080)
            max_wait_time: Maximum time to wait for batch to fill (5ms for better batching)
            verbose: Whether to print performance statistics
        """
        self.neural_network = neural_network
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.verbose = verbose
        
        # Set up device - ensure CUDA is used
        if device is None:
            if torch.cuda.is_available():
                # Use primary GPU
                self.device = torch.device('cuda:0')
                device_name = torch.cuda.get_device_name(0)
                device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if verbose:
                    print(f"AsyncNeuralNetworkServerCUDA using: {device_name}")
                    print(f"GPU Memory: {device_memory:.1f}GB")
                    
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set CUDA memory allocation strategy for better performance
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                
            else:
                raise RuntimeError("CUDA not available. This optimized version requires an NVIDIA GPU.")
        else:
            self.device = device
            
        # Move model to device and optimize
        self.neural_network = self.neural_network.to(self.device)
        self.neural_network.eval()
        
        # Compile model with torch.compile for additional speedup (PyTorch 2.0+)
        # Note: torch.compile requires Triton which has limited Windows support
        if hasattr(torch, 'compile') and sys.platform != 'win32':
            try:
                self.neural_network = torch.compile(self.neural_network, mode='reduce-overhead')
                if verbose:
                    print("Model compiled with torch.compile for additional speedup")
            except:
                if verbose:
                    print("torch.compile not available or failed, using eager mode")
        elif verbose and sys.platform == 'win32':
            print("Skipping torch.compile on Windows (Triton not fully supported)")
        
        # Disable gradients for inference
        for param in self.neural_network.parameters():
            param.requires_grad = False
        
        # Use larger queue for high-throughput scenarios
        self.request_queue = queue.Queue(maxsize=max_batch_size * 32)
        self.running = False
        self.server_thread = None
        
        # Pre-allocate tensors for batching to reduce allocation overhead
        self.position_buffer = torch.zeros((max_batch_size, 16, 8, 8), 
                                         dtype=torch.float32, device=self.device)
        self.mask_buffer = torch.zeros((max_batch_size, 72 * 8 * 8), 
                                     dtype=torch.float32, device=self.device)
        
        # Performance monitoring
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.total_wait_time = 0
        self.total_inference_time = 0
        
        # CUDA stream for async operations
        self.cuda_stream = torch.cuda.Stream()
        
    def start(self):
        """Start the async server thread."""
        if self.running:
            return
            
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        if self.verbose:
            print("AsyncNeuralNetworkServerCUDA started")
            
    def stop(self):
        """Stop the async server thread."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            
        if self.verbose:
            print("AsyncNeuralNetworkServerCUDA stopped")
            self._print_statistics()
            
    def _print_statistics(self):
        """Print performance statistics."""
        if self.total_batches == 0:
            return
            
        avg_batch_size = self.total_batch_size / self.total_batches
        avg_wait_time = self.total_wait_time / self.total_batches * 1000  # ms
        avg_inference_time = self.total_inference_time / self.total_batches * 1000  # ms
        throughput = self.total_requests / (self.total_wait_time + self.total_inference_time)
        
        print(f"\nAsyncNeuralNetworkServerCUDA Statistics:")
        print(f"  Total requests processed: {self.total_requests}")
        print(f"  Total batches: {self.total_batches}")
        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Average wait time: {avg_wait_time:.2f}ms")
        print(f"  Average inference time: {avg_inference_time:.2f}ms")
        print(f"  GPU utilization: {avg_batch_size / self.max_batch_size * 100:.1f}%")
        print(f"  Throughput: {throughput:.0f} requests/sec")
        
    def evaluate_async(self, board) -> 'Future':
        """
        Asynchronously evaluate a board position.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Future object that will contain (value, move_probabilities)
        """
        future = FutureCUDA()
        request_id = id(future)
        request = NNRequest(request_id, board, future)
        
        try:
            self.request_queue.put(request, timeout=0.1)
        except queue.Full:
            # Queue is full, process synchronously as fallback
            if self.verbose:
                print("Warning: Request queue full, falling back to sync evaluation")
            with torch.no_grad():
                value, move_probs = encoder.callNeuralNetwork(board, self.neural_network)
            future._set_result(value, move_probs)
            
        return future
        
    def _server_loop(self):
        """Main server loop that processes batches of requests."""
        while self.running:
            batch = []
            batch_start_time = time.time()
            
            # Collect requests up to max_batch_size or max_wait_time
            deadline = time.time() + self.max_wait_time
            
            while len(batch) < self.max_batch_size and time.time() < deadline:
                timeout = max(0.0001, deadline - time.time())
                try:
                    request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                    
                    # If we have a good batch size, process immediately
                    if len(batch) >= self.max_batch_size * 0.8:  # 80% full
                        break
                except queue.Empty:
                    break
                    
            if not batch:
                continue
                
            # Process batch
            wait_time = time.time() - batch_start_time
            inference_start = time.time()
            
            try:
                # Use CUDA stream for async processing
                with torch.cuda.stream(self.cuda_stream):
                    # Prepare batch data using pre-allocated buffers
                    batch_size = len(batch)
                    
                    # Fill buffers with batch data
                    for i, request in enumerate(batch):
                        position, mask = encoder.encodePositionForInference(request.board)
                        self.position_buffer[i] = torch.from_numpy(position)
                        # Flatten mask to match expected shape
                        self.mask_buffer[i] = torch.from_numpy(mask).flatten()
                    
                    # Slice buffers to actual batch size
                    positions = self.position_buffer[:batch_size]
                    masks = self.mask_buffer[:batch_size]
                    
                    # Batch evaluate
                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):  # Use automatic mixed precision
                            values, policies = self.neural_network(positions, policyMask=masks)
                    
                    # Convert to numpy (async transfer)
                    values_cpu = values.cpu().numpy()
                    policies_cpu = policies.cpu().numpy()
                
                # Synchronize stream
                self.cuda_stream.synchronize()
                
                # Process results
                for i, request in enumerate(batch):
                    value = values_cpu[i, 0]
                    policy = policies_cpu[i]
                    move_probs = encoder.decodePolicyOutput(request.board, policy)
                    request.future._set_result(value, move_probs)
                    
                # Update statistics
                inference_time = time.time() - inference_start
                self.total_requests += len(batch)
                self.total_batches += 1
                self.total_batch_size += len(batch)
                self.total_wait_time += wait_time
                self.total_inference_time += inference_time
                
                if self.verbose and self.total_batches % 100 == 0:
                    avg_batch_size = self.total_batch_size / self.total_batches
                    throughput = self.total_requests / (self.total_wait_time + self.total_inference_time)
                    print(f"Batch {self.total_batches}: size={len(batch)}, "
                          f"avg_size={avg_batch_size:.1f}, "
                          f"wait={wait_time*1000:.1f}ms, "
                          f"inference={inference_time*1000:.1f}ms, "
                          f"throughput={throughput:.0f} req/s")
                    
            except Exception as e:
                print(f"Error in neural network server: {e}")
                # Set error state for all requests in batch
                for request in batch:
                    request.future._set_error(e)


class FutureCUDA:
    """
    Future object optimized for CUDA operations.
    """
    
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._move_probabilities = None
        self._exception = None
        
    def result(self, timeout=None) -> Tuple[float, np.ndarray]:
        """
        Get the result of the neural network evaluation.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            Tuple of (value, move_probabilities)
        """
        if not self._event.wait(timeout):
            raise TimeoutError("Neural network evaluation timed out")
            
        if self._exception:
            raise self._exception
            
        return self._value, self._move_probabilities
        
    def done(self) -> bool:
        """Check if the result is ready."""
        return self._event.is_set()
        
    def _set_result(self, value: float, move_probabilities: np.ndarray):
        """Internal method to set the result."""
        self._value = value
        self._move_probabilities = move_probabilities
        self._event.set()
        
    def _set_error(self, exception: Exception):
        """Internal method to set an error."""
        self._exception = exception
        self._event.set()


class NeuralNetworkPoolCUDA:
    """
    Pool of neural network servers for multi-GPU setups.
    Optimized for systems with multiple RTX 4080s or similar.
    """
    
    def __init__(self, neural_network, num_gpus=None, **kwargs):
        """
        Initialize pool of neural network servers.
        
        Args:
            neural_network: The neural network model
            num_gpus: Number of GPUs to use (auto-detect if None)
            **kwargs: Additional arguments passed to AsyncNeuralNetworkServerCUDA
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
            
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")
            
        self.servers = []
        self.current_server = 0
        
        print(f"Initializing NeuralNetworkPoolCUDA with {num_gpus} GPUs")
        
        for gpu_id in range(num_gpus):
            device = torch.device(f'cuda:{gpu_id}')
            
            # Create a copy of the model for each GPU
            model_copy = type(neural_network)(
                neural_network.num_blocks,
                neural_network.num_channels
            )
            model_copy.load_state_dict(neural_network.state_dict())
            
            # Create server with GPU-specific settings
            server = AsyncNeuralNetworkServerCUDA(
                model_copy, 
                device=device, 
                **kwargs
            )
            self.servers.append(server)
            
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            
    def start(self):
        """Start all servers."""
        for server in self.servers:
            server.start()
            
    def stop(self):
        """Stop all servers."""
        for server in self.servers:
            server.stop()
            
    def evaluate_async(self, board) -> FutureCUDA:
        """
        Evaluate a board position using round-robin distribution.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Future object that will contain (value, move_probabilities)
        """
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        return server.evaluate_async(board)