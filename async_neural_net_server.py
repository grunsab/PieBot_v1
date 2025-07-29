import torch
import numpy as np
import threading
import queue
import time
from collections import namedtuple
from typing import List, Tuple, Optional
import encoder
from device_utils import get_optimal_device, optimize_for_device

# Request and Response types
NNRequest = namedtuple('NNRequest', ['request_id', 'board', 'future'])
NNResponse = namedtuple('NNResponse', ['request_id', 'value', 'move_probabilities'])

class AsyncNeuralNetworkServer:
    """
    Asynchronous neural network server that continuously batches inference requests
    for optimal GPU utilization. This follows the AlphaGo Zero architecture pattern.
    """
    
    def __init__(self, neural_network, device=None, max_batch_size=256, 
                 max_wait_time=0.005, verbose=False):
        """
        Initialize the async neural network server.
        
        Args:
            neural_network: The neural network model
            device: Device to run on (auto-detect if None)
            max_batch_size: Maximum batch size for GPU processing
            max_wait_time: Maximum time to wait for batch to fill (seconds)
            verbose: Whether to print performance statistics
        """
        self.neural_network = neural_network
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.verbose = verbose
        
        # Set up device
        if device is None:
            self.device, device_str = get_optimal_device()
            if verbose:
                print(f"AsyncNeuralNetworkServer using device: {device_str}")
        else:
            self.device = device
            
        # Move model to device and optimize
        self.neural_network = optimize_for_device(self.neural_network, self.device)
        self.neural_network.eval()
        
        # Disable gradients for inference
        for param in self.neural_network.parameters():
            param.requires_grad = False
        
        # Request queue and processing thread
        self.request_queue = queue.Queue(maxsize=max_batch_size * 64)  # Larger queue for heavy load
        self.running = False
        self.server_thread = None
        
        # Performance monitoring
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.total_wait_time = 0
        self.total_inference_time = 0
        
    def start(self):
        """Start the async server thread."""
        if self.running:
            return
            
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        if self.verbose:
            print("AsyncNeuralNetworkServer started")
            
    def stop(self):
        """Stop the async server thread."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            
        if self.verbose:
            print("AsyncNeuralNetworkServer stopped")
            self._print_statistics()
            
    def _print_statistics(self):
        """Print performance statistics."""
        if self.total_batches == 0:
            return
            
        avg_batch_size = self.total_batch_size / self.total_batches
        avg_wait_time = self.total_wait_time / self.total_batches * 1000  # ms
        avg_inference_time = self.total_inference_time / self.total_batches * 1000  # ms
        
        print(f"\nAsyncNeuralNetworkServer Statistics:")
        print(f"  Total requests processed: {self.total_requests}")
        print(f"  Total batches: {self.total_batches}")
        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Average wait time: {avg_wait_time:.2f}ms")
        print(f"  Average inference time: {avg_inference_time:.2f}ms")
        print(f"  GPU utilization: {avg_batch_size / self.max_batch_size * 100:.1f}%")
        
    def evaluate_async(self, board) -> 'Future':
        """
        Asynchronously evaluate a board position.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Future object that will contain (value, move_probabilities)
        """
        future = Future()
        request_id = id(future)
        request = NNRequest(request_id, board, future)
        
        try:
            self.request_queue.put(request, timeout=0.1)
        except queue.Full:
            # Queue is full, process synchronously as fallback
            if self.verbose:
                print("Warning: Request queue full, falling back to sync evaluation")
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
                except queue.Empty:
                    break
                    
            if not batch:
                continue
                
            # Process batch
            wait_time = time.time() - batch_start_time
            inference_start = time.time()
            
            try:
                # Extract boards from requests
                boards = [req.board for req in batch]
                
                # Batch evaluate
                with torch.no_grad():
                    values, move_probabilities = encoder.callNeuralNetworkBatched(
                        boards, self.neural_network
                    )
                
                # Send results back through futures
                for i, request in enumerate(batch):
                    request.future._set_result(values[i], move_probabilities[i])
                    
                # Update statistics
                inference_time = time.time() - inference_start
                self.total_requests += len(batch)
                self.total_batches += 1
                self.total_batch_size += len(batch)
                self.total_wait_time += wait_time
                self.total_inference_time += inference_time
                
                if self.verbose and self.total_batches % 100 == 0:
                    avg_batch_size = self.total_batch_size / self.total_batches
                    print(f"Batch {self.total_batches}: size={len(batch)}, "
                          f"avg_size={avg_batch_size:.1f}, "
                          f"wait={wait_time*1000:.1f}ms, "
                          f"inference={inference_time*1000:.1f}ms")
                    
            except Exception as e:
                print(f"Error in neural network server: {e}")
                # Set error state for all requests in batch
                for request in batch:
                    request.future._set_error(e)


class Future:
    """
    Future object for async neural network results.
    Similar to concurrent.futures.Future but simplified.
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


class NeuralNetworkPool:
    """
    Pool of neural network servers for multi-GPU setups.
    Distributes requests across multiple GPUs for maximum throughput.
    """
    
    def __init__(self, neural_network, num_gpus=None, **kwargs):
        """
        Initialize pool of neural network servers.
        
        Args:
            neural_network: The neural network model
            num_gpus: Number of GPUs to use (auto-detect if None)
            **kwargs: Additional arguments passed to AsyncNeuralNetworkServer
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            
        self.servers = []
        self.current_server = 0
        
        for gpu_id in range(num_gpus):
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cpu')
                
            # Create a copy of the model for each GPU
            model_copy = type(neural_network)(
                neural_network.num_blocks,
                neural_network.num_channels
            )
            model_copy.load_state_dict(neural_network.state_dict())
            
            server = AsyncNeuralNetworkServer(model_copy, device=device, **kwargs)
            self.servers.append(server)
            
    def start(self):
        """Start all servers."""
        for server in self.servers:
            server.start()
            
    def stop(self):
        """Stop all servers."""
        for server in self.servers:
            server.stop()
            
    def evaluate_async(self, board) -> Future:
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