"""
Dedicated neural network inference server for multi-process MCTS.

This process handles all neural network evaluations, batching requests
from multiple worker processes for efficient GPU utilization.
"""

import multiprocessing as mp
from multiprocessing import Queue
import torch
import numpy as np
import time
import chess
import encoder
from collections import defaultdict
import hashlib

class InferenceRequest:
    """Request for neural network evaluation."""
    
    def __init__(self, request_id, board_fen, worker_id=None):
        self.request_id = request_id
        self.board_fen = board_fen
        self.worker_id = worker_id  # To identify which worker sent the request
        
    def to_board(self):
        """Convert FEN to chess board."""
        return chess.Board(self.board_fen)


class InferenceResult:
    """Result from neural network evaluation."""
    
    def __init__(self, request_id, value, move_probabilities):
        self.request_id = request_id
        self.value = value
        self.move_probabilities = move_probabilities


class InferenceServer:
    """Server process for neural network inference."""
    
    def __init__(self, model, device, batch_size=64, timeout_ms=150):
        """
        Initialize inference server.
        
        Args:
            model: Neural network model
            device: Torch device (cuda/MPS/cpu)
            batch_size: Maximum batch size
            timeout_ms: Timeout in milliseconds to wait for batch
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        
        # Move model to device and optimize
        self.model = self.model.to(device)
        self.model.eval()
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Position cache
        self.position_cache = {}
        self.CACHE_MAX_SIZE = 100000
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_time = 0.0
        
    def get_position_hash(self, board):
        """Get hash for board position."""
        return hashlib.md5(board.fen().encode()).hexdigest()
        
    def encode_board(self, board):
        """Encode board for neural network."""
        board_hash = self.get_position_hash(board)
        
        if board_hash in self.position_cache:
            position, mask = self.position_cache[board_hash]
            return position.clone(), mask.clone()
        else:
            position, mask = encoder.encodePositionForInference(board)
            position = torch.from_numpy(position)
            mask = torch.from_numpy(mask)
            
            # Cache if not full
            if len(self.position_cache) < self.CACHE_MAX_SIZE:
                self.position_cache[board_hash] = (position.clone(), mask.clone())
                
            return position, mask
    
    def process_batch(self, request_tuples):
        """Process a batch of inference requests."""
        if not request_tuples:
            return
            
        start_time = time.time()
        batch_size = len(request_tuples)
        
        # Deduplicate requests by board position
        unique_boards = {}
        request_mapping = {}
        
        for req, result_queue in request_tuples:
            board = req.to_board()
            board_hash = self.get_position_hash(board)
            
            if board_hash not in unique_boards:
                unique_boards[board_hash] = board
                request_mapping[board_hash] = []
            request_mapping[board_hash].append((req, result_queue))
        
        # Prepare batch tensors
        unique_count = len(unique_boards)
        inputs = torch.zeros((unique_count, 16, 8, 8), dtype=torch.float32)
        masks = torch.zeros((unique_count, 72, 8, 8), dtype=torch.float32)
        board_list = []
        
        for i, (board_hash, board) in enumerate(unique_boards.items()):
            position, mask = self.encode_board(board)
            inputs[i] = position
            masks[i] = mask
            board_list.append(board)
        
        # Move to device
        inputs = inputs.to(self.device)
        masks = masks.to(self.device)
        
        # Convert to half precision if model is FP16
        if next(self.model.parameters()).dtype == torch.float16:
            inputs = inputs.half()
            masks = masks.half()
        
        # Flatten masks
        masks_flat = masks.view(masks.shape[0], -1)
        
        #print(inputs)

        # Run inference
        with torch.no_grad():
            values, policies = self.model(inputs, policyMask=masks_flat)
        
        # Process results - use flatten() instead of reshape to handle (batch_size, 1) shape
        values = values.cpu().numpy().flatten()
        policies = policies.cpu().numpy()

        move_probabilities = np.zeros( ( len(unique_boards), 200 ), dtype=np.float32 )
        # Distribute results to all requests
        for i, (board_hash, board) in enumerate(unique_boards.items()):
            value = values[i]
            move_probabilities_tmp = encoder.decodePolicyOutput( board, policies[ i ] )
            move_probabilities[ i, : move_probabilities_tmp.shape[0] ] = move_probabilities_tmp
            
            # Send to all requests for this position
            for req, result_queue in request_mapping[board_hash]:
                result = InferenceResult(req.request_id, value, move_probabilities[i])
                result_queue.put(result)
        
        # Update statistics
        elapsed = time.time() - start_time
        self.total_requests += batch_size
        self.total_batches += 1
        self.total_time += elapsed
        
    def run(self, request_queue, stop_event):
        """
        Main server loop.
        
        Args:
            request_queue: Queue for incoming requests (tuples of (request, result_queue))
            stop_event: Event to signal shutdown
        """
        print(f"Inference server started on {self.device}")
        
        while not stop_event.is_set():
            request_tuples = []
            deadline = time.time() + self.timeout_ms / 1000.0
            
            # Collect requests up to batch size or timeout
            while len(request_tuples) < self.batch_size and time.time() < deadline:
                timeout_remaining = max(0, deadline - time.time())
                
                try:
                    if timeout_remaining > 0:
                        req_tuple_list = request_queue.get(timeout=timeout_remaining)
                        request_tuples.extend(req_tuple_list)
                    else:
                        break
                except:
                    # Timeout or empty queue
                    break
            
            # Process batch if we have requests
            if request_tuples:
                self.process_batch(request_tuples)
                #print(f"Processed batch of {len(request_tuples)} requests with size {self.batch_size}")
                
            # Print statistics periodically
            if self.total_batches > 0 and self.total_batches % 1000 == 0:
                avg_batch_size = self.total_requests / self.total_batches
                avg_time = self.total_time / self.total_batches
                throughput = self.total_requests / self.total_time if self.total_time > 0 else 0
                print(f"Inference stats: {self.total_requests} requests, "
                      f"{self.total_batches} batches, "
                      f"avg batch size: {avg_batch_size:.1f}, "
                      f"maximum batch size: {self.batch_size}, "
                      f"timeout: {self.timeout_ms}ms, "
                      f"avg time: {avg_time*1000:.1f}ms, "
                      f"throughput: {throughput:.0f} req/s")
        
        print("Inference server stopped")
        
    def clear_cache(self):
        """Clear position cache."""
        self.position_cache.clear()


def start_inference_server(model, device, request_queue, stop_event, 
                         batch_size=22, timeout_ms=30):
    """
    Start inference server in current process.
    
    Args:
        model: Neural network model
        device: Torch device
        request_queue: Queue for requests
        stop_event: Event to signal shutdown
        batch_size: Maximum batch size
        timeout_ms: Timeout to wait for batch
    """
    server = InferenceServer(model, device, batch_size, timeout_ms)
    server.run(request_queue, stop_event)


def start_inference_server_from_state(model_state, model_config, device_type, 
                                    request_queue, stop_event,
                                    batch_size=22, timeout_ms=30):
    """
    Start inference server from model state dict (for MPS compatibility).
    
    Args:
        model_state: State dict of the model
        model_config: Configuration for model creation
        device_type: Device type as string ('cpu', 'cuda', etc)
        request_queue: Queue for requests
        stop_event: Event to signal shutdown
        batch_size: Maximum batch size
        timeout_ms: Timeout to wait for batch
    """
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
    server = InferenceServer(model, device, batch_size, timeout_ms)
    server.run(request_queue, stop_event)