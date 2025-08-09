"""
Simplified parallel neural network inference server for Windows compatibility.

This version uses a simpler architecture that works reliably on Windows:
- Single coordinator process that manages request routing
- Multiple inference worker processes
- Avoids complex object passing between processes
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import torch
import numpy as np
import time
import chess
import encoder
from collections import defaultdict
import hashlib
import queue
import os

from inference_server import InferenceRequest, InferenceResult


def inference_worker_process(worker_id, model_state, model_config, device_type,
                            request_queue, stop_event, batch_size=64, 
                            timeout_ms=150):
    """
    Worker process that handles inference requests.
    
    This function runs in a separate process and handles batched inference.
    """
    import AlphaZeroNetwork
    
    print(f"Inference worker {worker_id} starting...")
    
    # Create device
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # For multi-GPU systems, distribute workers across GPUs
        if torch.cuda.device_count() > 1:
            device = torch.device(f'cuda:{worker_id % torch.cuda.device_count()}')
    elif device_type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Inference worker {worker_id} using device: {device}")
    
    # Create and load model
    model = AlphaZeroNetwork.AlphaZeroNet(
        model_config['num_blocks'], 
        model_config['num_channels']
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Position cache
    position_cache = {}
    CACHE_MAX_SIZE = 50000
    
    # Statistics
    total_requests = 0
    total_batches = 0
    
    def get_position_hash(board):
        """Get hash for board position."""
        return hashlib.md5(board.fen().encode()).hexdigest()
    
    def encode_board_cached(board):
        """Encode board with caching."""
        board_hash = get_position_hash(board)
        
        if board_hash in position_cache:
            return position_cache[board_hash]
        else:
            position, mask = encoder.encodePositionForInference(board)
            position = torch.from_numpy(position)
            mask = torch.from_numpy(mask)
            
            # Cache if not full
            if len(position_cache) < CACHE_MAX_SIZE:
                position_cache[board_hash] = (position, mask)
                
            return position, mask
    
    def process_batch(batch_data):
        """Process a batch of requests."""
        if not batch_data:
            return
        
        # Deduplicate by position
        unique_boards = {}
        request_mapping = defaultdict(list)
        
        for req_id, board_fen, result_queue in batch_data:
            board = chess.Board(board_fen)
            board_hash = get_position_hash(board)
            
            if board_hash not in unique_boards:
                unique_boards[board_hash] = board
            request_mapping[board_hash].append((req_id, result_queue))
        
        # Prepare batch
        unique_count = len(unique_boards)
        inputs = torch.zeros((unique_count, 16, 8, 8), dtype=torch.float32)
        masks = torch.zeros((unique_count, 72, 8, 8), dtype=torch.float32)
        
        for i, board in enumerate(unique_boards.values()):
            position, mask = encode_board_cached(board)
            inputs[i] = position
            masks[i] = mask
        
        # Move to device
        inputs = inputs.to(device)
        masks = masks.to(device)
        
        # Convert to half precision if needed
        if next(model.parameters()).dtype == torch.float16:
            inputs = inputs.half()
            masks = masks.half()
        
        # Flatten masks
        masks_flat = masks.view(masks.shape[0], -1)
        
        # Run inference
        with torch.no_grad():
            values, policies = model(inputs, policyMask=masks_flat)
        
        # Process results
        values = values.cpu().numpy().reshape((unique_count,))
        policies = policies.cpu().numpy()
        
        # Send results back
        for i, (board_hash, board) in enumerate(unique_boards.items()):
            value = values[i]
            move_probs = encoder.decodePolicyOutput(board, policies[i])
            
            for req_id, result_queue in request_mapping[board_hash]:
                result = InferenceResult(req_id, value, move_probs)
                result_queue.put(result)
    
    # Main loop
    while not stop_event.is_set():
        batch_data = []
        deadline = time.time() + timeout_ms / 1000.0
        
        # Collect batch
        while len(batch_data) < batch_size and time.time() < deadline:
            timeout_remaining = max(0, deadline - time.time())
            
            try:
                if timeout_remaining > 0:
                    item_list = request_queue.get(timeout=timeout_remaining)
                    if item_list is not None and item_list is not []:  # None signals shutdown
                        batch_data.extend(item_list)
                else:
                    break
            except queue.Empty:
                break
        
        # Process batch
        if batch_data:
            process_batch(batch_data)
            total_requests += len(batch_data)
            total_batches += 1
    
    print(f"Inference worker {worker_id} stopped")


def coordinator_process(model_state, model_config, device_type, main_request_queue,
                       stop_event, num_workers=2, batch_size=64, timeout_ms=150):
    """
    Coordinator process that distributes requests to worker processes.
    """
    print(f"Starting coordinator with {num_workers} workers...")
    
    # Create queues for workers
    worker_queues = [mp.Queue() for _ in range(num_workers)]
    worker_processes = []
    
    total_requests = 0


    # Start worker processes
    for i in range(num_workers):
        p = Process(
            target=inference_worker_process,
            args=(i, model_state, model_config, device_type,
                  worker_queues[i], stop_event, batch_size, timeout_ms)
        )
        p.start()
        worker_processes.append(p)
    
    # Round-robin distribution
    next_worker = 0
    
    # Route requests to workers
    while not stop_event.is_set():
        try:
            request_tuples = []
            request_tuple_list = main_request_queue.get(timeout=0.1)
            worker_queue_conversion = []
            if len(request_tuple_list) > 0:
                for i in range(0, len(request_tuple_list)):
                    req_tuple = request_tuple_list[i]
                    request, result_queue = req_tuple
                    worker_queue_conversion.append((request.request_id, request.board_fen, result_queue))

                total_requests += len(worker_queue_conversion)
                worker_queues[next_worker].put(worker_queue_conversion)
                next_worker = (next_worker + 1) % num_workers
        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Coordinator error: {e}")
    
    # Shutdown workers
    for q in worker_queues:
        q.put(None)  # Signal shutdown
    
    # Wait for workers
    for p in worker_processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    print("Coordinator stopped")


def start_inference_server(model, device, main_queue, stop_event,
                                       num_servers=2, batch_size=64, timeout_ms=150):
    """
    Start parallel inference servers (simplified version for Windows).
    """
    # Get model info
    original_device = device
    model_state = model.cpu().state_dict()
    
    try:
        conv1_out_channels = model.conv1.out_channels
        num_blocks = len([m for m in model.modules() 
                        if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
        model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
    except:
        model_config = {'num_blocks': 20, 'num_channels': 256}
    
    # Determine device type
    if original_device.type == 'cuda':
        device_type = 'cuda'
    elif original_device.type == 'mps':
        device_type = 'mps'
    else:
        device_type = 'cpu'
    
    # Start coordinator process
    coordinator_process(model_state, model_config, device_type, main_queue,
                       stop_event, num_servers, batch_size, timeout_ms)


def start_inference_server_from_state(model_state, model_config, device_type,
                                                  main_queue, stop_event,
                                                  num_servers=2, batch_size=64, timeout_ms=150):
    """
    Start parallel inference servers from state dict (simplified version).
    """
    # Start coordinator process
    coordinator_process(model_state, model_config, device_type, main_queue,
                       stop_event, num_servers, batch_size, timeout_ms)