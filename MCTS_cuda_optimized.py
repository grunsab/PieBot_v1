"""
MCTS implementation with CUDA acceleration and C++ optimizations.
Designed for maximum performance on Windows CUDA systems.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
import math
from collections import deque, defaultdict
import hashlib
from functools import lru_cache
import time
from typing import List, Tuple, Optional
import encoder

# Try to import compiled extensions
try:
    import mcts_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("C++ extension not available. Run: python setup_extensions.py build_ext --inplace")

try:
    import mcts_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extension not available")

# Configuration
ENABLE_AGGRESSIVE_BATCHING = True
BATCH_SIZE = 256  # Neural network batch size
MAX_BATCH_WAIT_TIME = 0.001  # Max time to wait for batch to fill (seconds)
USE_GPU_TREE = CUDA_AVAILABLE  # Use GPU for tree operations
POSITION_CACHE_SIZE = 50000
MOVE_CACHE_SIZE = 50000

# Global caches
position_cache = {}
move_cache = {}
legal_move_cache = {}

# Batch queue for neural network inference
class BatchQueue:
    """Queue for aggregating neural network requests into batches."""
    
    def __init__(self, model, device, batch_size=BATCH_SIZE):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.queue = []
        self.results = {}
        self.last_process_time = time.time()
        
    def add_request(self, board_hash, position, mask):
        """Add a request to the queue."""
        request_id = len(self.results)
        self.queue.append((request_id, board_hash, position, mask))
        
        # Process if batch is full or timeout reached
        if len(self.queue) >= self.batch_size or \
           (time.time() - self.last_process_time) > MAX_BATCH_WAIT_TIME:
            self.process_batch()
        
        return request_id
    
    def process_batch(self):
        """Process all pending requests as a batch."""
        if not self.queue:
            return
        
        batch_size = len(self.queue)
        # Get model dtype to ensure compatibility
        model_dtype = next(self.model.parameters()).dtype
        positions = torch.zeros((batch_size, 16, 8, 8), device=self.device, dtype=model_dtype)
        masks = torch.zeros((batch_size, 72, 8, 8), device=self.device, dtype=model_dtype)
        
        request_ids = []
        board_hashes = []
        
        for i, (req_id, board_hash, position, mask) in enumerate(self.queue):
            positions[i] = position
            masks[i] = mask
            request_ids.append(req_id)
            board_hashes.append(board_hash)
        
        # Clear queue
        self.queue.clear()
        self.last_process_time = time.time()
        
        # Run inference
        with torch.no_grad():
            if self.model.training:
                self.model.eval()
            
            masks_flat = masks.view(batch_size, -1)
            values, policies = self.model(positions, policyMask=masks_flat)
            
            values = values.cpu().numpy()
            policies = policies.cpu().numpy()
        
        # Store results
        for i, req_id in enumerate(request_ids):
            self.results[req_id] = (values[i], policies[i])
            # Cache results
            if board_hashes[i] not in position_cache and len(position_cache) < POSITION_CACHE_SIZE:
                position_cache[board_hashes[i]] = (values[i], policies[i])
    
    def get_result(self, request_id):
        """Get result for a request, processing batch if needed."""
        if request_id not in self.results:
            self.process_batch()
        return self.results.pop(request_id)

# Global batch queue (initialized per model)
batch_queue = None

def get_position_hash(board):
    """Get hash for board position."""
    return hashlib.md5(board.fen().encode()).hexdigest()

class CudaNode:
    """GPU-accelerated node for MCTS tree."""
    
    def __init__(self, board, new_Q, move_probabilities, node_id=0):
        self.node_id = node_id
        self.N = 1.0
        self.sum_Q = new_Q if not math.isnan(new_Q) else 0.5
        
        # Get legal moves
        board_hash = get_position_hash(board)
        if board_hash in legal_move_cache:
            legal_moves = legal_move_cache[board_hash]
        else:
            legal_moves = list(board.legal_moves)
            if len(legal_move_cache) < MOVE_CACHE_SIZE:
                legal_move_cache[board_hash] = legal_moves
        
        self.num_edges = len(legal_moves)
        self.moves = legal_moves
        
        # Store edge data as tensors for GPU operations
        if self.num_edges > 0:
            self.edge_P = torch.tensor(
                [move_probabilities[i] for i in range(self.num_edges)], 
                dtype=torch.float32
            )
            self.edge_N = torch.zeros(self.num_edges, dtype=torch.float32)
            self.edge_Q = torch.zeros(self.num_edges, dtype=torch.float32)
            self.children = [None] * self.num_edges
            
            if CUDA_AVAILABLE and USE_GPU_TREE:
                self.edge_P = self.edge_P.cuda()
                self.edge_N = self.edge_N.cuda()
                self.edge_Q = self.edge_Q.cuda()
        else:
            self.edge_P = None
            self.edge_N = None
            self.edge_Q = None
            self.children = []
    
    def select_edge(self):
        """Select best edge using UCT."""
        if self.num_edges == 0:
            return None, -1
        
        if CUDA_AVAILABLE and USE_GPU_TREE and self.num_edges > 10:
            # Use GPU for larger edge sets
            Q_values = self.edge_Q
            N_values = self.edge_N
            P_values = self.edge_P
            
            uct_values = mcts_cuda.calc_uct(Q_values, N_values, P_values, self.N, 1.5)
            best_idx = torch.argmax(uct_values).item()
        elif CPP_AVAILABLE:
            # Use C++ implementation
            Q_values = self.edge_Q.cpu() if self.edge_Q.is_cuda else self.edge_Q
            N_values = self.edge_N.cpu() if self.edge_N.is_cuda else self.edge_N
            P_values = self.edge_P.cpu() if self.edge_P.is_cuda else self.edge_P
            
            uct_values = mcts_cpp.calc_uct_vectorized(Q_values, N_values, P_values, self.N, 1.5)
            best_idx = mcts_cpp.fast_argmax(uct_values)
        else:
            # Fallback to Python
            best_idx = -1
            best_uct = -float('inf')
            
            for i in range(self.num_edges):
                Q = self.edge_Q[i].item()
                N_c = self.edge_N[i].item()
                P = self.edge_P[i].item()
                
                if math.isnan(P):
                    P = 0.005
                
                uct = Q + P * 1.5 * math.sqrt(self.N) / (1 + N_c)
                
                if uct > best_uct:
                    best_uct = uct
                    best_idx = i
        
        return self.moves[best_idx], best_idx
    
    def expand_edge(self, edge_idx, board, new_Q, move_probabilities):
        """Expand an edge by creating child node."""
        if self.children[edge_idx] is None:
            child_id = self.node_id * 200 + edge_idx  # Simple ID scheme
            self.children[edge_idx] = CudaNode(board, new_Q, move_probabilities, child_id)
            return True
        return False
    
    def update_stats(self, value, from_child_perspective):
        """Update node statistics."""
        self.N += 1
        if from_child_perspective:
            self.sum_Q += 1.0 - value
        else:
            self.sum_Q += value
    
    def update_edge_stats(self, edge_idx, child_Q, child_N):
        """Update edge statistics from child."""
        if edge_idx < self.num_edges:
            self.edge_Q[edge_idx] = 1.0 - child_Q
            self.edge_N[edge_idx] = child_N

class CudaRoot(CudaNode):
    """Root node with additional functionality for parallel rollouts."""
    
    def __init__(self, board, neuralNetwork):
        global batch_queue
        
        # Set device first, before any neural network calls
        self.device = next(neuralNetwork.parameters()).device
        self.neuralNetwork = neuralNetwork
        
        # Initialize batch queue if needed
        if batch_queue is None:
            batch_queue = BatchQueue(neuralNetwork, self.device)
        
        # Get neural network evaluation
        value, move_probabilities = self._call_neural_network(board, neuralNetwork)
        Q = value / 2.0 + 0.5
        
        super().__init__(board, Q, move_probabilities, node_id=0)
        
        self.same_paths = 0
        
        # Pre-allocate tensors for batch operations
        self.batch_positions = None
        self.batch_masks = None
    
    def _call_neural_network(self, board, neuralNetwork):
        """Call neural network with caching and batching."""
        board_hash = get_position_hash(board)
        
        # Check cache
        if board_hash in position_cache:
            return position_cache[board_hash]
        
        # Encode position
        position, mask = encoder.encodePositionForInference(board)
        # Get model dtype to ensure compatibility
        model_dtype = next(neuralNetwork.parameters()).dtype
        position = torch.from_numpy(position).to(self.device, dtype=model_dtype)
        mask = torch.from_numpy(mask).to(self.device, dtype=model_dtype)
        
        if position.dim() == 3:
            position = position.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        if ENABLE_AGGRESSIVE_BATCHING and batch_queue is not None:
            # Add to batch queue
            request_id = batch_queue.add_request(board_hash, position, mask)
            value, policy = batch_queue.get_result(request_id)
        else:
            # Direct inference
            with torch.no_grad():
                mask_flat = mask.view(1, -1)
                value_tensor, policy_tensor = neuralNetwork(position, policyMask=mask_flat)
                value = value_tensor.item()
                policy = policy_tensor[0].cpu().numpy()
        
        move_probabilities = encoder.decodePolicyOutput(board, policy)
        
        # Cache result
        if len(position_cache) < POSITION_CACHE_SIZE:
            position_cache[board_hash] = (value, move_probabilities)
        
        return value, move_probabilities
    
    def parallel_rollouts_cuda(self, board, num_rollouts, num_threads=1):
        """GPU-accelerated parallel rollouts."""
        if not CUDA_AVAILABLE or not USE_GPU_TREE:
            # Fallback to regular rollouts
            for _ in range(num_rollouts):
                self.rollout(board.copy())
            return
        
        # Batch selection phase
        boards = [board.copy() for _ in range(num_rollouts)]
        paths = []
        leaf_boards = []
        leaf_indices = []
        
        # Selection phase - can be parallelized on GPU
        for i in range(num_rollouts):
            path = []
            edge_path = []
            current = self
            b = boards[i]
            
            while current is not None and current.num_edges > 0:
                move, edge_idx = current.select_edge()
                if move is None:
                    break
                
                path.append((current, edge_idx))
                edge_path.append(edge_idx)
                b.push(move)
                
                if current.children[edge_idx] is None:
                    # Unexpanded node
                    leaf_boards.append(b)
                    leaf_indices.append(i)
                    break
                
                current = current.children[edge_idx]
            
            paths.append(path)
        
        # Batch neural network evaluation for all leaf nodes
        if leaf_boards:
            batch_size = len(leaf_boards)
            
            if CUDA_AVAILABLE and batch_size > 1:
                # Batch encode positions on GPU
                # Get model dtype to ensure compatibility
                model_dtype = next(self.neuralNetwork.parameters()).dtype
                positions = torch.zeros((batch_size, 16, 8, 8), device=self.device, dtype=model_dtype)
                masks = torch.zeros((batch_size, 72, 8, 8), device=self.device, dtype=model_dtype)
                
                for i, b in enumerate(leaf_boards):
                    pos, mask = encoder.encodePositionForInference(b)
                    positions[i] = torch.from_numpy(pos).to(self.device, dtype=model_dtype)
                    masks[i] = torch.from_numpy(mask).to(self.device, dtype=model_dtype)
                
                # Batch inference
                with torch.no_grad():
                    masks_flat = masks.view(batch_size, -1)
                    values, policies = self.neuralNetwork(positions, policyMask=masks_flat)
                    values = values.cpu().numpy()
                    policies = policies.cpu().numpy()
            else:
                # Sequential evaluation
                values = []
                policies = []
                for b in leaf_boards:
                    value, policy = self._call_neural_network(b, self.neuralNetwork)
                    values.append(value)
                    move_probs = encoder.decodePolicyOutput(b, policy)
                    policies.append(move_probs)
                values = np.array(values)
        
        # Expansion and backpropagation
        leaf_idx = 0
        for i, path in enumerate(paths):
            if i in leaf_indices:
                # Expand leaf node
                if path:
                    parent, edge_idx = path[-1]
                    value = values[leaf_idx]
                    Q = value / 2.0 + 0.5
                    
                    if leaf_idx < len(policies):
                        move_probs = policies[leaf_idx]
                    else:
                        move_probs = encoder.decodePolicyOutput(
                            leaf_boards[leaf_idx], 
                            policies[leaf_idx]
                        )
                    
                    parent.expand_edge(edge_idx, leaf_boards[leaf_idx], Q, move_probs)
                    
                    # Backpropagate
                    for j in range(len(path) - 1, -1, -1):
                        node, _ = path[j]
                        is_from_child = (len(path) - 1 - j) % 2 == 1
                        node.update_stats(Q, is_from_child)
                        Q = 1.0 - Q
                    
                    # Update edge stats
                    for j in range(len(path) - 1):
                        parent_node, parent_edge = path[j]
                        child_node, _ = path[j + 1] if j + 1 < len(path) else (parent_node.children[parent_edge], 0)
                        if child_node:
                            parent_node.update_edge_stats(
                                parent_edge, 
                                child_node.sum_Q / child_node.N,
                                child_node.N
                            )
                
                leaf_idx += 1
    
    def rollout(self, board):
        """Single rollout for compatibility."""
        self.parallel_rollouts_cuda(board, 1)
    
    def parallelRollouts(self, board, neuralNetwork, num_threads):
        """Parallel rollouts with GPU acceleration."""
        self.parallel_rollouts_cuda(board, num_threads)
    
    def maxNSelect(self):
        """Select move with highest visit count."""
        if self.num_edges == 0:
            return None
        
        best_idx = torch.argmax(self.edge_N).item() if torch.is_tensor(self.edge_N) else np.argmax(self.edge_N)
        
        class EdgeWrapper:
            def __init__(self, move):
                self.move = move
            
            def getMove(self):
                return self.move
        
        return EdgeWrapper(self.moves[best_idx])
    
    def getN(self):
        """Get total visit count."""
        return self.N
    
    def getQ(self):
        """Get average Q value."""
        return self.sum_Q / self.N if self.N > 0 else 0.0
    
    def getStatisticsString(self):
        """Get statistics string for display."""
        if self.num_edges == 0:
            return "No legal moves"
        
        lines = ['|{: ^10}|{: ^10}|{: ^10}|{: ^10}|'.format('move', 'P', 'N', 'Q')]
        
        # Get indices sorted by N
        N_values = self.edge_N.cpu().numpy() if torch.is_tensor(self.edge_N) else self.edge_N
        sorted_indices = np.argsort(N_values)[::-1]
        
        for idx in sorted_indices[:20]:  # Top 20 moves
            move = self.moves[idx]
            P = self.edge_P[idx].item() if torch.is_tensor(self.edge_P) else self.edge_P[idx]
            N = self.edge_N[idx].item() if torch.is_tensor(self.edge_N) else self.edge_N[idx]
            Q = self.edge_Q[idx].item() if torch.is_tensor(self.edge_Q) else self.edge_Q[idx]
            
            lines.append('|{: ^10}|{:10.4f}|{:10.0f}|{:10.4f}|'.format(
                str(move), P, N, Q
            ))
        
        return '\n'.join(lines)

# Compatibility wrapper
def Root(board, neuralNetwork):
    """Create root node with appropriate implementation."""
    if CUDA_AVAILABLE and USE_GPU_TREE:
        return CudaRoot(board, neuralNetwork)
    else:
        # Fallback to CPU implementation
        from MCTS_advanced_optimizations import Root as CpuRoot
        return CpuRoot(board, neuralNetwork)

# Clear functions
def clear_caches():
    """Clear all caches."""
    position_cache.clear()
    move_cache.clear()
    legal_move_cache.clear()

def clear_batch_queue():
    """Clear batch queue."""
    global batch_queue
    if batch_queue is not None:
        batch_queue.queue.clear()
        batch_queue.results.clear()

# Build instructions
def build_extensions():
    """Build C++ and CUDA extensions."""
    import subprocess
    import sys
    
    print("Building extensions...")
    result = subprocess.run([sys.executable, "setup_extensions.py", "build_ext", "--inplace"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Extensions built successfully!")
        print(result.stdout)
    else:
        print("Error building extensions:")
        print(result.stderr)
        return False
    
    return True

# Alias for compatibility
Root = CudaRoot
Node = CudaNode

# Auto-build on import if needed
if not CPP_AVAILABLE and not CUDA_AVAILABLE:
    print("\nPerformance extensions not available.")
    print("To build them, run: python setup_extensions.py build_ext --inplace")
    print("Or call MCTS_cuda_optimized.build_extensions()\n")