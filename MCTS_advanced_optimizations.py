"""
Advanced optimizations for MCTS including:
1. Batch position encoding
2. Reduced lock contention with finer-grained locking
3. Pre-compiled numba functions for hot paths
4. Better memory management
"""

import encoder
import math
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from collections import deque
import torch
import hashlib
from functools import lru_cache
import multiprocessing as mp

# Try to import numba for JIT compilation
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - falling back to pure Python")
    # Define dummy decorator
    def njit(func):
        return func

# Global caches for optimization
position_cache = {}  # Cache for position encodings
legal_moves_cache = {}  # Cache for legal move generation
CACHE_MAX_SIZE = 20000  # Larger cache for better hit rate

# Object pools for Node/Edge creation
node_pool = deque(maxlen=100000)
edge_pool = deque(maxlen=400000)

# Thread-local storage for reducing contention
import threading
thread_local = threading.local()

@lru_cache(maxsize=10000)
def get_position_hash(fen_string):
    """Get a hash for the board position from FEN string."""
    return hashlib.md5(fen_string.encode()).hexdigest()

@njit
def calcUCT_numba(Q, N_c, P, N_p, C=1.5):
    """Numba-compiled UCT calculation for maximum speed."""
    if math.isnan(P):
        P = 0.005  # 1.0 / 200.0
    
    UCT = Q + P * C * math.sqrt(N_p) / (1 + N_c)
    
    if math.isnan(UCT):
        UCT = 0.0
    
    return UCT

def calcUCT(edge, N_p):
    """Calculate the UCT formula."""
    if NUMBA_AVAILABLE:
        return calcUCT_numba(edge.getQ(), edge.getN(), edge.getP(), N_p)
    else:
        Q = edge.getQ()
        N_c = edge.getN()
        P = edge.getP()
        
        if math.isnan(P):
            P = 1.0 / 200.0
        
        C = 1.5
        UCT = Q + P * C * math.sqrt(N_p) / (1 + N_c)
        
        if math.isnan(UCT):
            UCT = 0.0
        
        return UCT

@njit
def vectorized_calcUCT_numba(Q_values, N_values, P_values, N_p, C=1.5):
    """Numba-compiled vectorized UCT calculation."""
    n_edges = len(Q_values)
    UCT_values = np.zeros(n_edges, dtype=np.float32)
    sqrt_N_p = math.sqrt(N_p)
    
    for i in range(n_edges):
        P = P_values[i]
        if math.isnan(P):
            P = 0.005
        
        UCT = Q_values[i] + P * C * sqrt_N_p / (1 + N_values[i])
        
        if math.isnan(UCT):
            UCT = 0.0
            
        UCT_values[i] = UCT
    
    return UCT_values

def vectorized_calcUCT(edges, N_p):
    """Vectorized UCT calculation for multiple edges."""
    n_edges = len(edges)
    if n_edges == 0:
        return np.array([])
    
    # Pre-allocate arrays
    Q_values = np.empty(n_edges, dtype=np.float32)
    N_values = np.empty(n_edges, dtype=np.float32)
    P_values = np.empty(n_edges, dtype=np.float32)
    
    # Extract values in batch
    for i, edge in enumerate(edges):
        Q_values[i] = edge.getQ()
        N_values[i] = edge.getN()
        P_values[i] = edge.getP()
    
    if NUMBA_AVAILABLE:
        return vectorized_calcUCT_numba(Q_values, N_values, P_values, N_p)
    else:
        # Handle NaN values
        P_values = np.where(np.isnan(P_values), 1.0 / 200.0, P_values)
        
        C = 1.5
        sqrt_N_p = math.sqrt(N_p)
        
        # Vectorized UCT calculation
        UCT_values = Q_values + P_values * C * sqrt_N_p / (1 + N_values)
        
        # Handle NaN in results
        UCT_values = np.where(np.isnan(UCT_values), 0.0, UCT_values)
        
        return UCT_values

class Node:
    """Optimized node with reduced memory footprint and better cache locality."""
    
    __slots__ = ['_lock', 'N', 'sum_Q', 'edges', '_edges_array']  # Reduce memory usage
    
    def __init__(self, board, new_Q, move_probabilities):
        self._lock = RLock()
        self.N = 1.
        
        if math.isnan(new_Q):
            self.sum_Q = 0.5
        else:
            self.sum_Q = new_Q
        
        self.edges = []
        self._edges_array = None  # Cached array for vectorized operations
        
        # Use board FEN for hashing (more efficient than custom hash)
        board_fen = board.fen()
        board_hash = get_position_hash(board_fen)
        
        if board_hash in legal_moves_cache:
            legal_moves = legal_moves_cache[board_hash]
        else:
            legal_moves = list(board.legal_moves)
            if len(legal_moves_cache) < CACHE_MAX_SIZE:
                legal_moves_cache[board_hash] = legal_moves
        
        # Batch edge creation
        for idx, move in enumerate(legal_moves):
            if edge_pool:
                edge = edge_pool.pop()
                edge.reinit(move, move_probabilities[idx])
            else:
                edge = Edge(move, move_probabilities[idx])
            self.edges.append(edge)
    
    def getN(self):
        return self.N
    
    def getQ(self):
        return self.sum_Q / self.N
    
    def UCTSelect(self):
        """Optimized edge selection using vectorized operations."""
        if not self.edges:
            return None
        
        # Use cached edges array if available
        if self._edges_array is None or len(self._edges_array) != len(self.edges):
            self._edges_array = np.array(self.edges, dtype=object)
        
        uct_values = vectorized_calcUCT(self.edges, self.N)
        
        if len(uct_values) == 0:
            return None
        
        max_idx = np.argmax(uct_values)
        return self.edges[max_idx]
    
    def maxNSelect(self):
        """Optimized using numpy operations."""
        if not self.edges:
            return None
        
        N_values = np.array([edge.getN() for edge in self.edges])
        max_idx = np.argmax(N_values)
        return self.edges[max_idx]
    
    def getStatisticsString(self):
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
                'move', 'P', 'N', 'Q', 'UCT')
        
        # Sort by N descending
        edges_sorted = sorted(self.edges, key=lambda e: e.getN(), reverse=True)
        
        for edge in edges_sorted:
            move = edge.getMove()
            P = edge.getP()
            N = edge.getN()
            Q = edge.getQ()
            UCT = calcUCT(edge, self.N)
            
            string += '|{: ^10}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|\n'.format(
                str(move), P, N, Q, UCT)
        
        return string
    
    def isTerminal(self):
        return len(self.edges) == 0
    
    def updateStats(self, value, from_child_perspective):
        """Thread-safe update with minimal lock time."""
        adjustment = (1. - value) if from_child_perspective else value
        with self._lock:
            self.N += 1
            self.sum_Q += adjustment
    
    def cleanup(self):
        """Return edges to pool when node is no longer needed."""
        for edge in self.edges:
            edge.cleanup()
            if len(edge_pool) < edge_pool.maxlen:
                edge_pool.append(edge)
        self.edges.clear()
        self._edges_array = None

class Edge:
    """Optimized edge with reduced memory footprint."""
    
    __slots__ = ['move', 'P', 'child', '_lock', 'virtualLosses']  # Reduce memory usage
    
    def __init__(self, move, move_probability):
        self.move = move
        
        if math.isnan(move_probability) or move_probability < 0:
            self.P = 1.0 / 200.0
        else:
            self.P = move_probability
        
        self.child = None
        self._lock = Lock()
        self.virtualLosses = 0.
    
    def reinit(self, move, move_probability):
        """Reinitialize edge for reuse from pool."""
        self.move = move
        if math.isnan(move_probability) or move_probability < 0:
            self.P = 1.0 / 200.0
        else:
            self.P = move_probability
        self.child = None
        self.virtualLosses = 0.
    
    def has_child(self):
        return self.child is not None
    
    def getN(self):
        if self.has_child():
            return self.child.N + self.virtualLosses
        else:
            return self.virtualLosses
    
    def getQ(self):
        if self.has_child():
            return 1. - ((self.child.sum_Q + self.virtualLosses) / 
                        (self.child.N + self.virtualLosses))
        else:
            return 0.
    
    def getP(self):
        return self.P
    
    def expand(self, board, new_Q, move_probabilities):
        if self.child is None:
            if node_pool:
                self.child = node_pool.pop()
                self.child.__init__(board, new_Q, move_probabilities)
            else:
                self.child = Node(board, new_Q, move_probabilities)
            return True
        return False
    
    def getChild(self):
        return self.child
    
    def getMove(self):
        return self.move
    
    def addVirtualLoss(self):
        with self._lock:
            self.virtualLosses += 1
    
    def clearVirtualLoss(self):
        with self._lock:
            self.virtualLosses = 0.
    
    def cleanup(self):
        if self.child:
            self.child.cleanup()
            if len(node_pool) < node_pool.maxlen:
                node_pool.append(self.child)
        self.child = None
        self.virtualLosses = 0.

class Root(Node):
    """Optimized root node with better parallelization."""
    
    def __init__(self, board, neuralNetwork):
        # Use optimized neural network call
        value, move_probabilities = callNeuralNetworkOptimized(board, neuralNetwork)
        
        Q = value / 2. + 0.5
        
        super().__init__(board, Q, move_probabilities)
        
        self.same_paths = 0
        self.thread_pool = None
        self.max_workers = mp.cpu_count()
        self.nn_device = next(neuralNetwork.parameters()).device
        
        # Pre-allocate work arrays for parallel operations
        self._work_boards = None
        self._work_node_paths = None
        self._work_edge_paths = None
    
    def selectTask(self, board, node_path, edge_path):
        """Optimized selection with reduced overhead."""
        cNode = self
        
        while True:
            node_path.append(cNode)
            
            cEdge = cNode.UCTSelect()
            edge_path.append(cEdge)
            
            if cEdge is None:
                assert cNode.isTerminal()
                break
            
            cEdge.addVirtualLoss()
            board.push(cEdge.getMove())
            
            if not cEdge.has_child():
                break
            
            cNode = cEdge.getChild()
    
    def parallelRolloutsOptimized(self, board, neuralNetwork, num_parallel_rollouts):
        """Highly optimized parallel rollouts with better work distribution."""
        
        # Initialize thread pool if needed
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Pre-allocate work arrays if needed
        if (self._work_boards is None or 
            len(self._work_boards) < num_parallel_rollouts):
            self._work_boards = [None] * num_parallel_rollouts
            self._work_node_paths = [[] for _ in range(num_parallel_rollouts)]
            self._work_edge_paths = [[] for _ in range(num_parallel_rollouts)]
        
        # Reuse work arrays
        for i in range(num_parallel_rollouts):
            self._work_boards[i] = board.copy()
            self._work_node_paths[i].clear()
            self._work_edge_paths[i].clear()
        
        # Use only the arrays we need
        boards = self._work_boards[:num_parallel_rollouts]
        node_paths = self._work_node_paths[:num_parallel_rollouts]
        edge_paths = self._work_edge_paths[:num_parallel_rollouts]
        
        # Submit selection tasks
        futures = []
        for i in range(num_parallel_rollouts):
            future = self.thread_pool.submit(
                self.selectTask, boards[i], node_paths[i], edge_paths[i])
            futures.append(future)
        
        # Wait for all selections
        for future in futures:
            future.result()
        
        # Batch neural network evaluation
        values, move_probabilities = callNeuralNetworkBatchedOptimized(
            boards[:num_parallel_rollouts], neuralNetwork)
        
        # Process results in parallel
        for i in range(num_parallel_rollouts):
            edge = edge_paths[i][-1]
            board = boards[i]
            value = values[i]
            
            if edge is not None:
                new_Q = value / 2. + 0.5
                
                isunexpanded = edge.expand(board, new_Q, move_probabilities[i])
                
                if not isunexpanded:
                    self.same_paths += 1
                
                new_Q = 1. - new_Q
            else:
                winner = encoder.parseResult(board.result())
                if not board.turn:
                    winner *= -1
                new_Q = float(winner) / 2. + 0.5
            
            # Update tree
            last_node_idx = len(node_paths[i]) - 1
            for r in range(last_node_idx, -1, -1):
                node = node_paths[i][r]
                is_from_child = (last_node_idx - r) % 2 == 1
                node.updateStats(new_Q, is_from_child)
            
            # Clear virtual losses
            for edge in edge_paths[i]:
                if edge is not None:
                    edge.clearVirtualLoss()
    
    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        return self.parallelRolloutsOptimized(board, neuralNetwork, num_parallel_rollouts)
    
    def getVisitCounts(self, board):
        """Get visit counts for all possible moves."""
        visit_counts = np.zeros(4608, dtype=np.float32)
        
        for edge in self.edges:
            move = edge.getMove()
            
            if not board.turn:
                from encoder import mirrorMove
                mirrored_move = mirrorMove(move)
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(mirrored_move)
            else:
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
            
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            visit_counts[moveIdx] = edge.getN()
        
        return visit_counts
    
    def cleanup(self):
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        super().cleanup()

# Optimized neural network calls

def callNeuralNetworkOptimized(board, neuralNetwork):
    """Optimized single inference with minimal data transfer."""
    board_fen = board.fen()
    board_hash = get_position_hash(board_fen)
    
    # Check position cache
    if board_hash in position_cache:
        position, mask = position_cache[board_hash]
        # Clone to avoid modifying cached tensor
        position = position.clone()
        mask = mask.clone()
    else:
        position, mask = encoder.encodePositionForInference(board)
        position = torch.from_numpy(position)
        mask = torch.from_numpy(mask)
        
        # Cache if not full
        if len(position_cache) < CACHE_MAX_SIZE:
            position_cache[board_hash] = (position.clone(), mask.clone())
    
    # Ensure 4D tensor
    if position.dim() == 3:
        position = position.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Move to device
    model_device = next(neuralNetwork.parameters()).device
    position = position.to(model_device)
    mask = mask.to(model_device)
    
    # Convert to half precision if needed
    if next(neuralNetwork.parameters()).dtype == torch.float16:
        position = position.half()
        mask = mask.half()
    
    # Flatten mask
    mask_flat = mask.view(mask.shape[0], -1)
    
    with torch.no_grad():
        value, policy = neuralNetwork(position, policyMask=mask_flat)
    
    # Efficient conversion
    value = value.item()
    policy = policy[0].cpu().numpy()
    
    move_probabilities = encoder.decodePolicyOutput(board, policy)
    
    return value, move_probabilities

def callNeuralNetworkBatchedOptimized(boards, neuralNetwork):
    """Highly optimized batch inference."""
    num_inputs = len(boards)
    
    # Collect unique positions
    unique_positions = {}
    board_hashes = []
    
    for board in boards:
        board_fen = board.fen()
        board_hash = get_position_hash(board_fen)
        board_hashes.append(board_hash)
        
        if board_hash not in unique_positions:
            if board_hash in position_cache:
                unique_positions[board_hash] = position_cache[board_hash]
            else:
                position, mask = encoder.encodePositionForInference(board)
                position_tensor = torch.from_numpy(position)
                mask_tensor = torch.from_numpy(mask)
                unique_positions[board_hash] = (position_tensor, mask_tensor)
                
                if len(position_cache) < CACHE_MAX_SIZE:
                    position_cache[board_hash] = (position_tensor.clone(), 
                                                mask_tensor.clone())
    
    # Create batch tensors efficiently
    inputs = torch.stack([unique_positions[h][0] for h in board_hashes])
    masks = torch.stack([unique_positions[h][1] for h in board_hashes])
    
    # Move to device
    model_device = next(neuralNetwork.parameters()).device
    inputs = inputs.to(model_device)
    masks = masks.to(model_device)
    
    # Convert to half precision if needed
    if next(neuralNetwork.parameters()).dtype == torch.float16:
        inputs = inputs.half()
        masks = masks.half()
    
    # Flatten masks
    masks_flat = masks.view(masks.shape[0], -1)
    
    with torch.no_grad():
        value, policy = neuralNetwork(inputs, policyMask=masks_flat)
    
    # Efficient processing
    values = value.cpu().numpy().reshape(num_inputs)
    policy_cpu = policy.cpu().numpy()
    
    # Decode move probabilities
    move_probabilities = np.zeros((num_inputs, 200), dtype=np.float32)
    
    for i in range(num_inputs):
        move_probs = encoder.decodePolicyOutput(boards[i], policy_cpu[i])
        move_probabilities[i, :len(move_probs)] = move_probs
    
    return values, move_probabilities

# Cache management functions
def clear_caches():
    """Clear all caches to free memory."""
    position_cache.clear()
    legal_moves_cache.clear()
    get_position_hash.cache_clear()

def clear_pools():
    """Clear object pools."""
    node_pool.clear()
    edge_pool.clear()

def get_cache_stats():
    """Get cache statistics."""
    return {
        'position_cache_size': len(position_cache),
        'legal_moves_cache_size': len(legal_moves_cache),
        'position_hash_cache_info': get_position_hash.cache_info(),
        'node_pool_size': len(node_pool),
        'edge_pool_size': len(edge_pool)
    }