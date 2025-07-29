#!/usr/bin/env python3
"""
High-Performance MCTS Implementation for CUDA/Windows

This implementation is optimized for maximum throughput on high-end GPUs like RTX 4080.
Key optimizations:
1. Minimal overhead tree structure
2. Lock-free operations where possible
3. Efficient batching with pre-allocated buffers
4. Direct numpy/torch operations
5. Simplified backup process
"""

import chess
import numpy as np
import torch
import threading
import queue
import time
import math
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple
import encoder

# Simple node structure
MCTSNode = namedtuple('MCTSNode', ['board_hash', 'visits', 'total_value', 'prior', 'move', 'parent_idx'])

class HighPerformanceMCTS:
    """
    Simplified, high-performance MCTS optimized for GPU throughput.
    """
    
    def __init__(self, neural_network, device, batch_size=512, c_puct=1.5):
        self.neural_network = neural_network
        self.device = device
        self.batch_size = batch_size
        self.c_puct = c_puct
        
        # Pre-allocate arrays for tree
        self.max_nodes = 1000000
        self.nodes = []
        self.children = {}  # node_idx -> list of child indices
        self.node_lookup = {}  # board_hash -> node_idx
        
        # Pre-allocate GPU buffers
        self.position_buffer = torch.zeros((batch_size, 16, 8, 8), device=device, dtype=torch.float32)
        self.mask_buffer = torch.zeros((batch_size, 72 * 8 * 8), device=device, dtype=torch.float32)
        
        # Batch processing
        self.eval_queue = queue.Queue(maxsize=batch_size * 10)
        self.running = False
        self.server_thread = None
        
    def start(self):
        """Start the batch evaluation server"""
        self.running = True
        self.server_thread = threading.Thread(target=self._batch_server, daemon=True)
        self.server_thread.start()
        
    def stop(self):
        """Stop the batch evaluation server"""
        self.running = False
        if self.server_thread:
            self.server_thread.join()
            
    def _batch_server(self):
        """Efficient batch processing server"""
        while self.running:
            batch = []
            batch_boards = []
            
            # Collect batch with minimal wait
            deadline = time.time() + 0.002  # 2ms max wait
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0.0001, deadline - time.time())
                    item = self.eval_queue.get(timeout=timeout)
                    batch.append(item)
                    batch_boards.append(item[0])
                except queue.Empty:
                    break
                    
            if not batch:
                continue
                
            # Process batch on GPU
            batch_size = len(batch)
            
            # Fill buffers efficiently
            for i, (board, _) in enumerate(batch):
                pos, mask = encoder.encodePositionForInference(board)
                self.position_buffer[i] = torch.from_numpy(pos)
                self.mask_buffer[i] = torch.from_numpy(mask).flatten()
                
            # Single GPU evaluation
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    values, policies = self.neural_network(
                        self.position_buffer[:batch_size],
                        policyMask=self.mask_buffer[:batch_size]
                    )
                    
            # Process results
            values_np = values.cpu().numpy()
            policies_np = policies.cpu().numpy()
            
            for i, (board, callback) in enumerate(batch):
                value = float(values_np[i, 0])
                policy = policies_np[i]
                move_probs = encoder.decodePolicyOutput(board, policy)
                callback(value, move_probs)
                
    def search(self, board, num_simulations):
        """
        Run MCTS search from the given position.
        
        Args:
            board: chess.Board position
            num_simulations: number of simulations to run
            
        Returns:
            Best move based on visit counts
        """
        # Reset tree
        self.nodes = []
        self.children = {}
        self.node_lookup = {}
        
        # Create root node
        root_evaluated = threading.Event()
        root_value = None
        root_priors = None
        
        def root_callback(value, move_probs):
            nonlocal root_value, root_priors
            root_value = value
            root_priors = move_probs
            root_evaluated.set()
            
        self.eval_queue.put((board, root_callback))
        root_evaluated.wait(timeout=1.0)
        
        if root_value is None:
            raise RuntimeError("Failed to evaluate root position")
            
        # Initialize root
        root_hash = board.fen()
        root_idx = 0
        self.nodes.append(MCTSNode(root_hash, 1, root_value, 1.0, None, -1))
        self.node_lookup[root_hash] = root_idx
        self._expand_node(root_idx, board, root_priors)
        
        # Run parallel simulations
        completed_sims = 0
        sim_lock = threading.Lock()
        
        def run_simulation():
            nonlocal completed_sims
            
            while True:
                with sim_lock:
                    if completed_sims >= num_simulations:
                        break
                    completed_sims += 1
                    
                self._simulate(board.copy())
                
        # Use thread pool for simulations
        num_threads = min(32, num_simulations // 10)  # Fewer threads, more work per thread
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_simulation) for _ in range(num_threads)]
            for f in futures:
                f.result()
                
        # Select best move based on visits
        if root_idx not in self.children:
            return None
            
        best_child_idx = -1
        best_visits = -1
        
        for child_idx in self.children[root_idx]:
            child = self.nodes[child_idx]
            if child.visits > best_visits:
                best_visits = child.visits
                best_child_idx = child_idx
                
        if best_child_idx >= 0:
            return self.nodes[best_child_idx].move
            
        return None
        
    def _simulate(self, board):
        """Run a single simulation"""
        path = []
        node_idx = 0  # Start from root
        
        # Selection phase
        while True:
            path.append(node_idx)
            
            # Check if terminal
            if board.is_game_over():
                result = board.result()
                value = encoder.parseResult(result)
                if not board.turn:
                    value = -value
                self._backup(path, value)
                return
                
            # Check if leaf
            if node_idx not in self.children:
                # Expand and evaluate
                evaluated = threading.Event()
                leaf_value = None
                leaf_priors = None
                
                def callback(value, priors):
                    nonlocal leaf_value, leaf_priors
                    leaf_value = value
                    leaf_priors = priors
                    evaluated.set()
                    
                self.eval_queue.put((board, callback))
                
                if evaluated.wait(timeout=0.5):
                    self._expand_node(node_idx, board, leaf_priors)
                    self._backup(path, leaf_value)
                return
                
            # Select best child
            children = self.children[node_idx]
            if not children:
                return
                
            best_child_idx = self._select_child(children)
            best_child = self.nodes[best_child_idx]
            
            # Make move
            board.push(best_child.move)
            node_idx = best_child_idx
            
    def _select_child(self, children_indices):
        """Select best child using PUCT formula"""
        parent_visits = sum(self.nodes[idx].visits for idx in children_indices)
        sqrt_parent = math.sqrt(parent_visits)
        
        best_idx = -1
        best_puct = -float('inf')
        
        for child_idx in children_indices:
            child = self.nodes[child_idx]
            
            # Calculate PUCT value
            if child.visits > 0:
                q_value = child.total_value / child.visits
            else:
                q_value = 0
                
            u_value = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            puct = q_value + u_value
            
            if puct > best_puct:
                best_puct = puct
                best_idx = child_idx
                
        return best_idx
        
    def _expand_node(self, node_idx, board, move_probs):
        """Expand a node with its children"""
        children_indices = []
        
        for i, move in enumerate(board.legal_moves):
            if i < len(move_probs) and move_probs[i] > 0:
                # Create child node
                board.push(move)
                child_hash = board.fen()
                board.pop()
                
                # Check if already exists
                if child_hash in self.node_lookup:
                    child_idx = self.node_lookup[child_hash]
                else:
                    child_idx = len(self.nodes)
                    child_node = MCTSNode(
                        child_hash, 0, 0.0, 
                        float(move_probs[i]), move, node_idx
                    )
                    self.nodes.append(child_node)
                    self.node_lookup[child_hash] = child_idx
                    
                children_indices.append(child_idx)
                
        self.children[node_idx] = children_indices
        
    def _backup(self, path, value):
        """Backup value through the path"""
        for i, node_idx in enumerate(path):
            # Flip value for alternating players
            backup_value = value if i % 2 == 0 else -value
            
            # Update node - use simple increments (thread-safe for basic types)
            node = self.nodes[node_idx]
            self.nodes[node_idx] = MCTSNode(
                node.board_hash,
                node.visits + 1,
                node.total_value + backup_value,
                node.prior,
                node.move,
                node.parent_idx
            )
            
    def get_action_probabilities(self, board, temperature=1.0):
        """Get move probabilities based on visit counts"""
        root_idx = 0
        if root_idx not in self.children:
            return []
            
        moves = []
        visits = []
        
        for child_idx in self.children[root_idx]:
            child = self.nodes[child_idx]
            moves.append(child.move)
            visits.append(child.visits)
            
        visits = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            # Select best
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
            
        return list(zip(moves, probs))


class HighPerformanceMCTSEngine:
    """
    Engine wrapper for high-performance MCTS.
    """
    
    def __init__(self, neural_network, device=None, batch_size=512, verbose=False):
        if device is None:
            device = torch.device('cuda:0')
            
        self.device = device
        self.verbose = verbose
        
        # Ensure model is on GPU and optimized
        self.neural_network = neural_network.to(device)
        self.neural_network.eval()
        
        # Disable gradients
        for param in self.neural_network.parameters():
            param.requires_grad = False
            
        # Create MCTS instance
        self.mcts = HighPerformanceMCTS(self.neural_network, device, batch_size)
        
    def start(self):
        """Start the engine"""
        self.mcts.start()
        if self.verbose:
            print("High-Performance MCTS Engine started")
            
    def stop(self):
        """Stop the engine"""
        self.mcts.stop()
        if self.verbose:
            print("High-Performance MCTS Engine stopped")
            
    def search(self, board, num_rollouts):
        """
        Search for best move.
        
        Args:
            board: chess.Board position
            num_rollouts: number of MCTS simulations
            
        Returns:
            Best move
        """
        start_time = time.time()
        
        best_move = self.mcts.search(board, num_rollouts)
        
        elapsed = time.time() - start_time
        nps = num_rollouts / elapsed if elapsed > 0 else 0
        
        if self.verbose:
            print(f"Search complete: {num_rollouts} nodes in {elapsed:.2f}s ({nps:.0f} NPS)")
            
        return best_move
        
    def get_action_probabilities(self, board, num_rollouts, temperature=1.0):
        """Get move probabilities for training"""
        self.mcts.search(board, num_rollouts)
        return self.mcts.get_action_probabilities(board, temperature)