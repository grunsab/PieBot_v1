import chess
import torch
import numpy as np
import threading
import queue
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import encoder

@dataclass
class MCTSNode:
    """Optimized node structure for MCTS"""
    board_hash: str
    visits: int
    total_value: float
    prior: float
    move: chess.Move
    parent_idx: int
    virtual_loss: int = 0  # For preventing thread collisions

class UltraPerformanceMCTSEngine:
    """
    Ultra high-performance MCTS implementation using leaf batching.
    
    Key optimizations:
    - Collects multiple leaf nodes before neural network evaluation
    - Uses virtual loss to prevent thread collisions
    - Pre-allocated buffers for zero-copy operations
    - Async pipeline for continuous GPU utilization
    """
    
    def __init__(self, neural_network, device=None, batch_size=512, 
                 num_workers=16, c_puct=1.5, verbose=True):
        self.neural_network = neural_network
        self.neural_network.eval()
        
        if device is None:
            device = next(neural_network.parameters()).device
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.c_puct = c_puct
        self.verbose = verbose
        
        # Tree structures
        self.nodes = []
        self.children = {}  # node_idx -> list of child indices
        self.node_lookup = {}  # board_hash -> node_idx
        
        # Leaf collection buffers (double buffering)
        self.leaf_buffer_size = batch_size
        self.leaf_buffers = [
            {
                'boards': [],
                'node_indices': [],
                'paths': [],
                'count': 0
            }
            for _ in range(2)
        ]
        self.current_buffer = 0
        self.buffer_lock = threading.Lock()
        self.buffer_ready = threading.Condition(self.buffer_lock)
        
        # Pre-allocated GPU tensors
        self.position_tensor = torch.zeros((batch_size, 16, 8, 8), 
                                         device=device, dtype=torch.float32)
        self.mask_tensor = torch.zeros((batch_size, 72 * 8 * 8), 
                                      device=device, dtype=torch.float32)
        
        # CPU staging buffers with pinned memory
        if device.type == 'cuda':
            self.cpu_positions = torch.zeros((batch_size, 16, 8, 8), 
                                            dtype=torch.float32, pin_memory=True)
            self.cpu_masks = torch.zeros((batch_size, 72 * 8 * 8), 
                                        dtype=torch.float32, pin_memory=True)
        elif device.type == 'mps':
            # MPS doesn't support pinned memory, use regular tensors
            self.cpu_positions = torch.zeros((batch_size, 16, 8, 8), 
                                            dtype=torch.float32)
            self.cpu_masks = torch.zeros((batch_size, 72 * 8 * 8), 
                                        dtype=torch.float32)
        
        # Worker synchronization
        self.running = False
        self.gpu_thread = None
        self.total_simulations = 0
        self.simulation_lock = threading.Lock()
        
    def start(self):
        """Start the GPU evaluation thread"""
        self.running = True
        self.gpu_thread = threading.Thread(target=self._gpu_evaluator, daemon=True)
        self.gpu_thread.start()
        if self.verbose:
            print("Ultra-Performance MCTS Engine started")
    
    def stop(self):
        """Stop the engine"""
        self.running = False
        with self.buffer_lock:
            self.buffer_ready.notify_all()
        if self.gpu_thread:
            self.gpu_thread.join()
        if self.verbose:
            print("Ultra-Performance MCTS Engine stopped")
    
    def _gpu_evaluator(self):
        """GPU evaluation thread - processes leaf batches continuously"""
        while self.running:
            with self.buffer_lock:
                # Wait for a buffer to be ready
                while self.running and all(buf['count'] == 0 for buf in self.leaf_buffers):
                    self.buffer_ready.wait(timeout=0.001)
                
                if not self.running:
                    break
                
                # Find buffer with data
                eval_buffer_idx = -1
                for i in range(2):
                    if self.leaf_buffers[i]['count'] > 0:
                        eval_buffer_idx = i
                        break
                
                if eval_buffer_idx == -1:
                    continue
                
                # Swap buffers
                eval_buffer = self.leaf_buffers[eval_buffer_idx]
                batch_size = eval_buffer['count']
                boards = eval_buffer['boards'][:batch_size]
                node_indices = eval_buffer['node_indices'][:batch_size]
                paths = eval_buffer['paths'][:batch_size]
                
                # Clear the buffer for reuse
                eval_buffer['count'] = 0
                eval_buffer['boards'].clear()
                eval_buffer['node_indices'].clear()
                eval_buffer['paths'].clear()
                
            # Limit batch size to available buffer size
            batch_size = min(batch_size, self.batch_size)
            
            # Prepare batch on CPU with pinned memory
            if self.device.type in ['cuda', 'mps']:
                for i, board in enumerate(boards):
                    pos, mask = encoder.encodePositionForInference(board)
                    self.cpu_positions[i].copy_(torch.from_numpy(pos))
                    self.cpu_masks[i].copy_(torch.from_numpy(mask).flatten())
                
                # Copy to GPU (async for CUDA, sync for MPS)
                self.position_tensor[:batch_size].copy_(
                    self.cpu_positions[:batch_size], non_blocking=(self.device.type == 'cuda'))
                self.mask_tensor[:batch_size].copy_(
                    self.cpu_masks[:batch_size], non_blocking=(self.device.type == 'cuda'))
            else:
                # CPU path
                for i, board in enumerate(boards):
                    pos, mask = encoder.encodePositionForInference(board)
                    self.position_tensor[i].copy_(torch.from_numpy(pos))
                    self.mask_tensor[i].copy_(torch.from_numpy(mask).flatten())
            
            # GPU evaluation
            with torch.no_grad():
                # Use autocast only for CUDA, MPS doesn't support it yet
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        values, policies = self.neural_network(
                            self.position_tensor[:batch_size],
                            policyMask=self.mask_tensor[:batch_size]
                        )
                else:
                    values, policies = self.neural_network(
                        self.position_tensor[:batch_size],
                        policyMask=self.mask_tensor[:batch_size]
                    )
            
            # Process results
            values_np = values.cpu().numpy()
            policies_np = policies.cpu().numpy()
            
            # Expand nodes and backup values
            for i in range(batch_size):
                board = boards[i]
                node_idx = node_indices[i]
                path = paths[i]
                value = float(values_np[i, 0])
                policy = policies_np[i]
                
                # Decode policy
                move_probs = encoder.decodePolicyOutput(board, policy)
                
                # Expand node
                self._expand_node(node_idx, board, move_probs)
                
                # Remove virtual loss and backup real value
                self._backup(path, value, remove_virtual_loss=True)
    
    def search(self, board, num_simulations):
        """
        Run MCTS search with leaf batching.
        
        Args:
            board: chess.Board position
            num_simulations: total number of simulations to run
        """
        # Clear tree for new search (avoid stale moves)
        self.nodes.clear()
        self.children.clear()
        self.node_lookup.clear()
        
        # Initialize root
        root_hash = board.fen()
        if root_hash not in self.node_lookup:
            # Get initial evaluation
            pos, mask = encoder.encodePositionForInference(board)
            with torch.no_grad():
                pos_tensor = torch.from_numpy(pos).unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(mask).flatten().unsqueeze(0).to(self.device)
                value, policy = self.neural_network(pos_tensor, policyMask=mask_tensor)
            
            root_value = float(value[0, 0])
            policy_np = policy[0].cpu().numpy()
            root_priors = encoder.decodePolicyOutput(board, policy_np)
            
            # Create root
            root_idx = len(self.nodes)
            self.nodes.append(MCTSNode(root_hash, 1, root_value, 1.0, None, -1))
            self.node_lookup[root_hash] = root_idx
            self._expand_node(root_idx, board, root_priors)
        
        # Reset simulation counter
        with self.simulation_lock:
            self.total_simulations = 0
        
        # Run leaf collection workers
        start_time = time.time()
        workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._leaf_collector,
                args=(board.copy(), num_simulations)
            )
            worker.start()
            workers.append(worker)
        
        # Wait for completion
        for worker in workers:
            worker.join()
        
        # Ensure all leaves are processed
        with self.buffer_lock:
            # Force evaluation of any remaining leaves
            for buffer in self.leaf_buffers:
                if buffer['count'] > 0:
                    self.buffer_ready.notify()
        
        # Give GPU thread time to process final batch
        time.sleep(0.05)
        
        elapsed = time.time() - start_time
        if self.verbose:
            nps = num_simulations / elapsed
            print(f"Search complete: {num_simulations} nodes in {elapsed:.2f}s ({nps:.0f} NPS)")
        
        # Select best move
        root_idx = self.node_lookup[root_hash]
        return self._select_best_move(root_idx)
    
    def _leaf_collector(self, board, target_simulations):
        """Worker thread that collects leaf nodes"""
        while True:
            # Check if we've done enough simulations
            with self.simulation_lock:
                if self.total_simulations >= target_simulations:
                    break
                self.total_simulations += 1
            
            # Traverse to leaf
            path = []
            node_idx = self.node_lookup[board.fen()]
            sim_board = board.copy()
            
            # Selection phase with virtual loss
            while True:
                path.append(node_idx)
                
                # Apply virtual loss
                node = self.nodes[node_idx]
                self.nodes[node_idx] = MCTSNode(
                    node.board_hash, node.visits, node.total_value,
                    node.prior, node.move, node.parent_idx,
                    node.virtual_loss + 1
                )
                
                # Check if terminal
                if sim_board.is_game_over():
                    result = sim_board.result()
                    value = encoder.parseResult(result)
                    if not sim_board.turn:
                        value = -value
                    self._backup(path, value, remove_virtual_loss=True)
                    break
                
                # Check if leaf
                if node_idx not in self.children:
                    # Add to leaf buffer
                    with self.buffer_lock:
                        buffer = self.leaf_buffers[self.current_buffer]
                        buffer['boards'].append(sim_board.copy())
                        buffer['node_indices'].append(node_idx)
                        buffer['paths'].append(path.copy())
                        buffer['count'] += 1
                        
                        # Switch buffers if full
                        if buffer['count'] >= self.batch_size:
                            self.current_buffer = 1 - self.current_buffer
                            self.buffer_ready.notify()
                    break
                
                # Select child
                children = self.children[node_idx]
                if not children:
                    break
                
                best_child_idx = self._select_child_with_virtual_loss(children)
                best_child = self.nodes[best_child_idx]
                
                # Make move
                if best_child.move in sim_board.legal_moves:
                    sim_board.push(best_child.move)
                    node_idx = best_child_idx
                else:
                    print(f"Warning: Invalid move {best_child.move}")
                    break
        
        # Flush any remaining leaves
        with self.buffer_lock:
            if any(buf['count'] > 0 for buf in self.leaf_buffers):
                self.buffer_ready.notify()
    
    def _select_child_with_virtual_loss(self, children_indices):
        """Select best child using PUCT formula with virtual loss"""
        parent_visits = sum(
            self.nodes[idx].visits + self.nodes[idx].virtual_loss 
            for idx in children_indices
        )
        sqrt_parent = math.sqrt(parent_visits)
        
        best_idx = -1
        best_puct = -float('inf')
        
        for child_idx in children_indices:
            child = self.nodes[child_idx]
            
            # Calculate PUCT with virtual loss
            visits_with_vl = child.visits + child.virtual_loss
            if visits_with_vl > 0:
                q_value = child.total_value / visits_with_vl
            else:
                q_value = 0
            
            puct = q_value + self.c_puct * child.prior * sqrt_parent / (1 + visits_with_vl)
            
            if puct > best_puct:
                best_puct = puct
                best_idx = child_idx
        
        return best_idx
    
    def _expand_node(self, node_idx, board, move_probs):
        """Expand a node with children"""
        children_indices = []
        
        legal_moves = list(board.legal_moves)
        for i, move in enumerate(legal_moves):
            if i < len(move_probs) and move_probs[i] > 1e-8:
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
    
    def _backup(self, path, value, remove_virtual_loss=False):
        """Backup value through path"""
        for i, node_idx in enumerate(path):
            # Flip value for alternating players
            backup_value = value if i % 2 == 0 else -value
            
            # Update node
            node = self.nodes[node_idx]
            if remove_virtual_loss:
                new_vl = max(0, node.virtual_loss - 1)
            else:
                new_vl = node.virtual_loss
            
            self.nodes[node_idx] = MCTSNode(
                node.board_hash,
                node.visits + 1,
                node.total_value + backup_value,
                node.prior,
                node.move,
                node.parent_idx,
                new_vl
            )
    
    def _select_best_move(self, root_idx):
        """Select best move based on visit count"""
        if root_idx not in self.children:
            return None
        
        children = self.children[root_idx]
        if not children:
            return None
        
        # Select child with most visits
        best_visits = -1
        best_move = None
        
        for child_idx in children:
            child = self.nodes[child_idx]
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move
        
        return best_move