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
                    if i >= self.batch_size:  # Safety check
                        break
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
                # Convert neural network value to win probability
                value = float(values_np[i, 0]) / 2.0 + 0.5
                policy = policies_np[i]
                
                # Decode policy
                move_probs = encoder.decodePolicyOutput(board, policy)
                
                # Expand node (check if node still exists)
                if node_idx < len(self.nodes):
                    # Just expand the node - don't modify its statistics
                    # The node was already visited during selection
                    self._expand_node(node_idx, board, move_probs)
                    
                    # For an expanded node, we backup the value from the neural network
                    # Note: the value is from the perspective of the expanded position
                    # So we backup this value to all nodes in the path
                    self._backup(path, value, remove_virtual_loss=True)
    
    def search(self, board, num_simulations):
        """
        Run MCTS search with leaf batching.
        
        Args:
            board: chess.Board position
            num_simulations: total number of simulations to run
        """
        # Wait for any pending GPU operations to complete
        with self.buffer_lock:
            # Clear all buffers
            for buffer in self.leaf_buffers:
                buffer['count'] = 0
                buffer['boards'].clear()
                buffer['node_indices'].clear()
                buffer['paths'].clear()
        
        # Check if we can reuse existing tree
        root_hash = board.fen()
        if root_hash in self.node_lookup:
            # Root already exists in tree - reuse it
            root_idx = self.node_lookup[root_hash]
            if self.verbose:
                print(f"Reusing existing tree with root at index {root_idx}")
        else:
            # Need to create new root
            # Clear tree only if we can't reuse
            self.nodes.clear()
            self.children.clear()
            self.node_lookup.clear()
            
            # Get initial evaluation
            pos, mask = encoder.encodePositionForInference(board)
            with torch.no_grad():
                pos_tensor = torch.from_numpy(pos).unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(mask).flatten().unsqueeze(0).to(self.device)
                value, policy = self.neural_network(pos_tensor, policyMask=mask_tensor)
            
            # Convert neural network value to win probability
            root_value = float(value[0, 0]) / 2.0 + 0.5
            policy_np = policy[0].cpu().numpy()
            root_priors = encoder.decodePolicyOutput(board, policy_np)
            
            # Create root (matching original MCTS initialization)
            root_idx = len(self.nodes)
            # Root starts with N=1 and sum_Q=root_value (already converted to win probability)
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
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        if self.verbose:
            nps = num_simulations / elapsed
            print(f"Search complete: {num_simulations} nodes in {elapsed:.2f}s ({nps:.0f} NPS)")
        
        # Select best move
        root_idx = self.node_lookup[root_hash]
        return self._select_best_move(root_idx, temperature=0)  # Use temperature=0 for strong play
    
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
            sim_board = board.copy()
            root_fen = sim_board.fen()
            
            # Get root node index
            if root_fen not in self.node_lookup:
                # Race condition - tree was cleared, skip this simulation
                continue
            node_idx = self.node_lookup[root_fen]
            
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
                    # Convert to win probability
                    value = value / 2.0 + 0.5
                    self._backup(path, value, remove_virtual_loss=True)
                    break
                
                # Check if leaf
                if node_idx not in self.children:
                    # Add to leaf buffer
                    with self.buffer_lock:
                        buffer = self.leaf_buffers[self.current_buffer]
                        
                        # Check if buffer is full
                        if buffer['count'] >= self.batch_size:
                            # Switch to other buffer
                            self.current_buffer = 1 - self.current_buffer
                            self.buffer_ready.notify()
                            buffer = self.leaf_buffers[self.current_buffer]
                        
                        # Add to buffer if there's space
                        if buffer['count'] < self.batch_size:
                            buffer['boards'].append(sim_board.copy())
                            buffer['node_indices'].append(node_idx)
                            buffer['paths'].append(path.copy())
                            buffer['count'] += 1
                    break
                
                # Select child
                children = self.children[node_idx]
                if not children:
                    break
                
                best_child_idx = self._select_child_with_virtual_loss(children)
                if best_child_idx < 0 or best_child_idx >= len(self.nodes):
                    # Invalid child index
                    break
                best_child = self.nodes[best_child_idx]
                
                # Make move
                if best_child.move in sim_board.legal_moves:
                    sim_board.push(best_child.move)
                    node_idx = best_child_idx
                else:
                    # Move is invalid - this can happen during tree transitions
                    # or when the tree is being rebuilt
                    break
        
        # Flush any remaining leaves
        with self.buffer_lock:
            if any(buf['count'] > 0 for buf in self.leaf_buffers):
                self.buffer_ready.notify()
    
    def _select_child_with_virtual_loss(self, children_indices):
        """Select best child using PUCT formula with virtual loss"""
        valid_children = [idx for idx in children_indices if 0 <= idx < len(self.nodes)]
        if not valid_children:
            return -1
        
        # Get parent node to calculate parent visits correctly
        if valid_children:
            first_child = self.nodes[valid_children[0]]
            if first_child.parent_idx >= 0:
                parent_node = self.nodes[first_child.parent_idx]
                parent_visits = parent_node.visits + parent_node.virtual_loss
            else:
                # Fallback for root
                parent_visits = sum(
                    self.nodes[idx].visits + self.nodes[idx].virtual_loss 
                    for idx in valid_children
                )
        else:
            parent_visits = 1
            
        sqrt_parent = math.sqrt(parent_visits) if parent_visits > 0 else 1.0
        
        best_idx = -1
        best_puct = -float('inf')
        
        for child_idx in valid_children:
            child = self.nodes[child_idx]
            
            # Calculate PUCT with virtual loss
            visits_with_vl = child.visits + child.virtual_loss
            if visits_with_vl > 0:
                # Fix Q-value calculation: flip for opponent's perspective
                # Child stores value from its perspective, we need parent's perspective
                q_value = 1.0 - (child.total_value / visits_with_vl)
            else:
                # Unvisited nodes get a neutral Q-value
                # This matches original MCTS where getQ() returns 0 for unvisited edges
                q_value = 0.0
            
            puct = q_value + self.c_puct * child.prior * sqrt_parent / (1 + visits_with_vl)
            
            if puct > best_puct:
                best_puct = puct
                best_idx = child_idx
        
        return best_idx
    
    def get_visit_counts(self, board):
        """Get visit counts for all moves from current root position"""
        import numpy as np
        
        root_hash = board.fen()
        if root_hash not in self.node_lookup:
            return None
            
        root_idx = self.node_lookup[root_hash]
        if root_idx not in self.children:
            return None
            
        # Create visit count array indexed by move
        visit_counts = np.zeros(4608, dtype=np.float32)
        
        children = self.children[root_idx]
        for child_idx in children:
            child = self.nodes[child_idx]
            move = child.move
            
            # Encode move to get index
            if not board.turn:
                # For black, we need to mirror the move
                from encoder import mirrorMove
                mirrored_move = mirrorMove(move)
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(mirrored_move)
            else:
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
            
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            visit_counts[moveIdx] = child.visits
            
        return visit_counts
    
    def _expand_node(self, node_idx, board, move_probs):
        """Expand a node with children"""
        children_indices = []
        
        legal_moves = list(board.legal_moves)
        # Default prior for moves not in policy output
        default_prior = 1.0 / max(len(legal_moves), 200)
        
        for i, move in enumerate(legal_moves):
            # Get move probability - use exact value from neural network
            if i < len(move_probs):
                prior = float(move_probs[i])
                # Handle NaN or negative values (matching original MCTS)
                if math.isnan(prior) or prior < 0:
                    prior = default_prior
            else:
                # Move index beyond policy output - shouldn't happen
                prior = default_prior
            
            # Create child node for ALL legal moves
            board.push(move)
            child_hash = board.fen()
            board.pop()
            
            # Check if already exists
            if child_hash in self.node_lookup:
                child_idx = self.node_lookup[child_hash]
            else:
                child_idx = len(self.nodes)
                # Initialize with visits=0 (will be incremented when first visited)
                # This matches original MCTS where edges start unvisited
                child_node = MCTSNode(
                    child_hash, 0, 0.0,  # Unvisited node
                    prior, move, node_idx
                )
                self.nodes.append(child_node)
                self.node_lookup[child_hash] = child_idx
            
            children_indices.append(child_idx)
        
        self.children[node_idx] = children_indices
    
    def _backup(self, path, value, remove_virtual_loss=False):
        """Backup value through path"""
        # value is from the perspective of the last node in path
        # We need to flip it as we go up the tree
        
        for i in range(len(path) - 1, -1, -1):
            node_idx = path[i]
            
            # Check if node index is still valid
            if node_idx < 0 or node_idx >= len(self.nodes):
                # Node was removed or index is invalid, skip
                continue
            
            # Calculate backup value
            # The value alternates as we go up the tree
            # If we're at an even distance from the leaf, use the value as-is
            # If we're at an odd distance, flip it
            distance_from_leaf = len(path) - 1 - i
            if distance_from_leaf % 2 == 0:
                backup_value = value
            else:
                backup_value = 1.0 - value
            
            # Update node
            try:
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
            except IndexError:
                # Race condition - node was removed, skip
                continue
    
    def _select_best_move(self, root_idx, temperature=0):
        """Select best move based on visit count with optional temperature"""
        if root_idx not in self.children:
            return None
        
        children = self.children[root_idx]
        if not children:
            return None
        
        if temperature == 0:
            # Select child with most visits (deterministic)
            best_visits = -1
            best_move = None
            
            for child_idx in children:
                child = self.nodes[child_idx]
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_move = child.move
            
            return best_move
        else:
            # Temperature-based selection
            import numpy as np
            
            visits = []
            moves = []
            for child_idx in children:
                child = self.nodes[child_idx]
                visits.append(child.visits)
                moves.append(child.move)
            
            visits = np.array(visits, dtype=np.float32)
            # Apply temperature
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
            
            # Sample from probability distribution
            selected_idx = np.random.choice(len(moves), p=probs)
            return moves[selected_idx]