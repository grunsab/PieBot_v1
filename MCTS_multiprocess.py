"""
Multi-process MCTS implementation for AlphaZero chess engine.

This module provides a multi-process version of MCTS that can utilize
multiple CPU cores effectively by running tree search in parallel processes.
"""

import multiprocessing as mp
from multiprocessing import Queue, Process, Event, Manager
import numpy as np
import chess
import torch
import math
import time
import os
from collections import deque
import uuid

import encoder
from shared_tree import SharedTree, SharedTreeClient, SharedNode, SharedEdge
from inference_server import InferenceRequest, InferenceResult, start_inference_server, start_inference_server_from_state
import MCTS_profiling_speedups_v2 as MCTS  # For UCT calculation and other utilities


def run_worker(worker_id, tree_name, inference_queue, result_queue, stop_event,
               root_idx, root_board, num_rollouts):
    """
    Worker process entry point.
    
    Args:
        worker_id: Worker ID
        tree_name: Name of shared tree
        inference_queue: Queue for inference requests
        result_queue: Queue for receiving results
        stop_event: Event to signal shutdown
        root_idx: Root node index
        root_board: Root board position
        num_rollouts: Number of rollouts to perform
    """
    worker = MCTSWorker(worker_id, tree_name, inference_queue, result_queue, stop_event)
    
    # Create SharedTree instance for expansion
    shared_tree = SharedTreeClient(tree_name)
    
    try:
        worker.run(root_idx, root_board, shared_tree, num_rollouts)
    finally:
        shared_tree.cleanup()


class MCTSWorker:
    """Worker process for MCTS rollouts."""
    
    def __init__(self, worker_id, tree_name, inference_queue, result_queue, stop_event):
        """
        Initialize MCTS worker.
        
        Args:
            worker_id: Unique worker ID
            tree_name: Name of shared tree
            inference_queue: Queue for inference requests
            result_queue: Queue for receiving results
            stop_event: Event to signal shutdown
        """
        self.worker_id = worker_id
        self.tree_name = tree_name
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.tree_client = None
        
        # Statistics
        self.rollouts_completed = 0
        self.same_paths = 0
        
    def calc_uct(self, edge, parent_N):
        """Calculate UCT value for edge."""
        Q = edge.get_Q()
        N = edge.get_N()
        P = edge.get_P()
        
        # Handle NaN
        if math.isnan(P):
            P = 1.0 / 200.0
            
        C = 1.5
        UCT = Q + P * C * math.sqrt(parent_N) / (1 + N)
        
        if math.isnan(UCT):
            UCT = 0.0
            
        return UCT
    
    def get_edge_stats(self, edge):
        """Get edge statistics (N and Q)."""
        child_idx = edge.get_child_idx()
        virtual_loss = edge.get_virtual_loss()
        
        if child_idx >= 0:
            child = self.tree_client.get_node(child_idx)
            child_N = child.get_N()
            child_sum_Q = child.get_sum_Q()
            
            # Include virtual loss in calculations
            total_N = child_N + virtual_loss
            total_sum_Q = child_sum_Q + virtual_loss
            
            # Q from parent's perspective (1 - child's Q)
            if total_N > 0:
                Q = 1.0 - (total_sum_Q / total_N)
            else:
                Q = 0.0
                
            return total_N, Q
        else:
            return virtual_loss, 0.0
    
    def select_best_edge(self, node, edges_shm):
        """Select best edge using UCT."""
        edges = node.get_edges(edges_shm)
        if not edges:
            return None
            
        parent_N = node.get_N()
        best_edge = None
        best_uct = -float('inf')
        
        for edge in edges:
            # Calculate UCT with virtual loss
            N, Q = self.get_edge_stats(edge)
            P = edge.get_P()
            
            if math.isnan(P):
                P = 1.0 / 200.0
                
            C = 1.5
            UCT = Q + P * C * math.sqrt(parent_N) / (1 + N)
            
            if math.isnan(UCT):
                UCT = 0.0
                
            if UCT > best_uct:
                best_uct = UCT
                best_edge = edge
                
        return best_edge
    
    def select_path(self, root_idx, root_board, edges_shm, virtual_loss_scale=1.0):
        """
        Select path from root to leaf.
        
        Returns:
            board: Board at leaf position
            node_path: List of (node_idx, node) tuples
            edge_path: List of edges
            is_terminal: Whether leaf is terminal
        """
        board = root_board.copy()
        node_path = []
        edge_path = []
        
        current_idx = root_idx
        current_node = self.tree_client.get_node(current_idx)
        
        while True:
            node_path.append((current_idx, current_node))
            
            # Select best edge
            best_edge = self.select_best_edge(current_node, edges_shm)
            
            if best_edge is None:
                # Terminal node
                return board, node_path, edge_path, True
                
            edge_path.append(best_edge)
            
            # Add virtual loss
            best_edge.add_virtual_loss(virtual_loss_scale)
            
            # Make move
            move = best_edge.get_move()
            board.push(move)
            
            # Check if edge has child
            child_idx = best_edge.get_child_idx()
            if child_idx < 0:
                # Unexpanded node
                return board, node_path, edge_path, False
                
            # Move to child
            current_idx = child_idx
            current_node = self.tree_client.get_node(current_idx)
    
    def run_rollout(self, root_idx, root_board, shared_tree, edges_shm, virtual_loss_scale=1.0):
        """Execute one MCTS rollout."""
        # Selection phase
        board, node_path, edge_path, is_terminal = self.select_path(
            root_idx, root_board, edges_shm, virtual_loss_scale
        )
        
        if is_terminal:
            # Terminal node - evaluate based on game result
            winner = encoder.parseResult(board.result())
            if not board.turn:
                winner *= -1
            value = float(winner) / 2.0 + 0.5
        else:
            # Request neural network evaluation
            request_id = f"{self.worker_id}_{self.rollouts_completed}"
            request = InferenceRequest(request_id, board.fen(), self.worker_id)
            self.inference_queue.put((request, self.result_queue))
            
            # Wait for result
            result = self.result_queue.get()
            value = result.value / 2.0 + 0.5
            
            # Expand node if needed
            if len(edge_path) > 0:
                last_edge = edge_path[-1]
                expanded = shared_tree.expand_node(last_edge, board, value, result.move_probabilities)
                if not expanded:
                    self.same_paths += 1
                    
            # Flip value for backpropagation
            value = 1.0 - value
        
        # Backpropagation
        last_node_idx = len(node_path) - 1
        for i in range(last_node_idx, -1, -1):
            node_idx, node = node_path[i]
            is_from_child = (last_node_idx - i) % 2 == 1
            node.update_stats(value, is_from_child)
            
        # Clear virtual losses
        for edge in edge_path:
            edge.clear_virtual_loss()
            
        self.rollouts_completed += 1
    
    def run(self, root_idx, root_board, shared_tree, num_rollouts):
        """
        Main worker loop.
        
        Args:
            root_idx: Root node index
            root_board: Root board position
            shared_tree: SharedTree instance
            num_rollouts: Number of rollouts to perform
        """
        # Connect to shared tree
        self.tree_client = SharedTreeClient(self.tree_name)
        edges_shm = self.tree_client.edges_shm
        
        # Calculate virtual loss scale based on number of workers
        # This is a heuristic - more workers = higher virtual loss
        num_workers = mp.cpu_count() - 2  # Reserve cores for main and inference
        virtual_loss_scale = math.sqrt(num_workers / 10.0)
        
        try:
            for _ in range(num_rollouts):
                if self.stop_event.is_set():
                    break
                    
                self.run_rollout(root_idx, root_board, shared_tree, edges_shm, virtual_loss_scale)
                
        finally:
            self.tree_client.cleanup()


class MultiprocessMCTS:
    """Multi-process MCTS implementation."""
    
    def __init__(self, num_processes=None, inference_batch_size=64, inference_timeout_ms=5):
        """
        Initialize multi-process MCTS.
        
        Args:
            num_processes: Number of worker processes (None = auto)
            inference_batch_size: Batch size for inference
            inference_timeout_ms: Timeout for batching
        """
        if num_processes is None:
            # Use all cores minus 2 (one for main, one for inference)
            num_processes = max(1, mp.cpu_count() - 2)
            
        self.num_processes = num_processes
        self.inference_batch_size = inference_batch_size
        self.inference_timeout_ms = inference_timeout_ms
        
        # Manager for shared resources
        self.manager = Manager()
        
        # Shared resources
        self.shared_tree = None
        self.inference_queue = self.manager.Queue()
        self.result_queues = {}  # Will create per-worker result queues
        self.stop_event = self.manager.Event()
        self.processes = []
        
        # Statistics
        self.total_rollouts = 0
        self.total_time = 0.0
        
    def create_root(self, board, model):
        """Create root node and initialize tree."""
        # Clean up any existing tree
        if self.shared_tree:
            self.shared_tree.cleanup()
            
        # Create new shared tree
        tree_name = f"mcts_tree_{uuid.uuid4().hex[:8]}"
        self.shared_tree = SharedTree(tree_name)
        
        # Evaluate root position
        with torch.no_grad():
            value, move_probabilities = MCTS.callNeuralNetworkOptimized(board, model)
            
        # Create root node
        root_idx = self.shared_tree.create_root(board, value / 2.0 + 0.5, move_probabilities)
        
        return root_idx, tree_name
    
    def run_parallel_rollouts(self, board, model, num_rollouts):
        """
        Run parallel rollouts using multiple processes.
        
        Args:
            board: Current board position
            model: Neural network model
            num_rollouts: Total number of rollouts
        """
        start_time = time.time()
        
        # Create root and tree
        root_idx, tree_name = self.create_root(board, model)
        
        # Clear stop event
        self.stop_event.clear()
        
        # Get device for model
        device = next(model.parameters()).device
        
        # For MPS devices, we need to handle model sharing differently
        if device.type == 'mps':
            # Save model to CPU and reload in inference process
            model_state = model.cpu().state_dict()
            
            # Get model configuration
            if hasattr(self, 'model_config'):
                model_config = self.model_config
            else:
                # Try to infer from model
                try:
                    conv1_out_channels = model.conv1.out_channels
                    num_blocks = len([m for m in model.modules() if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
                    model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
                except:
                    model_config = {'num_blocks': 20, 'num_channels': 256}
            
            # Start inference server process with model state instead of model
            inference_process = Process(
                target=start_inference_server_from_state,
                args=(model_state, model_config, 'cpu', self.inference_queue, self.stop_event,
                      self.inference_batch_size, self.inference_timeout_ms)
            )
            
            # Move model back to MPS for root evaluation
            model = model.to(device)
        else:
            # For CUDA/CPU, we can pass the model directly
            inference_process = Process(
                target=start_inference_server,
                args=(model, device, self.inference_queue, self.stop_event,
                      self.inference_batch_size, self.inference_timeout_ms)
            )
        
        inference_process.start()
        self.processes.append(inference_process)
        
        # Calculate rollouts per worker
        rollouts_per_worker = num_rollouts // self.num_processes
        remainder = num_rollouts % self.num_processes
        
        # Start worker processes
        workers = []
        for i in range(self.num_processes):
            worker_rollouts = rollouts_per_worker
            if i < remainder:
                worker_rollouts += 1
            
            # Create a result queue for this worker
            result_queue = self.manager.Queue()
            self.result_queues[i] = result_queue
                
            # Start worker process
            process = Process(
                target=run_worker,
                args=(i, tree_name, self.inference_queue, result_queue, 
                      self.stop_event, root_idx, board, worker_rollouts)
            )
            process.start()
            self.processes.append(process)
            workers.append(process)
        
        # Wait for workers to complete
        for process in workers:
            process.join()
            
        # Stop inference server
        self.stop_event.set()
        inference_process.join()
        
        # Clear process list
        self.processes.clear()
        
        # Update statistics
        elapsed = time.time() - start_time
        self.total_rollouts += num_rollouts
        self.total_time += elapsed
        
        return root_idx
    
    def get_best_move(self, root_idx):
        """Get best move from root based on visit counts."""
        root = self.shared_tree.get_node(root_idx)
        edges = root.get_edges(self.shared_tree.edges_shm)
        
        best_edge = None
        best_N = -1
        
        for edge in edges:
            N, _ = self.get_edge_visit_count(edge)
            if N > best_N:
                best_N = N
                best_edge = edge
                
        if best_edge:
            return best_edge.get_move()
        return None
    
    def get_edge_visit_count(self, edge):
        """Get visit count for an edge."""
        child_idx = edge.get_child_idx()
        if child_idx >= 0:
            child = self.shared_tree.get_node(child_idx)
            return child.get_N(), child.get_Q()
        return 0, 0.0
    
    def get_statistics_string(self, root_idx):
        """Get statistics string for root position."""
        root = self.shared_tree.get_node(root_idx)
        edges = root.get_edges(self.shared_tree.edges_shm)
        
        # Sort edges by visit count
        edge_stats = []
        for edge in edges:
            move = edge.get_move()
            P = edge.get_P()
            N, Q = self.get_edge_visit_count(edge)
            edge_stats.append((move, P, N, Q))
            
        edge_stats.sort(key=lambda x: x[2], reverse=True)
        
        # Format string
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
            'move', 'P', 'N', 'Q'
        )
        
        for move, P, N, Q in edge_stats[:10]:  # Top 10 moves
            string += '|{: ^10}|{:10.4f}|{:10.0f}|{:10.4f}|\n'.format(
                str(move), P, N, Q
            )
            
        return string
    
    def get_root_q(self, root_idx):
        """Get Q value at root."""
        root = self.shared_tree.get_node(root_idx)
        return root.get_Q()
    
    def cleanup(self):
        """Clean up resources."""
        # Stop all processes
        self.stop_event.set()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join()
                
        # Clean up shared tree
        if self.shared_tree:
            self.shared_tree.cleanup()
            self.shared_tree = None
            
        # Clear queues
        try:
            while not self.inference_queue.empty():
                self.inference_queue.get_nowait()
        except:
            pass
            
        # Shutdown manager
        if hasattr(self, 'manager'):
            self.manager.shutdown()


# Compatibility class to match original MCTS interface
class Root:
    """Compatibility wrapper to match original MCTS Root interface."""
    
    def __init__(self, board, neuralNetwork):
        """Initialize root compatible with original MCTS."""
        self.board = board.copy()
        self.neuralNetwork = neuralNetwork
        self.mcts = MultiprocessMCTS()
        self.root_idx = None
        self.same_paths = 0
        
        # Extract model configuration for MPS compatibility
        # This assumes AlphaZeroNet structure - adjust if using different architecture
        try:
            # Try to infer model configuration from the model structure
            conv1_out_channels = neuralNetwork.conv1.out_channels
            num_blocks = len([m for m in neuralNetwork.modules() if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
            self.model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
        except:
            # Default configuration
            self.model_config = {'num_blocks': 20, 'num_channels': 256}
        
    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        """Run parallel rollouts (compatibility method)."""
        # Pass model config to MCTS if available
        if hasattr(self, 'model_config'):
            self.mcts.model_config = self.model_config
        self.root_idx = self.mcts.run_parallel_rollouts(
            self.board, self.neuralNetwork, num_parallel_rollouts
        )
        
    def maxNSelect(self):
        """Get edge with maximum visits (compatibility method)."""
        if self.root_idx is None:
            return None
            
        root = self.mcts.shared_tree.get_node(self.root_idx)
        edges = root.get_edges(self.mcts.shared_tree.edges_shm)
        
        best_edge = None
        best_N = -1
        
        for edge in edges:
            child_idx = edge.get_child_idx()
            if child_idx >= 0:
                child = self.mcts.shared_tree.get_node(child_idx)
                N = child.get_N()
                if N > best_N:
                    best_N = N
                    best_edge = edge
                    
        # Create a simple edge wrapper for compatibility
        if best_edge:
            class EdgeWrapper:
                def __init__(self, move, N, Q):
                    self.move = move
                    self.N = N
                    self.Q = Q
                def getMove(self):
                    return self.move
                def getN(self):
                    return self.N
                def getQ(self):
                    return self.Q
                    
            child = self.mcts.shared_tree.get_node(best_edge.get_child_idx())
            return EdgeWrapper(best_edge.get_move(), child.get_N(), child.get_Q())
            
        return None
    
    def getQ(self):
        """Get Q value at root (compatibility method)."""
        if self.root_idx is None:
            return 0.5
        return self.mcts.get_root_q(self.root_idx)
    
    def getStatisticsString(self):
        """Get statistics string (compatibility method)."""
        if self.root_idx is None:
            return ""
        return self.mcts.get_statistics_string(self.root_idx)
    
    def cleanup(self):
        """Clean up resources."""
        self.mcts.cleanup()


# Module-level functions for compatibility
def clear_caches():
    """Clear caches (compatibility function)."""
    pass  # Caches are per-process in multiprocess version

def clear_pools():
    """Clear object pools (compatibility function)."""
    pass  # Not used in multiprocess version