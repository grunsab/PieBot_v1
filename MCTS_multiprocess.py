"""
High-performance, multi-process MCTS implementation for AlphaZero chess engine.

This module provides a multi-process version of MCTS that utilizes
multiple CPU cores effectively by running tree search in parallel, persistent
worker processes.
"""

import multiprocessing as mp
from multiprocessing import Process, Event, Lock, Manager
import numpy as np
import chess
import torch
import math
import time
import os
from collections import deque
import uuid
import queue

import encoder
from shared_tree import SharedTree, SharedTreeClient
from inference_server import InferenceRequest, InferenceResult, start_inference_server, start_inference_server_from_state
import MCTS_profiling_speedups_v2 as MCTS  # For UCT calculation and other utilities

# A task for a worker to perform a set of rollouts
class RolloutTask:
    def __init__(self, task_id, tree_name, root_idx, root_board, num_rollouts):
        self.task_id = task_id
        self.tree_name = tree_name
        self.root_idx = root_idx
        self.root_board = root_board
        self.num_rollouts = num_rollouts

# A result message from a worker
class WorkerResult:
    def __init__(self, worker_id, task_id, rollouts_completed, same_paths):
        self.worker_id = worker_id
        self.task_id = task_id
        self.rollouts_completed = rollouts_completed
        self.same_paths = same_paths

def run_worker(worker_id, task_queue, completion_queue, inference_queue, stop_event, alloc_lock):
    """
    Worker process entry point. Loops indefinitely, waiting for rollout tasks.
    """
    worker = MCTSWorker(worker_id, inference_queue, stop_event)
    
    while not stop_event.is_set():
        try:
            # Wait for a task. Using a timeout allows checking the stop_event periodically.
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Create a client for the specific shared tree for this task
        shared_tree_client = SharedTreeClient(task.tree_name)
        shared_tree_client.alloc_lock = alloc_lock

        try:
            rollouts_completed, same_paths = worker.run_task(task, shared_tree_client)
            # Report completion
            completion_queue.put(WorkerResult(worker_id, task.task_id, rollouts_completed, same_paths))
        finally:
            # Clean up the client connection to the shared memory
            shared_tree_client.cleanup()


class MCTSWorker:
    """Worker process for MCTS rollouts."""
    
    def __init__(self, worker_id, inference_queue, stop_event):
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.stop_event = stop_event
        self.result_queue = Manager().Queue() # Each worker has its own result queue from inference
        
    def calc_uct(self, edge, parent_N):
        """Calculate UCT value for edge."""
        Q = edge.get_Q()
        N = edge.get_N()
        P = edge.get_P()
        
        if math.isnan(P):
            P = 1.0 / 200.0
            
        C = 1.5
        UCT = Q + P * C * math.sqrt(parent_N) / (1 + N)
        
        if math.isnan(UCT):
            UCT = 0.0
            
        return UCT
    
    def get_edge_stats(self, edge, tree_client):
        """Get edge statistics (N and Q)."""
        child_idx = edge.get_child_idx()
        virtual_loss = edge.get_virtual_loss()
        
        if child_idx >= 0:
            child = tree_client.get_node(child_idx)
            child_N = child.get_N()
            child_sum_Q = child.get_sum_Q()
            
            total_N = child_N + virtual_loss
            # Q from parent's perspective (1 - child's Q)
            if total_N > 0:
                # The value of a node is from the perspective of the player whose turn it is.
                # The Q value for an edge should be from the perspective of the parent node.
                # If it's the child's turn, the parent sees the value as 1 - child_value.
                Q = 1.0 - (child.get_sum_Q() / total_N)
            else:
                Q = 0.0
                
            return total_N, Q
        else:
            return virtual_loss, 0.0
    
    def select_best_edge(self, node, tree_client):
        """Select best edge using UCT."""
        edges = node.get_edges(tree_client.edges_shm)
        if not edges:
            return None
            
        parent_N = node.get_N()
        best_edge = None
        best_uct = -float('inf')
        
        for edge in edges:
            N, Q = self.get_edge_stats(edge, tree_client)
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
    
    def select_path(self, root_idx, root_board, tree_client, virtual_loss_scale=1.0):
        """Select path from root to leaf."""
        board = root_board.copy()
        node_path = []
        edge_path = []
        
        current_idx = root_idx
        
        while True:
            current_node = tree_client.get_node(current_idx)
            node_path.append((current_idx, current_node))
            
            if current_node.is_terminal():
                return board, node_path, edge_path, True

            best_edge = self.select_best_edge(current_node, tree_client)
            
            if best_edge is None:
                # Should not happen for non-terminal nodes, but handle defensively
                return board, node_path, edge_path, True
                
            edge_path.append(best_edge)
            best_edge.add_virtual_loss(virtual_loss_scale)
            
            move = best_edge.get_move()
            board.push(move)
            
            child_idx = best_edge.get_child_idx()
            if child_idx < 0:
                return board, node_path, edge_path, False
                
            current_idx = child_idx
    
    def run_rollout(self, root_idx, root_board, shared_tree_client, virtual_loss_scale=1.0):
        """Execute one MCTS rollout."""
        # Selection phase
        board, node_path, edge_path, is_terminal = self.select_path(
            root_idx, root_board, shared_tree_client, virtual_loss_scale
        )
        
        leaf_node = node_path[-1][1]

        if is_terminal:
            result_str = board.result(claim_draw=True)
            winner = encoder.parseResult(result_str)
            # Value is from the perspective of the current player at the leaf
            value = float(winner)
        else:
            # Request neural network evaluation
            request_id = f"{self.worker_id}_{uuid.uuid4().hex[:8]}"
            request = InferenceRequest(request_id, board.fen(), self.worker_id)
            self.inference_queue.put((request, self.result_queue))
            
            # Wait for result
            result = self.result_queue.get()
            value = result.value
            
            # Expand node if needed
            last_edge = edge_path[-1] if edge_path else None
            if last_edge and last_edge.get_child_idx() < 0:
                expanded = shared_tree_client.expand_node(last_edge, board, value, result.move_probabilities)
                if not expanded:
                    # Another worker expanded this node first.
                    # We still use the NN eval for backpropagation.
                    pass
        
        # Backpropagation
        # The value is from the perspective of the player at the leaf node.
        # We need to flip the value at each step up the tree.
        current_value = value
        for i in range(len(node_path) - 1, -1, -1):
            node_idx, node = node_path[i]
            node.update_stats(current_value)
            current_value *= -1
            
        # Clear virtual losses
        for edge in edge_path:
            edge.clear_virtual_loss()
            
    def run_task(self, task, shared_tree_client):
        """
        Run a specific rollout task.
        """
        num_workers = mp.cpu_count() - 2
        virtual_loss_scale = math.sqrt(num_workers / 10.0) if num_workers > 0 else 1.0
        
        rollouts_completed = 0
        for _ in range(task.num_rollouts):
            if self.stop_event.is_set():
                break
            self.run_rollout(task.root_idx, task.root_board, shared_tree_client, virtual_loss_scale)
            rollouts_completed += 1
        
        return rollouts_completed, 0 # same_paths not tracked anymore


class MultiprocessMCTS:
    """Persistent Multi-process MCTS implementation."""
    
    def __init__(self, model, num_processes=None, inference_batch_size=64, inference_timeout_ms=5):
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 2)
            
        self.num_processes = num_processes
        self.inference_batch_size = inference_batch_size
        self.inference_timeout_ms = inference_timeout_ms
        self.model = model
        
        # Use a manager for creating shared objects like queues
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.completion_queue = self.manager.Queue()
        self.inference_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        self.alloc_lock = self.manager.Lock()
        
        self.processes = []
        self.shared_tree = None
        
        self.start_persistent_processes()
        
    def start_persistent_processes(self):
        """Starts the inference server and worker processes."""
        # Start Inference Server
        device = next(self.model.parameters()).device
        if device.type == 'mps':
            model_state = self.model.cpu().state_dict()
            try:
                conv1_out_channels = self.model.conv1.out_channels
                num_blocks = len([m for m in self.model.modules() if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
                model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
            except:
                model_config = {'num_blocks': 20, 'num_channels': 256}
            
            inference_process = Process(
                target=start_inference_server_from_state,
                args=(model_state, model_config, 'cpu', self.inference_queue, self.stop_event,
                      self.inference_batch_size, self.inference_timeout_ms)
            )
        else:
            inference_process = Process(
                target=start_inference_server,
                args=(self.model, device, self.inference_queue, self.stop_event,
                      self.inference_batch_size, self.inference_timeout_ms)
            )
        
        inference_process.start()
        self.processes.append(inference_process)
        
        # Start Worker Processes
        for i in range(self.num_processes):
            process = Process(
                target=run_worker,
                args=(i, self.task_queue, self.completion_queue, self.inference_queue, 
                      self.stop_event, self.alloc_lock)
            )
            process.start()
            self.processes.append(process)
            
    def run_parallel_rollouts(self, board, num_rollouts):
        """
        Run parallel rollouts using the persistent worker pool.
        """
        # 1. Create a new shared tree for this search
        if self.shared_tree:
            self.shared_tree.cleanup()
        
        tree_name = f"mcts_tree_{uuid.uuid4().hex[:8]}"
        self.shared_tree = SharedTree(tree_name)
        self.shared_tree.alloc_lock = self.alloc_lock
        
        # 2. Evaluate root position and create root node
        with torch.no_grad():
            value, move_probabilities = MCTS.callNeuralNetworkOptimized(board, self.model)
        
        root_idx = self.shared_tree.create_root(board, value, move_probabilities)
        
        # 3. Distribute rollouts among workers
        task_id = uuid.uuid4().hex
        rollouts_per_worker = num_rollouts // self.num_processes
        remainder = num_rollouts % self.num_processes
        
        for i in range(self.num_processes):
            worker_rollouts = rollouts_per_worker
            if i < remainder:
                worker_rollouts += 1
            if worker_rollouts > 0:
                task = RolloutTask(task_id, tree_name, root_idx, board, worker_rollouts)
                self.task_queue.put(task)
        
        # 4. Wait for all workers to complete this task
        completed_workers = 0
        total_rollouts_done = 0
        while completed_workers < self.num_processes:
            result = self.completion_queue.get()
            if result.task_id == task_id:
                completed_workers += 1
                total_rollouts_done += result.rollouts_completed
        
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
        """Get visit count and Q value for an edge."""
        child_idx = edge.get_child_idx()
        if child_idx >= 0:
            child = self.shared_tree.get_node(child_idx)
            # Q is from the parent's perspective, so it's 1 - child's Q
            q_value = 1.0 - child.get_Q() if child.get_N() > 0 else 0.0
            return child.get_N(), q_value
        return 0, 0.0
    
    def get_statistics_string(self, root_idx):
        """Get statistics string for root position."""
        root = self.shared_tree.get_node(root_idx)
        edges = root.get_edges(self.shared_tree.edges_shm)
        
        edge_stats = []
        for edge in edges:
            move = edge.get_move()
            P = edge.get_P()
            N, Q = self.get_edge_visit_count(edge)
            edge_stats.append((move, P, N, Q))
            
        edge_stats.sort(key=lambda x: x[2], reverse=True)
        
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format('move', 'P', 'N', 'Q')
        for move, P, N, Q in edge_stats[:10]:
            string += '|{: ^10}|{:10.4f}|{:10.0f}|{:10.4f}|\n'.format(str(move), P, N, Q)
        return string
    
    def get_root_q(self, root_idx):
        """Get Q value at root."""
        root = self.shared_tree.get_node(root_idx)
        return root.get_Q()
    
    def cleanup(self):
        """Clean up resources by stopping all persistent processes."""
        self.stop_event.set()
        time.sleep(0.1) # Give processes time to see the event

        # Empty the queues to prevent deadlocks
        for q in [self.task_queue, self.completion_queue, self.inference_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        for p in self.processes:
            if p.is_alive():
                p.terminate() # Force terminate if it doesn't exit gracefully
            p.join()
            
        if self.shared_tree:
            self.shared_tree.cleanup()
            self.shared_tree = None

# Compatibility class to match original MCTS interface
class Root:
    """Compatibility wrapper for the persistent multi-process MCTS engine."""
    
    _mcts_engine = None
    _mcts_lock = Lock()

    def __init__(self, board, neuralNetwork):
        self.board = board.copy()
        self.neuralNetwork = neuralNetwork
        self.root_idx = None
        
        with Root._mcts_lock:
            if Root._mcts_engine is None:
                Root._mcts_engine = MultiprocessMCTS(neuralNetwork)

    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        """Run parallel rollouts (compatibility method)."""
        # The board and neural network are now mainly for API compatibility.
        # The engine uses the model it was initialized with.
        self.root_idx = Root._mcts_engine.run_parallel_rollouts(
            self.board, num_parallel_rollouts
        )
        
    def maxNSelect(self):
        """Get edge with maximum visits (compatibility method)."""
        if self.root_idx is None:
            return None
            
        best_move = Root._mcts_engine.get_best_move(self.root_idx)
        if not best_move:
            return None

        # For compatibility, we need to return an object with getMove(), getN(), getQ()
        class EdgeWrapper:
            def __init__(self, move, N, Q):
                self.move = move
                self.N = N
                self.Q = Q
            def getMove(self): return self.move
            def getN(self): return self.N
            def getQ(self): return self.Q

        root = Root._mcts_engine.shared_tree.get_node(self.root_idx)
        edges = root.get_edges(Root._mcts_engine.shared_tree.edges_shm)
        for edge in edges:
            if edge.get_move() == best_move:
                N, Q = Root._mcts_engine.get_edge_visit_count(edge)
                return EdgeWrapper(best_move, N, Q)
        return None
    
    def getQ(self):
        """Get Q value at root (compatibility method)."""
        if self.root_idx is None:
            return 0.5
        return Root._mcts_engine.get_root_q(self.root_idx)
    
    def getStatisticsString(self):
        """Get statistics string (compatibility method)."""
        if self.root_idx is None:
            return ""
        return Root._mcts_engine.get_statistics_string(self.root_idx)
    
    @staticmethod
    def cleanup_engine():
        """Static method to clean up the shared MCTS engine."""
        with Root._mcts_lock:
            if Root._mcts_engine:
                Root._mcts_engine.cleanup()
                Root._mcts_engine = None

# Module-level functions for compatibility
def clear_caches():
    pass

def clear_pools():
    pass

# It's good practice to ensure cleanup happens on exit
import atexit
atexit.register(Root.cleanup_engine)
