"""
Root Parallelization MCTS Implementation for AlphaZero Chess Engine.

This module implements root parallelization where each worker builds an independent
search tree from the same root position, with Dirichlet noise added at the root
for exploration diversity. This approach is more effective than tree parallelization
as it avoids synchronization overhead and provides better exploration.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
import numpy as np
import chess
import torch
import math
import time
import uuid
import queue
from collections import defaultdict

import encoder
from inference_server import InferenceRequest, InferenceResult, start_inference_server, start_inference_server_from_state
import MCTS  # Use the original single-threaded MCTS for node/edge structures


class WorkerTask:
    """Task for a worker to perform independent MCTS search."""
    def __init__(self, task_id, board, num_rollouts, noise_seed, epsilon, alpha):
        self.task_id = task_id
        self.board = board
        self.num_rollouts = num_rollouts
        self.noise_seed = noise_seed
        self.epsilon = epsilon  # Weight for Dirichlet noise
        self.alpha = alpha      # Dirichlet concentration parameter


class WorkerResult:
    """Result from a worker's independent search."""
    def __init__(self, worker_id, task_id, move_visits, move_q_values):
        self.worker_id = worker_id
        self.task_id = task_id
        self.move_visits = move_visits  # Dict: move -> visit count
        self.move_q_values = move_q_values  # Dict: move -> Q value


def apply_dirichlet_noise(move_probabilities, legal_moves, epsilon, alpha, rng):
    """
    Apply Dirichlet noise to move probabilities at the root.
    
    Args:
        move_probabilities: Original probabilities from neural network
        legal_moves: List of legal moves
        epsilon: Weight for Dirichlet noise (0.25 typical)
        alpha: Dirichlet concentration parameter (0.3 for chess)
        rng: Random number generator (numpy RandomState)
    
    Returns:
        Modified move probabilities with Dirichlet noise
    """
    num_moves = len(legal_moves)
    if num_moves == 0 or epsilon == 0:
        return move_probabilities
    
    # Generate Dirichlet noise
    noise = rng.dirichlet([alpha] * num_moves)
    
    # Apply noise to probabilities
    noisy_probs = move_probabilities.copy()
    for i, move in enumerate(legal_moves):
        original_p = move_probabilities[i]
        noisy_probs[i] = (1 - epsilon) * original_p + epsilon * noise[i]
    
    return noisy_probs


def run_worker(worker_id, task_queue, result_queue, inference_queue, stop_event):
    """
    Worker process that builds its own independent MCTS tree.
    """
    manager = Manager()  # Create manager instance once
    
    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Initialize random generator with unique seed
        rng = np.random.RandomState(task.noise_seed)
        
        # Create result queue for this worker's inference requests
        worker_result_queue = manager.Queue()
        
        # Build independent tree with Dirichlet noise at root
        tree_stats = run_independent_search(
            worker_id, task.board, task.num_rollouts, 
            task.epsilon, task.alpha, rng,
            inference_queue, worker_result_queue
        )
        
        # Extract move statistics
        move_visits = {}
        move_q_values = {}
        for move, stats in tree_stats.items():
            move_visits[move] = stats['visits']
            move_q_values[move] = stats['q_value']
        
        result = WorkerResult(worker_id, task.task_id, move_visits, move_q_values)
        result_queue.put(result)


def run_independent_search(worker_id, board, num_rollouts, epsilon, alpha, rng, 
                          inference_queue, result_queue):
    """
    Run independent MCTS search with Dirichlet noise at root.
    
    Returns:
        Dict mapping moves to their statistics (visits, Q-value)
    """
    # Get initial neural network evaluation
    request_id = f"{worker_id}_root_{uuid.uuid4().hex[:8]}"
    request = InferenceRequest(request_id, board.fen(), worker_id)
    inference_queue.put((request, result_queue))
    
    result = result_queue.get()
    value = result.value
    move_probabilities = result.move_probabilities
    
    # Apply Dirichlet noise at root for this worker
    legal_moves = list(board.legal_moves)
    if epsilon > 0 and len(legal_moves) > 0:
        move_probabilities = apply_dirichlet_noise(
            move_probabilities, legal_moves, epsilon, alpha, rng
        )
    
    # Create root node with noisy probabilities
    Q = value / 2.0 + 0.5
    root = MCTS.Node(board, Q, move_probabilities)
    
    # Run rollouts independently
    for _ in range(num_rollouts):
        rollout_board = board.copy()
        run_single_rollout(root, rollout_board, worker_id, inference_queue, result_queue)
    
    # Extract statistics from root
    stats = {}
    for edge in root.edges:
        move = edge.getMove()
        stats[move] = {
            'visits': edge.getN(),
            'q_value': edge.getQ()
        }
    
    return stats


def run_single_rollout(root, board, worker_id, inference_queue, result_queue):
    """
    Execute a single MCTS rollout on the independent tree.
    """
    node_path = []
    edge_path = []
    current_node = root
    
    # Selection phase
    while True:
        node_path.append(current_node)
        
        if current_node.isTerminal():
            edge_path.append(None)
            break
        
        # Select best edge using UCT
        edge = current_node.UCTSelect()
        edge_path.append(edge)
        
        if edge is None:
            break
        
        board.push(edge.getMove())
        
        if not edge.has_child():
            # Expand this node
            break
        
        current_node = edge.getChild()
    
    # Evaluation phase
    if edge_path[-1] is not None:
        # Neural network evaluation
        request_id = f"{worker_id}_{uuid.uuid4().hex[:8]}"
        request = InferenceRequest(request_id, board.fen(), worker_id)
        inference_queue.put((request, result_queue))
        
        result = result_queue.get()
        value = result.value
        new_Q = value / 2.0 + 0.5
        
        # Expand node
        edge_path[-1].expand(board, new_Q, result.move_probabilities)
        new_Q = 1.0 - new_Q
    else:
        # Terminal node
        winner = encoder.parseResult(board.result(claim_draw=True))
        if not board.turn:
            winner *= -1
        new_Q = float(winner) / 2.0 + 0.5
    
    # Backpropagation phase
    last_node_idx = len(node_path) - 1
    for i in range(last_node_idx, -1, -1):
        node = node_path[i]
        
        if (last_node_idx - i) % 2 == 0:
            # Same perspective as leaf
            node.N += 1
            node.sum_Q += new_Q
        else:
            # Opposite perspective
            node.N += 1
            node.sum_Q += 1.0 - new_Q


class RootParallelMCTS:
    """
    Root parallelization MCTS with Dirichlet noise for exploration diversity.
    """
    
    def __init__(self, model, num_workers=None, epsilon=0.25, alpha=0.3,
                 inference_batch_size=64, inference_timeout_ms=5):
        """
        Initialize root parallel MCTS.
        
        Args:
            model: Neural network model
            num_workers: Number of parallel workers (default: CPU count - 2)
            epsilon: Weight for Dirichlet noise (0.25 for training, 0.0 for play)
            alpha: Dirichlet concentration (0.3 for chess)
            inference_batch_size: Batch size for neural network inference
            inference_timeout_ms: Timeout for batching inference requests
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)
        
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.alpha = alpha
        self.inference_batch_size = inference_batch_size
        self.inference_timeout_ms = inference_timeout_ms
        self.model = model
        
        # Process management
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.inference_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        
        self.processes = []
        self.start_processes()
    
    def start_processes(self):
        """Start inference server and worker processes."""
        # Start inference server
        device = next(self.model.parameters()).device
        
        if device.type == 'mps':
            # Apple Silicon compatibility
            model_state = self.model.cpu().state_dict()
            try:
                conv1_out_channels = self.model.conv1.out_channels
                num_blocks = len([m for m in self.model.modules() 
                                if hasattr(m, 'conv1') and hasattr(m, 'conv2')]) // 2
                model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
            except:
                model_config = {'num_blocks': 20, 'num_channels': 256}
            
            inference_process = Process(
                target=start_inference_server_from_state,
                args=(model_state, model_config, 'cpu', self.inference_queue, 
                      self.stop_event, self.inference_batch_size, self.inference_timeout_ms)
            )
        else:
            inference_process = Process(
                target=start_inference_server,
                args=(self.model, device, self.inference_queue, self.stop_event,
                      self.inference_batch_size, self.inference_timeout_ms)
            )
        
        inference_process.start()
        self.processes.append(inference_process)
        
        # Start worker processes
        for i in range(self.num_workers):
            process = Process(
                target=run_worker,
                args=(i, self.task_queue, self.result_queue, 
                      self.inference_queue, self.stop_event)
            )
            process.start()
            self.processes.append(process)
    
    def run_parallel_search(self, board, num_rollouts):
        """
        Run parallel search with independent trees.
        
        Args:
            board: Chess board position
            num_rollouts: Total number of rollouts to perform
            
        Returns:
            Aggregated move statistics
        """
        task_id = uuid.uuid4().hex
        rollouts_per_worker = num_rollouts // self.num_workers
        remainder = num_rollouts % self.num_workers
        
        # Create tasks with unique noise seeds for each worker
        for i in range(self.num_workers):
            worker_rollouts = rollouts_per_worker
            if i < remainder:
                worker_rollouts += 1
            
            if worker_rollouts > 0:
                # Each worker gets a unique random seed for Dirichlet noise
                noise_seed = np.random.randint(0, 2**32)
                task = WorkerTask(
                    task_id, board, worker_rollouts, 
                    noise_seed, self.epsilon, self.alpha
                )
                self.task_queue.put(task)
        
        # Collect results from all workers
        results = []
        for _ in range(self.num_workers):
            result = self.result_queue.get()
            if result.task_id == task_id:
                results.append(result)
        
        # Aggregate statistics across all trees
        return self.aggregate_statistics(results)
    
    def aggregate_statistics(self, results):
        """
        Aggregate move statistics from all independent trees.
        
        Args:
            results: List of WorkerResult objects
            
        Returns:
            Dict with aggregated statistics per move
        """
        aggregated = defaultdict(lambda: {'total_visits': 0, 'weighted_q': 0})
        
        for result in results:
            for move, visits in result.move_visits.items():
                if visits > 0:
                    q_value = result.move_q_values[move]
                    aggregated[move]['total_visits'] += visits
                    aggregated[move]['weighted_q'] += visits * q_value
        
        # Calculate final Q values
        final_stats = {}
        for move, stats in aggregated.items():
            if stats['total_visits'] > 0:
                final_stats[move] = {
                    'visits': stats['total_visits'],
                    'q_value': stats['weighted_q'] / stats['total_visits']
                }
        
        return final_stats
    
    def get_best_move(self, stats):
        """Get best move based on visit counts."""
        if not stats:
            return None
        
        best_move = None
        best_visits = -1
        
        for move, move_stats in stats.items():
            if move_stats['visits'] > best_visits:
                best_visits = move_stats['visits']
                best_move = move
        
        return best_move
    
    def cleanup(self):
        """Clean up all processes."""
        self.stop_event.set()
        time.sleep(0.1)
        
        # Empty queues
        for q in [self.task_queue, self.result_queue, self.inference_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        # Terminate processes
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join()


# Compatibility wrapper to match original MCTS interface
class Root:
    """
    Compatibility wrapper for root parallel MCTS to match original interface.
    """
    
    _mcts_engine = None
    _mcts_lock = mp.Lock()
    
    def __init__(self, board, neuralNetwork, epsilon=0.0):
        """
        Initialize root with optional Dirichlet noise.
        
        Args:
            board: Chess board
            neuralNetwork: Neural network model
            epsilon: Dirichlet noise weight (0.0 for deterministic play)
        """
        self.board = board.copy()
        self.neuralNetwork = neuralNetwork
        self.stats = None
        self.same_paths = 0  # Compatibility attribute (not used in root parallel)
        
        with Root._mcts_lock:
            if Root._mcts_engine is None:
                # Create engine with epsilon for noise control
                Root._mcts_engine = RootParallelMCTS(neuralNetwork, epsilon=epsilon)
            elif Root._mcts_engine.epsilon != epsilon:
                # Recreate engine if epsilon changed
                Root._mcts_engine.cleanup()
                Root._mcts_engine = RootParallelMCTS(neuralNetwork, epsilon=epsilon)
    
    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        """Run parallel rollouts (compatibility method)."""
        self.stats = Root._mcts_engine.run_parallel_search(
            self.board, num_parallel_rollouts
        )
    
    def maxNSelect(self):
        """Get edge with maximum visits (compatibility method)."""
        if self.stats is None:
            return None
        
        best_move = Root._mcts_engine.get_best_move(self.stats)
        if not best_move:
            return None
        
        # Create compatibility wrapper
        class EdgeWrapper:
            def __init__(self, move, visits, q_value):
                self.move = move
                self.visits = visits
                self.q_value = q_value
            
            def getMove(self):
                return self.move
            
            def getN(self):
                return self.visits
            
            def getQ(self):
                return self.q_value
        
        move_stats = self.stats[best_move]
        return EdgeWrapper(best_move, move_stats['visits'], move_stats['q_value'])
    
    def getN(self):
        """Get total visit count at root (compatibility method)."""
        if self.stats is None:
            return 0
        
        total_visits = 0
        for move_stats in self.stats.values():
            total_visits += move_stats['visits']
        
        return total_visits
    
    def getQ(self):
        """Get Q value at root (compatibility method)."""
        if self.stats is None:
            return 0.5
        
        # Calculate weighted average Q across all moves
        total_visits = 0
        weighted_q = 0
        
        for move_stats in self.stats.values():
            visits = move_stats['visits']
            q_value = move_stats['q_value']
            total_visits += visits
            weighted_q += visits * q_value
        
        if total_visits > 0:
            return weighted_q / total_visits
        return 0.5
    
    def getStatisticsString(self):
        """Get statistics string (compatibility method)."""
        if self.stats is None:
            return ""
        
        # Sort moves by visit count
        sorted_moves = sorted(
            self.stats.items(), 
            key=lambda x: x[1]['visits'], 
            reverse=True
        )
        
        string = '|{: ^10}|{: ^10}|{: ^10}|\n'.format('move', 'N', 'Q')
        for move, move_stats in sorted_moves[:10]:
            string += '|{: ^10}|{:10.0f}|{:10.4f}|\n'.format(
                str(move), move_stats['visits'], move_stats['q_value']
            )
        
        return string
    
    @staticmethod
    def cleanup_engine():
        """Clean up the shared MCTS engine."""
        with Root._mcts_lock:
            if Root._mcts_engine:
                Root._mcts_engine.cleanup()
                Root._mcts_engine = None


# Module cleanup
import atexit
atexit.register(Root.cleanup_engine)