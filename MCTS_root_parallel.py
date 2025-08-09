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

# Set multiprocessing start method for CUDA compatibility
if torch.cuda.is_available():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

import encoder
#from inference_server import InferenceRequest, InferenceResult, start_inference_server, start_inference_server_from_state
from inference_server_parallel_v2 import InferenceRequest, InferenceResult, start_inference_server_from_state, start_inference_server
import MCTS  # Use the original single-threaded MCTS for node/edge structures with virtual loss support


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
            #print(f"Worker {worker_id} exiting due to empty task queue.")
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
    # Get legal moves first
    legal_moves = list(board.legal_moves)
    
    # Get initial neural network evaluation
    request_id = f"{worker_id}_root_{uuid.uuid4().hex[:8]}"
    request = InferenceRequest(request_id, board.fen(), worker_id)
    inference_queue.put([(request, result_queue)] )
    
    try:
        result = result_queue.get(timeout=5.0)  # 5 second timeout
        value = result.value
        move_probabilities = result.move_probabilities
    except queue.Empty:
        print(f"Warning: Timeout waiting for root evaluation in worker {worker_id}")
        # Use neutral values if timeout
        value = 0.0
        move_probabilities = np.ones(len(legal_moves)) / len(legal_moves) if legal_moves else np.array([1.0])
    
    # Apply Dirichlet noise at root for this worker
    if epsilon > 0 and len(legal_moves) > 0:
        move_probabilities = apply_dirichlet_noise(
            move_probabilities, legal_moves, epsilon, alpha, rng
        )
    
    # Create root node with noisy probabilities
    Q = value / 2.0 + 0.5
    root = MCTS.Node(board, Q, move_probabilities)
    
    # Run rollouts with smaller batching for better sequential learning
    run_parallel_rollouts(root, board, num_rollouts, worker_id, 
                         inference_queue, result_queue, batch_size=8)
    
    # Extract statistics from root
    stats = {}
    for edge in root.edges:
        move = edge.getMove()
        stats[move] = {
            'visits': edge.getN(),
            'q_value': edge.getQ()
        }
    return stats


def run_parallel_rollouts(root, base_board, num_rollouts, worker_id, 
                         inference_queue, result_queue, batch_size=8):
    """
    Execute multiple MCTS rollouts with batched inference requests.
    
    Args:
        root: Root node of the search tree
        base_board: Base chess board position
        num_rollouts: Number of rollouts to perform
        worker_id: Worker process ID
        inference_queue: Queue for sending inference requests
        result_queue: Queue for receiving inference results
        batch_size: Target batch size for inference requests (default 64)
    """
    rollout_batch = []
    pending_evaluations = []
    
    TOTAL_REQUESTS_SENT_FOR_INFERENCE = 0


    # Prepare all rollouts
    for rollout_idx in range(num_rollouts):
        # Pass base_board, prepare_rollout will make its own copy
        rollout_data = prepare_rollout(root, base_board, worker_id)
        rollout_batch.append(rollout_data)
        
        # Process batch when we reach batch_size or last rollout
        if len(rollout_batch) >= batch_size or rollout_idx == num_rollouts - 1:
            # Collect all inference requests for this batch
            batch_requests = []
            batch_metadata = []
            
            for data in rollout_batch:
                if data['needs_evaluation']:
                    request_id = f"{worker_id}_batch_{rollout_idx}_{uuid.uuid4().hex[:8]}"
                    # Store the FEN at request time to ensure consistency
                    board_fen = data['board'].fen()
                    data['request_fen'] = board_fen
                    request = InferenceRequest(request_id, board_fen, worker_id)
                    batch_requests.append((request, result_queue))
                    batch_metadata.append(data)
            
            inference_queue.put(batch_requests)
            TOTAL_REQUESTS_SENT_FOR_INFERENCE += len(batch_requests)

            # Collect all results with timeout protection
            results = []
            # Reduce timeout for better responsiveness
            timeout_per_result = 2.0  # 2 seconds timeout per result
            for i in range(len(batch_requests)):
                try:
                    result = result_queue.get(timeout=timeout_per_result)
                    results.append(result)
                except queue.Empty:
                    print(f"Warning: Timeout waiting for inference result {i+1}/{len(batch_requests)} in worker {worker_id}")
                    # Create a dummy result to avoid hanging
                    # Use neutral values to minimize impact
                    board = batch_metadata[i]['board']
                    legal_moves = list(board.legal_moves)
                    dummy_result = type('InferenceResult', (), {
                        'value': 0.0,  # Neutral value
                        'move_probabilities': np.ones(len(legal_moves)) / max(1, len(legal_moves))
                    })()
                    results.append(dummy_result)

            # Apply results and backpropagate
            for data, result in zip(batch_metadata, results):
                apply_evaluation_and_backpropagate(data, result)
            
            # Process terminal nodes that didn't need evaluation
            for data in rollout_batch:
                if not data['needs_evaluation']:
                    process_terminal_node(data)
            
            # Clear batch for next iteration
            rollout_batch = []
    
    #print(f"Worker ID: {worker_id}. Total requets sent for inference: {TOTAL_REQUESTS_SENT_FOR_INFERENCE}")


def prepare_rollout(root, board, worker_id):
    """
    Prepare a single rollout by traversing the tree until expansion or terminal.
    Virtual losses are applied during selection to ensure diverse exploration.
    
    Returns:
        Dict containing rollout data including paths and whether evaluation is needed
    """
    node_path = []
    edge_path = []
    current_node = root
    
    # Create a fresh copy of the board to avoid corruption
    rollout_board = board.copy()
    
    # Selection phase with virtual loss
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
        
        # Apply virtual loss to discourage other workers from taking the same path
        edge.addVirtualLoss()
        
        rollout_board.push(edge.getMove())
        
        if not edge.has_child():
            # Need to expand this node
            break
        
        current_node = edge.getChild()
    
    # Determine if we need neural network evaluation
    needs_evaluation = edge_path[-1] is not None
    
    return {
        'board': rollout_board,
        'node_path': node_path,
        'edge_path': edge_path,
        'needs_evaluation': needs_evaluation,
        'worker_id': worker_id
    }


def apply_evaluation_and_backpropagate(rollout_data, result):
    """
    Apply neural network evaluation results and backpropagate.
    Clear virtual losses after completion.
    """
    try:
        value = result.value
        new_Q = value / 2.0 + 0.5
        
        # Use the board from rollout_data directly
        board = rollout_data['board']
        
        # Validate board state matches what was sent to inference
        if 'request_fen' in rollout_data:
            if board.fen() != rollout_data['request_fen']:
                # Board state changed! This is a critical error
                print(f"ERROR: Board state corruption detected!")
                print(f"  Expected FEN: {rollout_data['request_fen']}")  
                print(f"  Current FEN:  {board.fen()}")
                # Use uniform probabilities as fallback
                num_legal_moves = len(list(board.legal_moves))
                result.move_probabilities = np.ones(num_legal_moves) / num_legal_moves
        
        # The result.move_probabilities should already match the board's legal moves
        # from the inference server. Just verify the count matches.
        num_legal_moves = len(list(board.legal_moves))
        num_probs = len(result.move_probabilities)
        
        if num_legal_moves != num_probs:
            # This shouldn't happen with proper inference server implementation
            # but handle gracefully by using uniform probabilities
            # Don't print warning anymore since we know this is happening
            result.move_probabilities = np.ones(num_legal_moves) / num_legal_moves
        
        # Expand node - the Edge.expand method expects move probabilities  
        # in the order of the board's legal moves
        # Since the inference server already decoded probabilities for this board,
        # we can pass them directly
        rollout_data['edge_path'][-1].expand(
            board, new_Q, result.move_probabilities
        )
        new_Q = 1.0 - new_Q
        
        # Backpropagation
        backpropagate(rollout_data['node_path'], new_Q)
        
    finally:
        # Always clear virtual losses, even if an error occurred
        for edge in rollout_data['edge_path']:
            if edge is not None:
                edge.clearVirtualLoss()


def process_terminal_node(rollout_data):
    """
    Process a terminal node that doesn't need neural network evaluation.
    Clear virtual losses after completion.
    """
    try:
        board = rollout_data['board']
        result = board.result()
        
        # Check if the game is actually over
        if result == "*":
            # Game is not over - this shouldn't happen for a true terminal node
            # Use a neutral value as fallback
            new_Q = 0.5
        else:
            winner = encoder.parseResult(result)
            if not board.turn:
                winner *= -1
            new_Q = float(winner) / 2.0 + 0.5
        
        # Backpropagation
        backpropagate(rollout_data['node_path'], new_Q)
        
    finally:
        # Always clear virtual losses, even if an error occurred
        for edge in rollout_data['edge_path']:
            if edge is not None:
                edge.clearVirtualLoss()


def backpropagate(node_path, leaf_Q):
    """
    Backpropagate value through the node path.
    """
    last_node_idx = len(node_path) - 1
    for i in range(last_node_idx, -1, -1):
        node = node_path[i]
        
        if (last_node_idx - i) % 2 == 0:
            # Same perspective as leaf
            node.N += 1
            node.sum_Q += leaf_Q
        else:
            # Opposite perspective
            node.N += 1
            node.sum_Q += 1.0 - leaf_Q


def run_single_rollout(root, board, worker_id, inference_queue, result_queue):
    """
    Execute a single MCTS rollout on the independent tree with virtual loss.
    """
    node_path = []
    edge_path = []
    current_node = root
    virtual_losses_applied = []  # Track edges with virtual losses
    
    try:
        # Selection phase with virtual loss
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
            
            # Apply virtual loss to discourage other workers from taking the same path
            edge.addVirtualLoss()
            virtual_losses_applied.append(edge)  # Track for cleanup
            
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
            inference_queue.put([(request, result_queue)])
            
            try:
                result = result_queue.get(timeout=5.0)  # 5 second timeout
                value = result.value
                new_Q = value / 2.0 + 0.5
            except queue.Empty:
                print(f"Warning: Timeout in single rollout for worker {worker_id}")
                # Use neutral value if timeout
                value = 0.0
                new_Q = 0.5
                result = type('InferenceResult', (), {
                    'value': value,
                    'move_probabilities': np.ones(len(list(board.legal_moves))) / len(list(board.legal_moves))
                })()
            
            # Ensure move_probabilities matches the board's legal moves
            num_legal_moves = len(list(board.legal_moves))
            num_probs = len(result.move_probabilities)
            
            if num_legal_moves != num_probs:
                # print("Warning: Number of legal moves does not match number of probabilities.")
                # This should not happen in a well-formed game, but handle gracefully
                # Re-decode probabilities for this specific board instance
                # Get fresh board from FEN to match what inference server saw
                fresh_board = chess.Board(board.fen())
                # The result.move_probabilities are already decoded for fresh_board
                # We need to map them to our current board's move ordering
                move_probs = np.zeros(num_legal_moves)
                fresh_moves = list(fresh_board.legal_moves)
                our_moves = list(board.legal_moves)
                
                for i, our_move in enumerate(our_moves):
                    try:
                        # Find this move in the fresh board's move list
                        fresh_idx = fresh_moves.index(our_move)
                        if fresh_idx < num_probs:
                            move_probs[i] = result.move_probabilities[fresh_idx]
                        else:
                            move_probs[i] = 1e-8  # Small probability for missing moves
                    except ValueError:
                        # Move not found in fresh board's list
                        move_probs[i] = 1e-8
                
                result.move_probabilities = move_probs
            
            # Expand node
            edge_path[-1].expand(board, new_Q, result.move_probabilities)
            new_Q = 1.0 - new_Q
        else:
            # Terminal node
            result = board.result()
            if result == "*":
                # Game is not over - use neutral value
                new_Q = 0.5
            else:
                winner = encoder.parseResult(result)
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
    
    finally:
        # Always clear virtual losses after rollout completion
        # Use the tracked list to ensure we only clear losses we applied
        for edge in virtual_losses_applied:
            edge.clearVirtualLoss()


class RootParallelMCTS:
    """
    Root parallelization MCTS with Dirichlet noise for exploration diversity.
    """
    
    def __init__(self, model, num_workers=None, epsilon=0.1, alpha=0.3,
                 inference_batch_size=256, inference_timeout_ms=1):
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
            num_workers = 1
        
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
        
        # Use state dict approach for all devices to avoid multiprocessing issues
        model_state = self.model.cpu().state_dict()
        try:
            # Get model architecture parameters
            conv1_out_channels = self.model.convBlock1.conv1.out_channels
            num_blocks = len(self.model.residualBlocks)
            model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out_channels}
        except:
            # Fallback to default config
            model_config = {'num_blocks': 20, 'num_channels': 256}
        
        # Determine device type for inference server
        if device.type == 'cuda':
            device_type = 'cuda'
        elif device.type == 'mps':
            device_type = 'mps'
        else:
            device_type = 'cpu'
        
        NUM_WORKERS_INFERENCE_SERVER = 3

        inference_process = Process(
            target=start_inference_server_from_state,
            args=(model_state, model_config, device_type, self.inference_queue, 
                  self.stop_event, NUM_WORKERS_INFERENCE_SERVER, self.inference_batch_size, self.inference_timeout_ms)
        )

        # inference_process = Process(
        #     target=start_inference_server_from_state,
        #     args=(model_state, model_config, device_type, self.inference_queue, 
        #           self.stop_event, self.inference_batch_size, self.inference_timeout_ms)
        # )
        

        
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
        
        Simplified approach with stronger tactical handling.
        
        Args:
            board: Chess board position
            num_rollouts: Total number of rollouts to perform
            
        Returns:
            Aggregated move statistics
        """
        # Store board for later use in get_best_move
        self.current_board = board.copy()
        
        # Use standard epsilon for all positions
        effective_epsilon = self.epsilon
        
        # Simple single-phase search for better convergence
        task_id = uuid.uuid4().hex
        rollouts_per_worker = num_rollouts // self.num_workers
        
        # Ensure at least one rollout per worker
        if num_rollouts > 0 and rollouts_per_worker == 0:
            rollouts_per_worker = 1
        
        # Create tasks
        tasks_created = 0
        for i in range(self.num_workers):
            worker_rollouts = rollouts_per_worker
            if i < (num_rollouts % self.num_workers):
                worker_rollouts += 1
            
            if worker_rollouts > 0:
                noise_seed = np.random.randint(0, 2**16)
                task = WorkerTask(
                    task_id, board, worker_rollouts,
                    noise_seed, effective_epsilon, self.alpha
                )
                self.task_queue.put(task)
                tasks_created += 1
        
        # Collect results
        results = []
        timeout_per_worker = 100
        for i in range(tasks_created):
            try:
                result = self.result_queue.get(timeout=timeout_per_worker)
                if result.task_id == task_id:
                    results.append(result)
            except queue.Empty:
                print(f"Warning: Timeout waiting for worker {i+1}/{tasks_created}")
        
        # Aggregate statistics across all trees
        return self.aggregate_statistics(results)
    
    def _run_search_phase(self, board, num_rollouts, epsilon, phase_id):
        """Run a single phase of search with given rollouts."""
        task_id = f"{uuid.uuid4().hex}_{phase_id}"
        rollouts_per_worker = num_rollouts // self.num_workers
        
        # Ensure at least one rollout per worker
        if num_rollouts > 0 and rollouts_per_worker == 0:
            rollouts_per_worker = 1
        
        # Create tasks
        tasks_created = 0
        for i in range(self.num_workers):
            worker_rollouts = rollouts_per_worker
            if i < (num_rollouts % self.num_workers):
                worker_rollouts += 1
            
            if worker_rollouts > 0:
                noise_seed = np.random.randint(0, 2**16)
                task = WorkerTask(
                    task_id, board, worker_rollouts,
                    noise_seed, epsilon, self.alpha
                )
                self.task_queue.put(task)
                tasks_created += 1
        
        # Collect results
        results = []
        timeout_per_worker = 30.0
        for i in range(tasks_created):
            try:
                result = self.result_queue.get(timeout=timeout_per_worker)
                if result.task_id == task_id:
                    results.append(result)
            except queue.Empty:
                print(f"Warning: Timeout in {phase_id} phase, worker {i+1}/{tasks_created}")
        
        return results
    
    def _calculate_move_variances(self, results):
        """Calculate Q-value variance for each move across workers."""
        if not results:
            return {}
        
        move_q_values = defaultdict(list)
        
        # Collect Q-values from each worker
        for result in results:
            for move, visits in result.move_visits.items():
                if visits > 0:
                    move_q_values[move].append(result.move_q_values[move])
        
        # Calculate variance for each move
        move_variances = {}
        for move, q_values in move_q_values.items():
            if len(q_values) > 1:
                variance = np.var(q_values)
                move_variances[move] = variance
            else:
                move_variances[move] = 0.0
        
        return move_variances
    
    def _run_variance_based_search(self, board, num_rollouts, epsilon, move_variances):
        """Run search with rollouts allocated based on move variance."""
        if not move_variances:
            return []
        
        # Normalize variances to get allocation proportions
        total_variance = sum(move_variances.values())
        if total_variance == 0:
            # Equal allocation if no variance
            return self._run_search_phase(board, num_rollouts, epsilon, "focus")
        
        # For simplicity, we'll still use equal worker allocation
        # but could extend this to focus workers on high-variance moves
        # For now, just run with awareness that high-variance positions need more exploration
        return self._run_search_phase(board, num_rollouts, epsilon, "focus")
    
    def aggregate_statistics(self, results):
        """
        Aggregate move statistics from all independent trees.
        
        Use confidence-based weighting for better aggregation.
        
        Args:
            results: List of WorkerResult objects
            
        Returns:
            Dict with aggregated statistics per move
        """
        if not results:
            return {}
            
        # Collect per-worker statistics for each move
        move_worker_stats = defaultdict(list)
        
        for result in results:
            for move, visits in result.move_visits.items():
                if visits > 0:
                    q_value = result.move_q_values[move]
                    move_worker_stats[move].append({
                        'visits': visits,
                        'q_value': q_value,
                        'confidence': np.sqrt(visits)  # Confidence weight
                    })
        
        # Aggregate with confidence weighting
        final_stats = {}
        for move, worker_stats in move_worker_stats.items():
            total_visits = sum(ws['visits'] for ws in worker_stats)
            
            # Use confidence-weighted average for all positions
            total_confidence = sum(ws['confidence'] for ws in worker_stats)
            if total_confidence > 0:
                final_q = sum(ws['q_value'] * ws['confidence'] for ws in worker_stats) / total_confidence
            else:
                # Fallback to simple average
                final_q = sum(ws['q_value'] for ws in worker_stats) / len(worker_stats)
            
            final_stats[move] = {
                'visits': total_visits,
                'q_value': final_q
            }
        
        return final_stats
    
    def get_best_move(self, stats, board=None):
        """Get best move based on visit counts."""
        if not stats:
            # No stats means no viable moves found (e.g., drawn position)
            return None
        
        # Use traditional visit-based selection for all positions
        best_move = None
        best_visits = -1
        
        for move, move_stats in stats.items():
            if move_stats['visits'] > best_visits:
                best_visits = move_stats['visits']
                best_move = move
        
        return best_move
    
    def partial_cleanup(self):
        for q in [self.task_queue, self.result_queue, self.inference_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

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
    
    def __init__(self, board, neuralNetwork, epsilon=0.1):
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
            board, num_parallel_rollouts
        )

    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts, num_parallel_rollouts):
        """Run total parallel rollouts (compatibility method)."""
        self.stats = Root._mcts_engine.run_parallel_search(
            board, total_rollouts
        )
    
    def maxNSelect(self):
        """Get edge with maximum visits (compatibility method)."""
        if self.stats is None:
            return None
        
        best_move = Root._mcts_engine.get_best_move(self.stats, self.board)
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

    @staticmethod
    def partial_cleanup_engine():
        with Root._mcts_lock:
            if Root._mcts_engine:
                Root._mcts_engine.partial_cleanup()



# Module cleanup
import atexit
atexit.register(Root.cleanup_engine)