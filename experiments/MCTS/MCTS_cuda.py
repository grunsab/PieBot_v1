import sys
sys.path.append('../..')
import encoder
import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from experiments.async_neural_net_server_cuda import AsyncNeuralNetworkServerCUDA, NeuralNetworkPoolCUDA
from collections import deque
import numpy as np
import queue
import os

# Import the original classes we'll reuse
from MCTS_async import calcUCT, Node, Edge


class AsyncRootCUDA(Node):
    """
    High-performance root node optimized for CUDA GPUs (RTX 4080).
    Uses aggressive parallelization and batching strategies.
    """

    def __init__(self, board, nn_server):
        """
        Create the root of the search tree.

        Args:
            board (chess.Board) the chess position
            nn_server (AsyncNeuralNetworkServerCUDA) the async neural network server
        """
        self.nn_server = nn_server
        self.board = board
        
        # Get initial evaluation synchronously
        future = nn_server.evaluate_async(board)
        value, move_probabilities = future.result()

        Q = value / 2. + 0.5

        super().__init__(board, Q, move_probabilities)

        # Statistics
        self.same_paths = 0
        self.rollouts_started = 0
        self.rollouts_completed = 0
        
        # Pending expansions with priority queue for better batching
        self.pending_lock = Lock()
        self.pending_queue = queue.PriorityQueue(maxsize=10000)
        self.pending_count = 0
        
        # Thread control
        self.running = False
        self.worker_pool = None
        self.backup_threads = []
        
        # Optimized for high core count CPUs
        self.max_selection_workers = min(128, os.cpu_count() * 2)
        self.max_backup_workers = 4  # Multiple backup threads for high throughput

    def start_async_search(self, num_rollouts, num_selection_workers=None):
        """
        Start asynchronous search with specified number of rollouts.
        
        Args:
            num_rollouts: Total number of rollouts to perform
            num_selection_workers: Number of selection worker threads (auto-tune if None)
        """
        if num_selection_workers is None:
            # Auto-tune based on rollouts and CPU cores
            num_selection_workers = min(
                self.max_selection_workers,
                max(32, num_rollouts // 100)  # At least 32 workers for good batching
            )
        
        self.running = True
        self.target_rollouts = num_rollouts
        
        # Create worker pool for selection
        self.worker_pool = ThreadPoolExecutor(max_workers=num_selection_workers)
        
        # Start multiple backup threads for better throughput
        for i in range(self.max_backup_workers):
            backup_thread = Thread(target=self._backup_worker, args=(i,), daemon=True)
            backup_thread.start()
            self.backup_threads.append(backup_thread)
        
        # Launch selection workers with staggered start for better batching
        selection_futures = []
        rollouts_per_worker = num_rollouts // num_selection_workers
        remainder = num_rollouts % num_selection_workers
        
        for i in range(num_selection_workers):
            n_rollouts = rollouts_per_worker
            if i < remainder:
                n_rollouts += 1
            
            if n_rollouts > 0:
                # Stagger worker start times slightly for better batching
                start_delay = i * 0.0001  # 0.1ms between workers
                future = self.worker_pool.submit(
                    self._selection_worker, i, n_rollouts, start_delay
                )
                selection_futures.append(future)
        
        # Wait for all selection workers to complete
        for future in as_completed(selection_futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in selection worker: {e}")
                import traceback
                traceback.print_exc()
        
        # Stop backup threads
        self.running = False
        
        # Process any remaining pending expansions
        self._process_all_pending()
        
        # Wait for backup threads to finish
        for thread in self.backup_threads:
            thread.join(timeout=1.0)

    def _selection_worker(self, worker_id, num_rollouts, start_delay):
        """
        Worker thread that performs selection and submits neural network requests.
        """
        # Stagger start for better batching
        if start_delay > 0:
            time.sleep(start_delay)
            
        for i in range(num_rollouts):
            try:
                # Perform selection
                board_copy = self.board.copy()
                node_path = []
                edge_path = []
                
                self._select_leaf(board_copy, node_path, edge_path)
                
                edge = edge_path[-1]
                
                if edge is None:
                    # Terminal node - backup immediately
                    winner = encoder.parseResult(board_copy.result())
                    if not board_copy.turn:
                        winner *= -1
                    new_Q = float(winner) / 2. + 0.5
                    self._backup_values(node_path, edge_path, new_Q)
                    
                    with self.pending_lock:
                        self.rollouts_completed += 1
                else:
                    # Submit neural network request
                    future = self.nn_server.evaluate_async(board_copy)
                    
                    # Add to pending queue with priority (earlier = higher priority)
                    priority = time.time()
                    item = (priority, board_copy, node_path, edge_path, future)
                    
                    try:
                        self.pending_queue.put_nowait(item)
                        with self.pending_lock:
                            self.pending_count += 1
                            self.rollouts_started += 1
                    except queue.Full:
                        # Queue full, process synchronously
                        value, move_probs = future.result()
                        new_Q = value / 2. + 0.5
                        expanded = edge.expand(board_copy, new_Q, move_probs)
                        if not expanded:
                            with self.pending_lock:
                                self.same_paths += 1
                        new_Q = 1. - new_Q
                        self._backup_values(node_path, edge_path, new_Q)
                        with self.pending_lock:
                            self.rollouts_completed += 1
                
            except Exception as e:
                print(f"Error in selection worker {worker_id}: {e}")
                import traceback
                traceback.print_exc()

    def _backup_worker(self, worker_id):
        """
        Worker thread that processes neural network results and performs backups.
        """
        batch_timeout = 0.001  # 1ms timeout for batching
        
        while self.running or self.pending_count > 0:
            items_to_process = []
            
            # Collect a batch of completed items
            deadline = time.time() + batch_timeout
            
            while time.time() < deadline and len(items_to_process) < 32:
                try:
                    timeout = max(0.0001, deadline - time.time())
                    item = self.pending_queue.get(timeout=timeout)
                    items_to_process.append(item)
                except queue.Empty:
                    break
            
            # Process collected items
            for item in items_to_process:
                try:
                    priority, board, node_path, edge_path, future = item
                    
                    # Check if result is ready
                    if future.done():
                        # Process immediately
                        value, move_probabilities = future.result()
                        new_Q = value / 2. + 0.5
                        
                        edge = edge_path[-1]
                        expanded = edge.expand(board, new_Q, move_probabilities)
                        
                        if not expanded:
                            with self.pending_lock:
                                self.same_paths += 1
                        
                        new_Q = 1. - new_Q
                        self._backup_values(node_path, edge_path, new_Q)
                        
                        with self.pending_lock:
                            self.pending_count -= 1
                            self.rollouts_completed += 1
                    else:
                        # Put back in queue if not ready
                        self.pending_queue.put(item)
                        
                except Exception as e:
                    print(f"Error in backup worker {worker_id}: {e}")
                    with self.pending_lock:
                        self.pending_count -= 1
                    # Remove virtual losses on error
                    for edge in edge_path:
                        if edge is not None:
                            edge.removeVirtualLoss()
                            
            # Small sleep to prevent busy waiting
            if not items_to_process:
                time.sleep(0.0001)

    def _process_all_pending(self):
        """Process all remaining pending expansions."""
        while self.pending_count > 0:
            try:
                item = self.pending_queue.get_nowait()
                priority, board, node_path, edge_path, future = item
                
                # Wait for result
                value, move_probabilities = future.result(timeout=5.0)
                new_Q = value / 2. + 0.5
                
                edge = edge_path[-1]
                expanded = edge.expand(board, new_Q, move_probabilities)
                
                if not expanded:
                    with self.pending_lock:
                        self.same_paths += 1
                
                new_Q = 1. - new_Q
                self._backup_values(node_path, edge_path, new_Q)
                
                with self.pending_lock:
                    self.pending_count -= 1
                    self.rollouts_completed += 1
                    
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing pending expansion: {e}")
                with self.pending_lock:
                    self.pending_count -= 1

    def _select_leaf(self, board, node_path, edge_path):
        """
        Select a leaf node for expansion.
        """
        cNode = self

        while True:
            node_path.append(cNode)

            cEdge = cNode.UCTSelect()

            edge_path.append(cEdge)

            if cEdge == None:
                # Terminal node
                assert cNode.isTerminal()
                break
            
            cEdge.addVirtualLoss()

            board.push(cEdge.getMove())

            if not cEdge.has_child():
                # Unexpanded node
                break

            cNode = cEdge.getChild()

    def _backup_values(self, node_path, edge_path, value):
        """
        Backup value through the tree and remove virtual losses.
        """
        last_node_idx = len(node_path) - 1
        
        # Update node values
        for i in range(last_node_idx, -1, -1):
            node = node_path[i]
            is_from_child = (last_node_idx - i) % 2 == 1
            node.updateStats(value, is_from_child)
        
        # Remove virtual losses
        for edge in edge_path:
            if edge is not None:
                edge.removeVirtualLoss()

    def wait_for_completion(self, timeout=None):
        """Wait for all rollouts to complete."""
        start_time = time.time()
        
        while self.rollouts_completed < self.target_rollouts:
            if timeout and (time.time() - start_time) > timeout:
                return False
                
            time.sleep(0.01)
        
        return True

    def get_statistics(self):
        """Get search statistics."""
        with self.pending_lock:
            return {
                'rollouts_started': self.rollouts_started,
                'rollouts_completed': self.rollouts_completed,
                'pending_expansions': self.pending_count,
                'same_paths': self.same_paths
            }

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
            
        for thread in self.backup_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

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


class MCTSEngineCUDA:
    """
    High-performance MCTS engine optimized for CUDA GPUs.
    """
    
    def __init__(self, neural_network, device=None, max_batch_size=512, 
                 num_workers=64, verbose=False):
        """
        Initialize MCTS engine with CUDA-optimized settings.
        
        Args:
            neural_network: The neural network model
            device: Device to run on (auto-detect if None)
            max_batch_size: Maximum batch size (512 optimal for RTX 4080)
            num_workers: Number of MCTS worker threads (64+ for high-end CPUs)
            verbose: Whether to print performance info
        """
        self.neural_network = neural_network
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Create CUDA-optimized async neural network server
        self.nn_server = AsyncNeuralNetworkServerCUDA(
            neural_network, 
            device=device,
            max_batch_size=max_batch_size,
            max_wait_time=0.005,  # 5ms for better batching
            verbose=verbose
        )
        
        # Performance monitoring
        self.total_rollouts = 0
        self.total_time = 0
        
    def start(self):
        """Start the engine."""
        self.nn_server.start()
        
    def stop(self):
        """Stop the engine and print statistics."""
        self.nn_server.stop()
        
        if self.verbose and self.total_rollouts > 0:
            print(f"\nMCTSEngineCUDA Statistics:")
            print(f"  Total rollouts: {self.total_rollouts}")
            print(f"  Total time: {self.total_time:.2f}s")
            print(f"  Average NPS: {self.total_rollouts / self.total_time:.0f}")
            
    def search(self, board, num_rollouts, return_root=False):
        """
        Search for the best move using CUDA-optimized async MCTS.
        
        Args:
            board: Current board position
            num_rollouts: Number of rollouts to perform
            return_root: If True, return the root node instead of best move
            
        Returns:
            Best move or root node if return_root=True
        """
        start_time = time.time()
        
        # Create root with async evaluation
        root = AsyncRootCUDA(board, self.nn_server)
        
        # Start async search with auto-tuned worker count
        root.start_async_search(num_rollouts)
        
        # Wait for completion
        root.wait_for_completion(timeout=300.0)  # 5 minute timeout
        
        elapsed = time.time() - start_time
        self.total_rollouts += num_rollouts
        self.total_time += elapsed
        
        stats = root.get_statistics()
        actual_rollouts = stats['rollouts_completed']
        
        if self.verbose:
            nps = actual_rollouts / elapsed if elapsed > 0 else 0
            print(f"Completed {actual_rollouts} rollouts in {elapsed:.2f}s ({nps:.0f} NPS)")
            print(f"Pending expansions: {stats['pending_expansions']}")
            print(root.getStatisticsString())
            
        if return_root:
            return root
            
        # Select best move
        best_edge = root.maxNSelect()
        return best_edge.getMove() if best_edge else None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()