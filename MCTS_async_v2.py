import encoder
import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from async_neural_net_server import AsyncNeuralNetworkServer, NeuralNetworkPool
from collections import deque
import numpy as np
import queue

# Import the original classes we'll reuse
from MCTS_async import calcUCT, Node, Edge


class PendingExpansion:
    """Represents a pending node expansion waiting for neural network evaluation."""
    def __init__(self, board, node_path, edge_path, future):
        self.board = board
        self.node_path = node_path
        self.edge_path = edge_path
        self.future = future
        self.timestamp = time.time()


class AsyncRootV2(Node):
    """
    Optimized async root node that maximizes parallelism and batching.
    Key improvements:
    1. Separate threads for selection and backup
    2. Pending expansion queue for better batching
    3. Continuous rollout generation
    """

    def __init__(self, board, nn_server):
        """
        Create the root of the search tree.

        Args:
            board (chess.Board) the chess position
            nn_server (AsyncNeuralNetworkServer) the async neural network server
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
        
        # Pending expansions
        self.pending_lock = Lock()
        self.pending_expansions = deque()
        
        # Thread control
        self.running = False
        self.worker_pool = None
        self.backup_thread = None

    def start_async_search(self, num_rollouts, num_selection_workers=32):
        """
        Start asynchronous search with specified number of rollouts.
        
        Args:
            num_rollouts: Total number of rollouts to perform
            num_selection_workers: Number of selection worker threads
        """
        self.running = True
        self.target_rollouts = num_rollouts
        
        # Create worker pool for selection
        self.worker_pool = ThreadPoolExecutor(max_workers=num_selection_workers)
        
        # Start backup thread
        self.backup_thread = Thread(target=self._backup_worker, daemon=True)
        self.backup_thread.start()
        
        # Launch selection workers
        selection_futures = []
        rollouts_per_worker = num_rollouts // num_selection_workers
        remainder = num_rollouts % num_selection_workers
        
        for i in range(num_selection_workers):
            n_rollouts = rollouts_per_worker
            if i < remainder:
                n_rollouts += 1
            
            if n_rollouts > 0:
                future = self.worker_pool.submit(self._selection_worker, i, n_rollouts)
                selection_futures.append(future)
        
        # Wait for all selection workers to complete
        for future in as_completed(selection_futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in selection worker: {e}")
                import traceback
                traceback.print_exc()
        
        # Stop backup thread
        self.running = False
        
        # Process any remaining pending expansions
        self._process_pending_expansions()
        
        # Wait for backup thread to finish
        if self.backup_thread:
            self.backup_thread.join(timeout=1.0)

    def _selection_worker(self, worker_id, num_rollouts):
        """
        Worker thread that performs selection and submits neural network requests.
        """
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
                    
                    # Add to pending expansions
                    pending = PendingExpansion(board_copy, node_path, edge_path, future)
                    
                    with self.pending_lock:
                        self.pending_expansions.append(pending)
                        self.rollouts_started += 1
                
            except Exception as e:
                print(f"Error in selection worker {worker_id}: {e}")
                import traceback
                traceback.print_exc()

    def _backup_worker(self):
        """
        Worker thread that processes neural network results and performs backups.
        """
        while self.running or len(self.pending_expansions) > 0:
            self._process_pending_expansions()
            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def _process_pending_expansions(self):
        """Process completed neural network evaluations."""
        completed = []
        
        with self.pending_lock:
            # Check which futures are done
            remaining = deque()
            for pending in self.pending_expansions:
                if pending.future.done():
                    completed.append(pending)
                else:
                    remaining.append(pending)
            self.pending_expansions = remaining
        
        # Process completed expansions
        for pending in completed:
            try:
                # Get neural network result
                value, move_probabilities = pending.future.result()
                new_Q = value / 2. + 0.5
                
                # Expand node
                edge = pending.edge_path[-1]
                expanded = edge.expand(pending.board, new_Q, move_probabilities)
                
                if not expanded:
                    with self.pending_lock:
                        self.same_paths += 1
                
                # Backup values
                new_Q = 1. - new_Q
                self._backup_values(pending.node_path, pending.edge_path, new_Q)
                
                with self.pending_lock:
                    self.rollouts_completed += 1
                    
            except Exception as e:
                print(f"Error processing expansion: {e}")
                # Remove virtual losses on error
                for edge in pending.edge_path:
                    if edge is not None:
                        edge.removeVirtualLoss()

    def _select_leaf(self, board, node_path, edge_path):
        """
        Select a leaf node for expansion.
        Modified from original to use self as root.
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
            
            with self.pending_lock:
                if self.rollouts_completed % 100 == 0:
                    pending_count = len(self.pending_expansions)
                    
            time.sleep(0.01)
        
        return True

    def get_statistics(self):
        """Get search statistics."""
        with self.pending_lock:
            return {
                'rollouts_started': self.rollouts_started,
                'rollouts_completed': self.rollouts_completed,
                'pending_expansions': len(self.pending_expansions),
                'same_paths': self.same_paths
            }

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
            
        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=1.0)

    # Reuse these methods from parent
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


class MCTSEngineV2:
    """
    Optimized MCTS engine with better async performance.
    """
    
    def __init__(self, neural_network, device=None, max_batch_size=256, 
                 num_workers=32, verbose=False):
        """
        Initialize MCTS engine with async neural network support.
        
        Args:
            neural_network: The neural network model
            device: Device to run on (auto-detect if None)
            max_batch_size: Maximum batch size for neural network
            num_workers: Number of MCTS worker threads
            verbose: Whether to print performance info
        """
        self.neural_network = neural_network
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Create async neural network server with longer wait time for better batching
        self.nn_server = AsyncNeuralNetworkServer(
            neural_network, 
            device=device,
            max_batch_size=max_batch_size,
            max_wait_time=0.002,  # 2ms for better batching
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
            print(f"\nMCTSEngineV2 Statistics:")
            print(f"  Total rollouts: {self.total_rollouts}")
            print(f"  Total time: {self.total_time:.2f}s")
            print(f"  Average NPS: {self.total_rollouts / self.total_time:.0f}")
            
    def search(self, board, num_rollouts, return_root=False):
        """
        Search for the best move using async MCTS.
        
        Args:
            board: Current board position
            num_rollouts: Number of rollouts to perform
            return_root: If True, return the root node instead of best move
            
        Returns:
            Best move or root node if return_root=True
        """
        start_time = time.time()
        
        # Create root with async evaluation
        root = AsyncRootV2(board, self.nn_server)
        
        # Start async search
        root.start_async_search(num_rollouts, num_selection_workers=self.num_workers)
        
        # Wait for completion
        root.wait_for_completion(timeout=60.0)
        
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