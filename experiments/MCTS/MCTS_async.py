import encoder
import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from async_neural_net_server import AsyncNeuralNetworkServer, NeuralNetworkPool
from collections import deque
import numpy as np

def calcUCT( edge, N_p ):
    """
    Calculate the UCT formula.

    Args:
        edge (Edge) the edge which the UCT formula is for
        N_p (float) the parents visit count

    Returns:
        (float) the calculated value
    """

    Q = edge.getQ()

    N_c = edge.getN()

    P = edge.getP()

    # Handle NaN values from neural network
    if math.isnan( P ):
        P = 1.0 / 200.0  # Small uniform probability

    C = 1.5

    UCT = Q + P * C * math.sqrt( N_p ) / ( 1 + N_c )

    # Final safety check
    if math.isnan( UCT ):
        UCT = 0.0
    
    return UCT

class Node:
    """
    A node in the search tree.
    Nodes store their visit count (N), the sum of the
    win probabilities in the subtree from the point
    of view of this node (sum_Q), and a list of
    edges
    """

    def __init__( self, board, new_Q, move_probabilities ):
        """
        Args:
            board (chess.Board) the chess board
            new_Q (float) the probability of winning according to neural network
            move_probabilities (numpy.array (200) float) probability distribution across move list
        """
        self._lock = Lock()  # Add thread-safe lock
        self.N = 1.

        # Handle NaN values in Q
        if math.isnan(new_Q):
            self.sum_Q = 0.5  # Neutral value
        else:
            self.sum_Q = new_Q

        self.edges = []

        for idx, move in enumerate( board.legal_moves ):
            edge = Edge( move, move_probabilities[ idx ] )
            self.edges.append( edge )

    def getN( self ):
        """
        Returns:
            (float) the number of rollouts performed
        """

        return self.N
    
    def getQ( self ):
        """
        Returns:
            (float) the number of rollouts performed
        """

        return self.sum_Q / self.N

    def UCTSelect( self ):
        """
        Get the edge that maximizes the UCT formula, or none
        if this node is terminal.
        Returns:
            max_edge (Edge) the edge maximizing the UCT formula.
        """

        max_uct = -1000.
        max_edge = None

        for edge in self.edges:
            try:
                uct = calcUCT( edge, self.N )

                if max_uct < uct:
                    max_uct = uct
                    max_edge = edge
            except Exception as e:
                # Skip this edge if there's an error
                continue

        # If no valid edge was found but node is not terminal, pick first edge
        if max_edge is None and not self.isTerminal() and self.edges:
            max_edge = self.edges[0]

        assert not ( max_edge == None and not self.isTerminal() )

        return max_edge
    
    def maxNSelect( self ):
        """
        Returns:
            max_edge (Edge) the edge with maximum N.
        """

        max_N = -1
        max_edge = None

        for edge in self.edges:

            N = edge.getN()

            if max_N < N:
                max_N = N
                max_edge = edge

        return max_edge

    def getStatisticsString( self ):
        """
        Get a string containing the current search statistics.
        Returns:
            string (string) a string describing all the moves from this node
        """

        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
                'move', 'P', 'N', 'Q', 'UCT' )

        edges = self.edges.copy()

        edges.sort( key=lambda edge: edge.getN() )

        edges.reverse()

        for edge in edges:

            move = edge.getMove()

            P = edge.getP()

            N = edge.getN()

            Q = edge.getQ()

            UCT = calcUCT( edge, self.N )

            string += '|{: ^10}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|\n'.format(
                str( move ), P, N, Q, UCT )

        return string

    def isTerminal( self ):
        """
        Checks if this node is terminal.'
        """
        return len( self.edges ) == 0

    def updateStats(self, value, from_child_perspective):
        """Thread-safe update of node statistics"""
        with self._lock:
            self.N += 1
            if from_child_perspective:
                self.sum_Q += 1. - value
            else:
                self.sum_Q += value

class Edge:
    """
    An edge in the search tree.
    Each edge stores a move, a move probability,
    virtual losses and a child.
    """

    def __init__( self, move, move_probability ):
        """
        Args:
            move (chess.Move) the move this edge represents
            move_probability (float) this move's probability from the neural network
        """

        self.move = move

        # Handle NaN or invalid probabilities
        if math.isnan(move_probability) or move_probability < 0:
            self.P = 1.0 / 200.0  # Small uniform probability
        else:
            self.P = move_probability

        self.child = None
        
        self._lock = Lock()  # Thread-safe lock for virtual losses
        self.virtualLosses = 0.

    def has_child( self ):
        """
        Returns:
            (bool) whether this edge has a child
        """

        return self.child != None

    def getN( self ):
        """
        Returns:
            (int) the child's N
        """

        if self.has_child():
            return self.child.N + self.virtualLosses
        else:
            return 0. + self.virtualLosses

    def getQ( self ):
        """
        Returns:
            (int) the child's Q
        """
        if self.has_child():
            return 1. - ( ( self.child.sum_Q + self.virtualLosses ) / ( self.child.N + self.virtualLosses ) )
        else:
            return 0.

    def getP( self ):
        """
        Returns:
            (int) this move's probability (P)
        """

        return self.P

    def expand( self, board, new_Q, move_probabilities ):
        """
        Create the child node with the given board position. Return
        True if we are expanding an unexpanded node, and otherwise false.
        Args:
            board (chess.Board) the chess position
            new_Q (float) the probability of winning according to the neural network
            move_probabilities (numpy.array (200) float) the move probabilities according to the neural network

        Returns:
            (bool) whether we are expanding an unexpanded node
        """

        if self.child == None:

            self.child = Node( board, new_Q, move_probabilities )

            return True

        else:

            return False

    def getChild( self ):
        """
        Returns:
            (Node) this edge's child node
        """

        return self.child

    def getMove( self ):
        """
        Returns:
            (chess.Move) this edge's move
        """

        return self.move

    def addVirtualLoss( self ):
        """
        When doing multiple rollouts in parallel,
        we can discourage threads from taking
        the same path by adding fake losses
        to visited nodes.
        """
        with self._lock:
            self.virtualLosses += 1

    def removeVirtualLoss( self ):
        """Remove a single virtual loss."""
        with self._lock:
            self.virtualLosses = max(0, self.virtualLosses - 1)

    def clearVirtualLoss( self ):
        with self._lock:
            self.virtualLosses = 0.


class AsyncRoot( Node ):
    """
    Root node that supports asynchronous neural network evaluation.
    This enables much higher throughput by overlapping CPU and GPU work.
    """

    def __init__( self, board, nn_server ):
        """
        Create the root of the search tree.

        Args:
            board (chess.Board) the chess position
            nn_server (AsyncNeuralNetworkServer) the async neural network server
        """
        self.nn_server = nn_server
        self.board = board  # Store the root board position
        
        # Get initial evaluation synchronously
        future = nn_server.evaluate_async(board)
        value, move_probabilities = future.result()

        Q = value / 2. + 0.5

        super().__init__( board, Q, move_probabilities )

        # Statistics
        self.same_paths = 0
        self.pending_evaluations = 0
        self.completed_evaluations = 0
        
        # Thread pool for workers
        self.worker_pool = None
        self.max_workers = 64  # Much higher worker count for better parallelism

    def async_rollout_worker(self, worker_id, num_rollouts):
        """
        Worker thread that performs rollouts asynchronously.
        
        Args:
            worker_id: Unique identifier for this worker
            num_rollouts: Number of rollouts to perform
        """
        for _ in range(num_rollouts):
            # Phase 1: Select leaf node
            board = self.board.copy()
            node_path = []
            edge_path = []
            
            self.selectTask(board, node_path, edge_path)
            
            edge = edge_path[-1]
            
            if edge is None:
                # Terminal node
                winner = encoder.parseResult(board.result())
                if not board.turn:
                    winner *= -1
                new_Q = float(winner) / 2. + 0.5
                
                # Update immediately
                self._backup(node_path, edge_path, new_Q)
                
            else:
                # Phase 2: Request neural network evaluation
                future = self.nn_server.evaluate_async(board)
                
                # Phase 3: Continue with other work while waiting
                # The virtual losses ensure other threads explore different paths
                
                # Phase 4: Get result and update tree
                try:
                    value, move_probabilities = future.result(timeout=0.1)
                    new_Q = value / 2. + 0.5
                    
                    # Expand node
                    expanded = edge.expand(board, new_Q, move_probabilities)
                    if not expanded:
                        self.same_paths += 1
                    
                    new_Q = 1. - new_Q
                    
                    # Backup
                    self._backup(node_path, edge_path, new_Q)
                    
                except TimeoutError:
                    # Timeout - remove virtual losses but don't update values
                    for edge in edge_path:
                        if edge is not None:
                            edge.removeVirtualLoss()

    def _backup(self, node_path, edge_path, value):
        """
        Backup value through the tree and clear virtual losses.
        
        Args:
            node_path: Path of nodes visited
            edge_path: Path of edges visited
            value: Value to propagate
        """
        last_node_idx = len(node_path) - 1
        
        # Update node values
        for i in range(last_node_idx, -1, -1):
            node = node_path[i]
            is_from_child = (last_node_idx - i) % 2 == 1
            node.updateStats(value, is_from_child)
        
        # Clear virtual losses
        for edge in edge_path:
            if edge is not None:
                edge.removeVirtualLoss()

    def selectTask( self, board, node_path, edge_path ):
        """
        Do the selection stage of MCTS with virtual losses.

        Args/Returns:
            board (chess.Board) the root position on input,
                on return, either the positon of the selected unexpanded node,
                or the last node visited, if that is terminal
            node_path (list of Node) ordered list of nodes traversed
            edge_path (list of Edge) ordered list of edges traversed
        """

        cNode = self

        while True:

            node_path.append( cNode )

            cEdge = cNode.UCTSelect()

            edge_path.append( cEdge )

            if cEdge == None:

                #cNode is terminal. Return with board set to the same position as cNode
                #and edge_path[ -1 ] = None

                assert cNode.isTerminal()

                break
            
            cEdge.addVirtualLoss()

            board.push( cEdge.getMove() )

            if not cEdge.has_child():

                #cEdge has not been expanded. Return with board set to the same
                #position as the unexpanded Node

                break

            cNode = cEdge.getChild()

    def async_rollouts(self, board, num_rollouts, num_workers=None):
        """
        Perform rollouts using asynchronous neural network evaluation.
        
        Args:
            board: Current board position
            num_rollouts: Total number of rollouts to perform
            num_workers: Number of worker threads (auto-select if None)
        """
        if num_workers is None:
            num_workers = min(self.max_workers, max(16, num_rollouts // 50))
        
        # Initialize worker pool if needed
        if self.worker_pool is None or self.worker_pool._max_workers != num_workers:
            if self.worker_pool:
                self.worker_pool.shutdown(wait=False)
            self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        # Simple approach: each worker does a fixed number of rollouts
        rollouts_per_worker = num_rollouts // num_workers
        remainder = num_rollouts % num_workers
        
        def worker_task(worker_id, n_rollouts):
            """Worker function that performs n_rollouts."""
            for i in range(n_rollouts):
                try:
                    # Perform single rollout
                    board_copy = self.board.copy()
                    node_path = []
                    edge_path = []
                    
                    self.selectTask(board_copy, node_path, edge_path)
                    
                    edge = edge_path[-1]
                    
                    if edge is None:
                        # Terminal node
                        winner = encoder.parseResult(board_copy.result())
                        if not board_copy.turn:
                            winner *= -1
                        new_Q = float(winner) / 2. + 0.5
                        self._backup(node_path, edge_path, new_Q)
                    else:
                        # Request neural network evaluation
                        future = self.nn_server.evaluate_async(board_copy)
                        
                        # Get result and update tree
                        try:
                            value, move_probabilities = future.result(timeout=0.1)
                            new_Q = value / 2. + 0.5
                            
                            expanded = edge.expand(board_copy, new_Q, move_probabilities)
                            if not expanded:
                                self.same_paths += 1
                            
                            new_Q = 1. - new_Q
                            self._backup(node_path, edge_path, new_Q)
                            
                        except TimeoutError:
                            # Timeout - remove virtual losses but don't update values
                            for e in edge_path:
                                if e is not None:
                                    e.removeVirtualLoss()
                    
                except Exception as e:
                    print(f"Error in worker {worker_id} rollout {i}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Launch all workers
        futures = []
        for i in range(num_workers):
            n_rollouts = rollouts_per_worker
            if i < remainder:
                n_rollouts += 1
            
            if n_rollouts > 0:
                future = self.worker_pool.submit(worker_task, i, n_rollouts)
                futures.append(future)
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in rollout worker: {e}")
                import traceback
                traceback.print_exc()

    def getVisitCounts(self, board):
        """
        Get visit counts for all possible moves as a 4608-dimensional vector.
        This is used for training the policy network in reinforcement learning.
        
        Args:
            board (chess.Board): Current board position (needed for move encoding)
            
        Returns:
            numpy.array (4608,): Visit counts indexed by move encoding
        """
        visit_counts = np.zeros(4608, dtype=np.float32)
        
        for edge in self.edges:
            move = edge.getMove()
            
            # Use encoder to get the move index
            # Need to mirror if it's black's turn since encoder mirrors positions
            if not board.turn:
                # For black, we need to mirror the move
                from encoder import mirrorMove
                mirrored_move = mirrorMove(move)
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(mirrored_move)
            else:
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
            
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            visit_counts[moveIdx] = edge.getN()
        
        return visit_counts
    
    def cleanup(self):
        """Clean up thread pool when done"""
        if self.worker_pool is not None:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None


class MCTSEngine:
    """
    High-level MCTS engine that manages the async neural network server
    and provides a simple interface for move selection.
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
        
        # Create async neural network server
        self.nn_server = AsyncNeuralNetworkServer(
            neural_network, 
            device=device,
            max_batch_size=max_batch_size,
            max_wait_time=0.005,  # 5ms max wait for better batching
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
            print(f"\nMCTSEngine Statistics:")
            print(f"  Total rollouts: {self.total_rollouts}")
            print(f"  Total time: {self.total_time:.2f}s")
            print(f"  Average NPS: {self.total_rollouts / self.total_time:.0f}")
            
    def search(self, board, num_rollouts, return_root=False):
        """
        Search for the best move using MCTS with async neural network.
        
        Args:
            board: Current board position
            num_rollouts: Number of rollouts to perform
            return_root: If True, return the root node instead of best move
            
        Returns:
            Best move or root node if return_root=True
        """
        start_time = time.time()
        
        # Create root with async evaluation
        root = AsyncRoot(board, self.nn_server)
        
        # Perform async rollouts
        root.async_rollouts(board, num_rollouts, num_workers=self.num_workers)
        
        elapsed = time.time() - start_time
        self.total_rollouts += num_rollouts
        self.total_time += elapsed
        
        if self.verbose:
            nps = num_rollouts / elapsed if elapsed > 0 else 0
            print(f"Completed {num_rollouts} rollouts in {elapsed:.2f}s ({nps:.0f} NPS)")
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