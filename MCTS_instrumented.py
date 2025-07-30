import encoder
import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict

class PerformanceMonitor:
    """Track performance metrics for MCTS operations."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self._lock = Lock()
    
    def record_time(self, operation, duration):
        """Record timing for an operation."""
        with self._lock:
            self.timings[operation].append(duration)
            self.counts[operation] += 1
    
    def get_stats(self):
        """Get performance statistics."""
        stats = {}
        with self._lock:
            for op, times in self.timings.items():
                if times:
                    stats[op] = {
                        'count': self.counts[op],
                        'total': sum(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }
        return stats
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print("\n" + "="*80)
        print("MCTS Performance Statistics:")
        print("="*80)
        print(f"{'Operation':<30} {'Count':>10} {'Total(s)':>12} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        print("-"*80)
        
        for op, data in sorted(stats.items()):
            print(f"{op:<30} {data['count']:>10} {data['total']:>12.3f} "
                  f"{data['avg']*1000:>12.3f} {data['min']*1000:>12.3f} {data['max']*1000:>12.3f}")

# Global performance monitor
perf_monitor = PerformanceMonitor()

def calcUCT( edge, N_p ):
    """
    Calculate the UCT formula.

    Args:
        edge (Edge) the edge which the UCT formula is for
        N_p (float) the parents visit count

    Returns:
        (float) the calculated value
    """
    start_time = time.perf_counter()
    
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
    
    perf_monitor.record_time('calcUCT', time.perf_counter() - start_time)
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
        start_time = time.perf_counter()
        
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
        
        perf_monitor.record_time('Node.__init__', time.perf_counter() - start_time)

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
        start_time = time.perf_counter()

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

        perf_monitor.record_time('UCTSelect', time.perf_counter() - start_time)
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
        start_time = time.perf_counter()
        with self._lock:
            self.N += 1
            if from_child_perspective:
                self.sum_Q += 1. - value
            else:
                self.sum_Q += value
        perf_monitor.record_time('updateStats', time.perf_counter() - start_time)

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
        start_time = time.perf_counter()

        if self.child == None:

            self.child = Node( board, new_Q, move_probabilities )
            perf_monitor.record_time('expand', time.perf_counter() - start_time)
            return True

        else:
            perf_monitor.record_time('expand', time.perf_counter() - start_time)
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

    def clearVirtualLoss( self ):
        with self._lock:
            self.virtualLosses = 0.
   
class Root( Node ):

    def __init__( self, board, neuralNetwork ):
        """
        Create the root of the search tree.

        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network

        """
        start_time = time.perf_counter()
        
        value, move_probabilities = encoder.callNeuralNetwork( board, neuralNetwork )

        Q = value / 2. + 0.5

        super().__init__( board, Q, move_probabilities )

        self.same_paths = 0
        
        # Pre-create thread pool for reuse
        self.thread_pool = None
        self.max_workers = 20  # Maximum number of threads we might use
        
        perf_monitor.record_time('Root.__init__', time.perf_counter() - start_time)

    def selectTask( self, board, node_path, edge_path ):
        """
        Do the selection stage of MCTS.

        Args/Returns:
            board (chess.Board) the root position on input,
                on return, either the positon of the selected unexpanded node,
                or the last node visited, if that is terminal
            node_path (list of Node) ordered list of nodes traversed
            edge_path (list of Edge) ordered list of edges traversed
        """
        start_time = time.perf_counter()

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

        perf_monitor.record_time('selectTask', time.perf_counter() - start_time)

    def rollout( self, board, neuralNetwork ):
        """
        Each rollout traverses the tree until
        it reaches an un-expanded node or a terminal node.
        Unexpanded nodes are expanded and their
        win probability propagated.
        Terminal nodes have their win probability
        propagated as well.

        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network
        """
        start_time = time.perf_counter()
        
        node_path = []
        edge_path = []

        self.selectTask( board, node_path, edge_path )

        edge = edge_path[ -1 ]

        if edge != None:
            nn_start = time.perf_counter()
            value, move_probabilities = encoder.callNeuralNetwork( board, neuralNetwork )
            perf_monitor.record_time('neural_network_single', time.perf_counter() - nn_start)

            new_Q = value / 2. + 0.5

            edge.expand( board, new_Q, move_probabilities )

            new_Q = 1. - new_Q

        else:
            winner = encoder.parseResult( board.result() )

            if not board.turn:
                winner *= -1

            new_Q = float( winner ) / 2. + 0.5

        last_node_idx = len( node_path ) - 1
        
        backprop_start = time.perf_counter()
        for i in range( last_node_idx, -1, -1 ):

            node = node_path[ i ]
            
            # Use thread-safe update
            is_from_child = ( last_node_idx - i ) % 2 == 1
            node.updateStats(new_Q, is_from_child)

        for edge in edge_path:
                
            if edge != None:
               edge.clearVirtualLoss()

        perf_monitor.record_time('backpropagation', time.perf_counter() - backprop_start)
        perf_monitor.record_time('rollout', time.perf_counter() - start_time)


    def parallelRolloutsOptimized( self, board, neuralNetwork, num_parallel_rollouts ):
        """
        Optimized parallel rollouts using thread pool and batch processing.

        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network
            num_parallel_rollouts (int) the number of rollouts done in parallel
        """
        start_time = time.perf_counter()
        
        # Initialize thread pool if not already created
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        boards = []
        node_paths = []
        edge_paths = []
        
        # Pre-allocate lists
        for i in range( num_parallel_rollouts ):
            boards.append( board.copy() )
            node_paths.append( [] )
            edge_paths.append( [] )

        # Submit all selection tasks to thread pool
        select_start = time.perf_counter()
        futures = []
        for i in range( num_parallel_rollouts ):
            future = self.thread_pool.submit(self.selectTask, boards[i], node_paths[i], edge_paths[i])
            futures.append(future)
        
        # Wait for all selections to complete
        for future in futures:
            future.result()
        perf_monitor.record_time('parallel_selection', time.perf_counter() - select_start)

        # Batch neural network evaluation
        nn_start = time.perf_counter()
        values, move_probabilities = encoder.callNeuralNetworkBatched( boards, neuralNetwork )
        perf_monitor.record_time('neural_network_batch', time.perf_counter() - nn_start)

        # Process results and update tree
        update_start = time.perf_counter()
        for i in range( num_parallel_rollouts ):
            edge = edge_paths[ i ][ -1 ]
            board = boards[ i ]
            value = values[ i ]
            
            if edge != None:
                
                new_Q = value / 2. + 0.5
                
                isunexpanded = edge.expand( board, new_Q,
                        move_probabilities[ i ] )

                if not isunexpanded:
                    self.same_paths += 1

                new_Q = 1. - new_Q
                
            else:
                winner = encoder.parseResult( board.result() )

                if not board.turn:
                    winner *= -1

                new_Q = float( winner ) / 2. + 0.5

            last_node_idx = len( node_paths[ i ] ) - 1
            
            # Update node statistics with thread-safe method
            for r in range( last_node_idx, -1, -1 ):
               
                node = node_paths[ i ][ r ]
                is_from_child = ( last_node_idx - r ) % 2 == 1
                node.updateStats(new_Q, is_from_child)

            # Clear virtual losses
            for edge in edge_paths[ i ]:
                if edge != None:
                    edge.clearVirtualLoss()
        
        perf_monitor.record_time('parallel_update', time.perf_counter() - update_start)
        perf_monitor.record_time('parallelRollouts', time.perf_counter() - start_time)

    def parallelRollouts( self, board, neuralNetwork, num_parallel_rollouts ):
        """
        Wrapper to use optimized version while maintaining compatibility.
        """
        return self.parallelRolloutsOptimized(board, neuralNetwork, num_parallel_rollouts)

    def getVisitCounts(self, board):
        """
        Get visit counts for all possible moves as a 4608-dimensional vector.
        This is used for training the policy network in reinforcement learning.
        
        Args:
            board (chess.Board): Current board position (needed for move encoding)
            
        Returns:
            numpy.array (4608,): Visit counts indexed by move encoding
        """
        import numpy as np
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
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

