import encoder
import math
from threading import Thread, Lock, RLock
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from collections import deque
import torch
import hashlib
import multiprocessing as mp

# Global caches for optimization
position_cache = {}  # Cache for position encodings
legal_moves_cache = {}  # Cache for legal move generation
CACHE_MAX_SIZE = 20  # Maximum cache size to prevent memory issues

# Object pools for Node/Edge creation
node_pool = deque(maxlen=500)
edge_pool = deque(maxlen=200)

def get_position_hash(board):
    """Get a hash for the current board position."""
    return hashlib.md5(board.fen().encode()).hexdigest()

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

def vectorized_calcUCT(edges, N_p):
    """
    Vectorized UCT calculation for multiple edges.
    
    Args:
        edges (list of Edge): edges to calculate UCT for
        N_p (float): parent visit count
        
    Returns:
        numpy array of UCT values
    """
    n_edges = len(edges)
    if n_edges == 0:
        return np.array([])
    
    # Extract values in batch
    Q_values = np.array([edge.getQ() for edge in edges], dtype=np.float32)
    N_values = np.array([edge.getN() for edge in edges], dtype=np.float32)
    P_values = np.array([edge.getP() for edge in edges], dtype=np.float32)
    
    # Handle NaN values
    P_values = np.where(np.isnan(P_values), 1.0 / 200.0, P_values)
    
    C = 1.5
    sqrt_N_p = math.sqrt(N_p)
    
    # Vectorized UCT calculation
    UCT_values = Q_values + P_values * C * sqrt_N_p / (1 + N_values)
    
    # Handle NaN in results
    UCT_values = np.where(np.isnan(UCT_values), 0.0, UCT_values)
    
    return UCT_values

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
        self._lock = RLock()  # Use RLock for better performance
        self.N = 1.

        # Handle NaN values in Q
        if math.isnan(new_Q):
            self.sum_Q = 0.5  # Neutral value
        else:
            self.sum_Q = new_Q

        self.edges = []
        
        # Check legal moves cache
        board_hash = get_position_hash(board)
        if board_hash in legal_moves_cache:
            legal_moves = legal_moves_cache[board_hash]
        else:
            legal_moves = list(board.legal_moves)
            if len(legal_moves_cache) < CACHE_MAX_SIZE:
                legal_moves_cache[board_hash] = legal_moves

        for idx, move in enumerate(legal_moves):
            # Try to reuse Edge from pool
            if edge_pool:
                edge = edge_pool.pop()
                edge.reinit(move, move_probabilities[idx])
            else:
                edge = Edge(move, move_probabilities[idx])
            self.edges.append(edge)

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
        
        if not self.edges:
            return None
            
        # Use vectorized UCT calculation
        uct_values = vectorized_calcUCT(self.edges, self.N)
        
        if len(uct_values) == 0:
            return None
            
        max_idx = np.argmax(uct_values)
        return self.edges[max_idx]
    
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
    
    def cleanup(self):
        """Return edges to pool when node is no longer needed."""
        for edge in self.edges:
            edge.cleanup()
            if len(edge_pool) < edge_pool.maxlen:
                edge_pool.append(edge)
        self.edges.clear()

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
    
    def reinit(self, move, move_probability):
        """Reinitialize edge for reuse from pool."""
        self.move = move
        if math.isnan(move_probability) or move_probability < 0:
            self.P = 1.0 / 200.0
        else:
            self.P = move_probability
        self.child = None
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
            # Try to reuse Node from pool
            if node_pool:
                self.child = node_pool.pop()
                # Reinitialize the node - we need to implement a reinit method
                self.child.__init__(board, new_Q, move_probabilities)
            else:
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

    def addVirtualLoss( self, scale_factor=1.0 ):
        """
        When doing multiple rollouts in parallel,
        we can discourage threads from taking
        the same path by adding fake losses
        to visited nodes.
        
        Args:
            scale_factor (float): Scale factor for virtual loss based on parallelism
        """
        with self._lock:
            self.virtualLosses += scale_factor

    def clearVirtualLoss( self ):
        with self._lock:
            self.virtualLosses = 0.
    
    def cleanup(self):
        """Cleanup for returning to pool."""
        if self.child:
            self.child.cleanup()
            if len(node_pool) < node_pool.maxlen:
                node_pool.append(self.child)
        self.child = None
        self.virtualLosses = 0.
   
class Root( Node ):

    def __init__( self, board, neuralNetwork ):
        """
        Create the root of the search tree.

        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network

        """
        # Use optimized neural network call that keeps data on device
        value, move_probabilities = callNeuralNetworkOptimized( board, neuralNetwork )

        Q = value / 2. + 0.5

        super().__init__( board, Q, move_probabilities )

        self.same_paths = 0
        
        # Pre-create thread pool for reuse
        self.thread_pool = None
        self.max_workers = mp.cpu_count()  # Maximum number of threads we might use
        
        # Store the neural network device for optimization
        self.nn_device = next(neuralNetwork.parameters()).device
        
        # Position encoding cache for this tree
        self.position_cache_local = {}
        
        # Track parallelism level for adaptive virtual loss
        self.current_parallelism = 0

    def selectTask( self, board, node_path, edge_path, virtual_loss_scale=1.0 ):
        """
        Do the selection stage of MCTS with scaled virtual losses.

        Args/Returns:
            board (chess.Board) the root position on input,
                on return, either the positon of the selected unexpanded node,
                or the last node visited, if that is terminal
            node_path (list of Node) ordered list of nodes traversed
            edge_path (list of Edge) ordered list of edges traversed
            virtual_loss_scale (float) scale factor for virtual losses
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
            
            cEdge.addVirtualLoss(virtual_loss_scale)

            board.push( cEdge.getMove() )

            if not cEdge.has_child():

                #cEdge has not been expanded. Return with board set to the same
                #position as the unexpanded Node

                break

            cNode = cEdge.getChild()

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
        
        node_path = []
        edge_path = []

        self.selectTask( board, node_path, edge_path )

        edge = edge_path[ -1 ]

        if edge != None:
            value, move_probabilities = callNeuralNetworkOptimized( board, neuralNetwork )

            new_Q = value / 2. + 0.5

            edge.expand( board, new_Q, move_probabilities )

            new_Q = 1. - new_Q

        else:
            winner = encoder.parseResult( board.result() )

            if not board.turn:
                winner *= -1

            new_Q = float( winner ) / 2. + 0.5

        last_node_idx = len( node_path ) - 1
            
        for i in range( last_node_idx, -1, -1 ):

            node = node_path[ i ]
            
            # Use thread-safe update
            is_from_child = ( last_node_idx - i ) % 2 == 1
            node.updateStats(new_Q, is_from_child)

        for edge in edge_path:
                
            if edge != None:
               edge.clearVirtualLoss()


    def parallelRolloutsOptimized( self, board, neuralNetwork, num_parallel_rollouts ):
        """
        Optimized parallel rollouts using thread pool and batch processing.
        Now with adaptive virtual loss scaling based on parallelism level.

        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network
            num_parallel_rollouts (int) the number of rollouts done in parallel
        """
        
        # Initialize thread pool if not already created
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Calculate virtual loss scale based on parallelism
        # Higher parallelism = higher virtual loss to discourage path collision
        virtual_loss_scale = math.sqrt(num_parallel_rollouts / 10.0)  # Adaptive scaling

        boards = []
        node_paths = []
        edge_paths = []
        
        # Pre-allocate lists
        for i in range( num_parallel_rollouts ):
            boards.append( board.copy() )
            node_paths.append( [] )
            edge_paths.append( [] )

        # Submit all selection tasks to thread pool with scaled virtual loss
        futures = []
        for i in range( num_parallel_rollouts ):
            future = self.thread_pool.submit(
                self.selectTask, boards[i], node_paths[i], edge_paths[i], virtual_loss_scale
            )
            futures.append(future)
        
        # Wait for all selections to complete
        for future in futures:
            future.result()

        # Batch neural network evaluation - optimized version
        values, move_probabilities = callNeuralNetworkBatchedOptimized( boards, neuralNetwork )

        # Process results and update tree
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

    def parallelRolloutsProgressive( self, board, neuralNetwork, total_rollouts, 
                                   initial_batch_size=50, max_batch_size=512 ):
        """
        Progressive batching to reduce path collisions while maintaining high throughput.
        
        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network
            total_rollouts (int) total number of rollouts to perform
            initial_batch_size (int) starting batch size
            max_batch_size (int) maximum batch size
        """
        remaining = total_rollouts
        batch_size = initial_batch_size
        batch_count = 0
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            # Run batch with current parallelism level
            self.current_parallelism = current_batch
            self.parallelRolloutsOptimized(board, neuralNetwork, current_batch)
            
            remaining -= current_batch
            batch_count += 1
            
            # Gradually increase batch size to maintain efficiency
            # but cap it to prevent too many collisions
            if batch_count % 5 == 0:  # Every 5 batches
                batch_size = min(int(batch_size * 1.5), max_batch_size)

    def parallelRollouts( self, board, neuralNetwork, num_parallel_rollouts ):
        """
        Wrapper that uses progressive batching for very high parallelism.
        """
        # For very high parallelism, use progressive batching
        if num_parallel_rollouts > 200:
            return self.parallelRolloutsProgressive(board, neuralNetwork, num_parallel_rollouts)
        else:
            # For lower parallelism, use the standard optimized version
            return self.parallelRolloutsOptimized(board, neuralNetwork, num_parallel_rollouts)
    

    def parallelRolloutsTotal( self, board, neuralNetwork, total_rollouts, num_parallel_rollouts ):
        """
        Run total parallel rollouts (compatibility method).
        This is a wrapper for the optimized version.
        
        Args:
            board (chess.Board) the chess position
            neuralNetwork (torch.nn.Module) the neural network
            total_rollouts (int) total number of rollouts to perform
            num_parallel_rollouts (int) number of rollouts per batch
        """
        return self.parallelRollouts(board, neuralNetwork, total_rollouts)

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
        
        # Clean up the tree and return nodes/edges to pools
        super().cleanup()


# Optimized neural network calls that minimize CPU/GPU transfers

def callNeuralNetworkOptimized( board, neuralNetwork ):
    """
    Optimized version that keeps tensors on device as long as possible.
    Only transfers the final results to CPU.
    """
    board_hash = get_position_hash(board)
    
    # Check position cache
    if board_hash in position_cache:
        cached_position, cached_mask = position_cache[board_hash]
        position = cached_position.clone()
        mask = cached_mask.clone()
    else:
        position, mask = encoder.encodePositionForInference( board )
        position = torch.from_numpy( position )
        mask = torch.from_numpy( mask )
        
        # Cache the position if cache not full
        if len(position_cache) < CACHE_MAX_SIZE:
            position_cache[board_hash] = (position.clone(), mask.clone())
    
    # Ensure 4D tensor (batch dimension)
    if position.dim() == 3:
        position = position.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Get the device from the model parameters
    model_device = next(neuralNetwork.parameters()).device
    position = position.to(model_device)
    mask = mask.to(model_device)
    
    # Convert to half precision if model is FP16
    if next(neuralNetwork.parameters()).dtype == torch.float16:
        position = position.half()
        mask = mask.half()
    
    # Flatten mask to match expected shape
    mask_flat = mask.view(mask.shape[0], -1)
    
    with torch.no_grad():
        value, policy = neuralNetwork( position, policyMask=mask_flat )
    
    # Only convert to CPU at the very end
    value = value.item()  # More efficient than .cpu().numpy()[0, 0]
    policy = policy[0].cpu().numpy()  # Keep on GPU until needed
    
    move_probabilities = encoder.decodePolicyOutput( board, policy )
    
    return value, move_probabilities


def callNeuralNetworkBatchedOptimized( boards, neuralNetwork ):
    """
    Optimized batch version that minimizes CPU/GPU transfers.
    """
    num_inputs = len( boards )
    
    # Collect unique positions to avoid redundant encoding
    unique_positions = {}
    board_hashes = []
    
    for board in boards:
        board_hash = get_position_hash(board)
        board_hashes.append(board_hash)
        
        if board_hash not in unique_positions:
            if board_hash in position_cache:
                unique_positions[board_hash] = position_cache[board_hash]
            else:
                position, mask = encoder.encodePositionForInference(board)
                position_tensor = torch.from_numpy(position)
                mask_tensor = torch.from_numpy(mask)
                unique_positions[board_hash] = (position_tensor, mask_tensor)
                
                # Cache if not full
                if len(position_cache) < CACHE_MAX_SIZE:
                    position_cache[board_hash] = (position_tensor.clone(), mask_tensor.clone())
    
    # Create batch tensors
    inputs = torch.zeros( (num_inputs, 16, 8, 8), dtype=torch.float32 )
    masks = torch.zeros( (num_inputs, 72, 8, 8), dtype=torch.float32 )
    
    # Fill batch tensors
    for i, board_hash in enumerate(board_hashes):
        position, mask = unique_positions[board_hash]
        inputs[i] = position
        masks[i] = mask
    
    # Get the device from the model parameters
    model_device = next(neuralNetwork.parameters()).device
    inputs = inputs.to(model_device)
    masks = masks.to(model_device)
    
    # Convert to half precision if model is FP16
    if next(neuralNetwork.parameters()).dtype == torch.float16:
        inputs = inputs.half()
        masks = masks.half()
    
    # Flatten masks to match expected shape
    masks_flat = masks.view(masks.shape[0], -1)
    
    with torch.no_grad():
        value, policy = neuralNetwork( inputs, policyMask=masks_flat )
    
    # Process outputs more efficiently
    move_probabilities = np.zeros( ( num_inputs, 200 ), dtype=np.float32 )
    
    # Convert values to numpy in one operation
    value = value.cpu().numpy().reshape( (num_inputs) )
    
    # Keep policy on GPU until needed for each board
    policy_cpu = policy.cpu().numpy()
    
    for i in range( num_inputs ):
        move_probabilities_tmp = encoder.decodePolicyOutput( boards[ i ], policy_cpu[ i ] )
        move_probabilities[ i, : move_probabilities_tmp.shape[0] ] = move_probabilities_tmp
    
    return value, move_probabilities

# Clear caches periodically to prevent memory issues
def clear_caches():
    """Clear all caches to free memory."""
    global position_cache, legal_moves_cache
    position_cache.clear()
    legal_moves_cache.clear()

# Clear object pools
def clear_pools():
    """Clear object pools."""
    global node_pool, edge_pool
    node_pool.clear()
    edge_pool.clear()