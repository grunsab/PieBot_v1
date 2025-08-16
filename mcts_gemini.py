import math
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import hashlib
import multiprocessing as mp
import encoder  # Assuming encoder module is available
import chess
import time
import atexit

# Configuration
C_PUCT = 1.5 # Standard AlphaZero/LC0 configuration

# Caching Configuration
CACHE_MAX_SIZE = 200000 # Adjust based on available memory.
LEGAL_MOVES_CACHE = {}
LEGAL_MOVES_CACHE_LOCK = RLock()

# Enhanced encoder support (if available)
try:
    from encoder_enhanced import PositionHistory
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    # print("Warning: encoder_enhanced not available, history tracking disabled")

# --- Helper Functions ---

def get_position_hash(board):
    """Get a hash for the current board position based on FEN."""
    # Using FEN for hashing as it includes necessary info like castling/en passant
    return hashlib.md5(board.fen().encode()).hexdigest()

def get_legal_moves_cached(board):
    """Get legal moves with efficient caching (Double-Checked Locking)."""
    board_hash = get_position_hash(board)
    
    # 1. Optimistic read (lock-free)
    if board_hash in LEGAL_MOVES_CACHE:
        return LEGAL_MOVES_CACHE[board_hash]
        
    # 2. If not found, acquire lock
    with LEGAL_MOVES_CACHE_LOCK:
        # 3. Check again inside the lock (in case another thread added it)
        if board_hash in LEGAL_MOVES_CACHE:
            return LEGAL_MOVES_CACHE[board_hash]
        
        legal_moves = list(board.legal_moves)
        
        # Cache management
        if len(LEGAL_MOVES_CACHE) < CACHE_MAX_SIZE:
            LEGAL_MOVES_CACHE[board_hash] = legal_moves
            
        return legal_moves

# --- Optimized MCTS Structure ---

class NumpyNode:
    """
    A node optimized for performance using NumPy arrays (Structure of Arrays design).
    Eliminates the Edge class for better memory locality and faster vectorized computation.
    """
    # Use __slots__ to minimize memory footprint of individual node objects
    __slots__ = ('_lock', 'N_total', 'moves', 'P', 'N', 'W', 'children', 'virtual_losses', 'num_edges')

    def __init__(self, move_probabilities, legal_moves):
        """
        Args:
            move_probabilities (numpy.array): Normalized probability distribution across legal moves.
            legal_moves (list): List of chess.Move objects.
        """
        # RLock is used ONLY for updates (VL, Backpropagation, Expansion). Selection reads are lock-free.
        self._lock = RLock() 
        
        # Total visit count of this node (N_p - Parent visits)
        self.N_total = 1.0

        self.num_edges = len(legal_moves)
        
        if self.num_edges > 0:
            self.moves = legal_moves
            self.P = move_probabilities # Assumes input is normalized and validated

            # Edge statistics stored in contiguous arrays (SoA)
            self.N = np.zeros(self.num_edges, dtype=np.float32)
            # W (Total action value) from the perspective of the current player
            self.W = np.zeros(self.num_edges, dtype=np.float32)
            self.virtual_losses = np.zeros(self.num_edges, dtype=np.float32)
            
            # Children nodes (initialized to None)
            self.children = [None] * self.num_edges

    def isTerminal(self):
        return self.num_edges == 0

    def UCTSelect(self):
        """
        Select the edge maximizing the PUCT formula using vectorized operations.
        Performs optimistic (lock-free) reads for maximum parallelism.
        Returns:
            (int) index of the selected edge, or None if terminal.
        """
        if self.num_edges == 0:
            return None
            
        # --- Critical Performance Section: Lock-Free Reads ---
        # We read N, W, VL, and N_total optimistically. MCTS is robust to slightly stale data.
        
        # Apply Virtual Loss (VL) effect during calculation.
        # VL increases the effective N and decreases the perceived W (acts as a temporary loss).
        # Note: The virtual_losses array contains the accumulated (potentially scaled) VL values.
        N_adjusted = self.N + self.virtual_losses
        # When we apply VL, we assume a loss (W decreases).
        W_adjusted = self.W - self.virtual_losses 

        # Calculate Q (Mean action value). Q=0 for unvisited nodes (standard practice).
        # Efficient and safe division using numpy
        Q = np.divide(W_adjusted, N_adjusted, out=np.zeros_like(W_adjusted), where=(N_adjusted!=0))

        # Calculate U (Exploration bonus)
        # U = P * C * sqrt(N_p) / (1 + N_c)
        # We use N_total (parent visits) for sqrt_N_p.
        sqrt_N_p = math.sqrt(self.N_total) 
        
        U = self.P * C_PUCT * sqrt_N_p / (1.0 + N_adjusted)
        
        PUCT = Q + U
        
        # Final safety check for NaNs
        if np.isnan(PUCT).any():
             # If NaNs occur, replace them with 0.0 to allow argmax to proceed
             PUCT = np.nan_to_num(PUCT, nan=0.0)

        max_idx = np.argmax(PUCT)
        return max_idx

    def addVirtualLoss(self, edge_idx, loss_value):
        """
        Apply virtual loss. Thread-safe (Locked).
        Args:
            loss_value (float): The amount of virtual loss to add (can be scaled).
        """
        with self._lock:
            self.virtual_losses[edge_idx] += loss_value

    def updateStatsAndClearVL(self, edge_idx, value, virtual_loss_value):
        """
        Update statistics during backpropagation and clear the virtual loss applied during selection. Thread-safe (Locked).

        Args:
            edge_idx (int or None): Index of the edge traversed (None if terminal leaf update).
            value (float): The value to backpropagate (from the perspective of the current player).
            virtual_loss_value (float): The amount of virtual loss to clear (must match what was added).
        """
        with self._lock:
            self.N_total += 1.0
            if edge_idx is not None:
                self.N[edge_idx] += 1.0
                self.W[edge_idx] += value
                # Clear the virtual loss applied during selection
                self.virtual_losses[edge_idx] -= virtual_loss_value

    def expand_node(self, edge_idx, move_probabilities, legal_moves):
        """Helper function for safe parallel expansion using Double-Checked Locking."""
        # 1. Optimistic check
        if self.children[edge_idx] is not None:
            return False
            
        with self._lock:
            # 2. Check again inside the lock
            if self.children[edge_idx] is not None:
                return False
            
            # Create the new node
            self.children[edge_idx] = NumpyNode(move_probabilities, legal_moves)
            return True

    def maxNSelect(self):
        """
        Returns the index of the edge with the maximum visit count (for final move selection).
        """
        if self.num_edges == 0:
            return None

        # Read N optimistically (VL is ignored for final move selection)
        return np.argmax(self.N)

    def getMove(self, idx):
        return self.moves[idx]
        
    def getQ(self, idx):
        # Helper to get the Q value of a specific edge (optimistic read)
        N = self.N[idx]
        W = self.W[idx]
        if N == 0:
            return 0.5 # Default Q for unvisited nodes
        return W / N

    def getStatisticsString(self):
        """
        Get a string containing the current search statistics.
        """
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
                'move', 'P', 'N', 'Q', 'PUCT' )

        if self.num_edges == 0:
            return "Terminal Node\n"

        # Read statistics (optimistic). We assume the search has stopped, so VL should be zero.
        N = self.N
        W = self.W
        P = self.P
        N_total = self.N_total

        # Calculate stats for display
        Q_values = np.divide(W, N, out=np.zeros_like(W), where=(N!=0))
        sqrt_N_p = math.sqrt(N_total)
        U_values = P * C_PUCT * sqrt_N_p / (1.0 + N)
        PUCT_values = Q_values + U_values

        # Sort by visit count (descending)
        sorted_indices = np.argsort(N)[::-1]

        for idx in sorted_indices:
            move = str(self.moves[idx])
            P_val = P[idx]
            N_val = N[idx]
            Q_val = Q_values[idx]
            PUCT_val = PUCT_values[idx]

            # Display N as float for compatibility with the original implementation format
            string += '|{: ^10}|{:10.4f}|{:10.4f}|{:10.4f}|{:10.4f}|\n'.format(
                move, P_val, N_val, Q_val, PUCT_val )

        return string

class Root(NumpyNode):
    """
    The root of the search tree. Manages the parallel MCTS process.
    """
    # Global Thread Pool shared across Root instances for efficiency
    _shared_thread_pool = None
    _pool_lock = RLock()

    def __init__(self, board, neuralNetwork, position_history=None, use_enhanced_encoder=False):
        
        self.use_enhanced_encoder = use_enhanced_encoder
        self.position_history = position_history if use_enhanced_encoder and HISTORY_AVAILABLE else None
        
        # Evaluate the root position
        # NN returns value (-1 to 1) and full policy vector (e.g., 4608)
        self.root_value, move_probabilities_full = callNeuralNetworkOptimized(board, neuralNetwork, self.position_history)

        # Get legal moves for the root
        legal_moves = get_legal_moves_cached(board)
        
        # Extract and normalize probabilities corresponding to legal moves
        move_probabilities = extract_and_normalize_policy(board, move_probabilities_full, legal_moves)

        super().__init__(move_probabilities, legal_moves)

        self.same_paths = 0 # Track expansion race conditions (racy updates are fine for monitoring)
        
        # Initialize Global Thread Pool Executor if not already initialized
        with Root._pool_lock:
            if Root._shared_thread_pool is None:
                # Use all available cores for CPU intensive tasks (selection/update)
                max_workers = mp.cpu_count() 
                Root._shared_thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        self.thread_pool = Root._shared_thread_pool

        # Store device if model is valid
        try:
            self.nn_device = next(neuralNetwork.parameters()).device
        except (StopIteration, AttributeError):
            self.nn_device = torch.device("cpu")

    # Compatibility methods for original interface
    def getN(self):
        return self.N_total

    def getQ(self):
        # Return the Q value of the root (0 to 1)
        return self.root_value / 2.0 + 0.5

    def selectTask(self, board, path_indices, virtual_loss_value):
        """
        Selection stage of MCTS. Executed in parallel threads.
        
        Args:
            board (chess.Board): The board state (will be modified in place).
            path_indices (list): List to store the path taken [(Node, edge_idx), ...].
            virtual_loss_value (float): The VL value to apply during this traversal.
        """
        cNode = self

        while True:
            # Selection using optimistic reads (Lock-free)
            edge_idx = cNode.UCTSelect()

            if edge_idx == None:
                # Terminal node reached
                path_indices.append((cNode, None))
                break

            # Apply virtual loss (Locked operation)
            # This immediately discourages other threads from taking this path.
            cNode.addVirtualLoss(edge_idx, virtual_loss_value)
            path_indices.append((cNode, edge_idx))

            # Update board state (CPU intensive step in python-chess)
            move = cNode.getMove(edge_idx)
            board.push(move)

            # Check if expanded (Optimistic check)
            if cNode.children[edge_idx] is None:
                # Unexpanded node. Selection phase ends here.
                break

            cNode = cNode.children[edge_idx]

    def updateTask(self, board, path_indices, value, move_probs_full, virtual_loss_value):
        """
        Expansion and Backpropagation stage. Executed in parallel threads.
        
        Args:
            board (chess.Board): The board state at the leaf node.
            path_indices (list): The path taken during selection.
            value (float): The NN evaluation (-1 to 1) if applicable.
            move_probs_full (np.array): The NN policy output if applicable.
            virtual_loss_value (float): The VL value applied during selection (to be cleared).
        """
        
        node, edge_idx = path_indices[-1]

        # --- Expansion Phase ---
        if edge_idx is not None:
            # Expansion needed (NN evaluation was performed)
            
            # Prepare data for the child node
            child_legal_moves = get_legal_moves_cached(board)
            move_probabilities = extract_and_normalize_policy(board, move_probs_full, child_legal_moves)

            # Attempt expansion (Thread-safe, Double-Checked Locking)
            is_expanded = node.expand_node(edge_idx, move_probabilities, child_legal_moves)

            if not is_expanded:
                # Another thread expanded this node first (path collision)
                self.same_paths += 1 # Racy update, used for monitoring

            # Value from NN is (-1 to 1). Convert to Q (0 to 1) from child perspective.
            new_Q = value / 2.0 + 0.5
            
            # Value to backpropagate starting from the perspective of the new child
            backprop_start_value = new_Q
            
        else:
            # Terminal node: Calculate game result (No NN evaluation needed)
            try:
                # Use claim_draw=True if available for accurate evaluation during search (handles repetition)
                result = board.result(claim_draw=True)
            except TypeError:
                result = board.result() # Fallback for older python-chess versions

            winner = encoder.parseResult(result) # 1 (White win), 0 (Draw), -1 (Black win)

            # Adjust perspective based on the turn at the terminal node
            # If it's Black's turn (board.turn=False), White just moved and caused the termination.
            if not board.turn: 
                winner *= -1

            # Convert result (-1 to 1) to Q (0 to 1)
            backprop_start_value = float(winner) / 2.0 + 0.5

        # --- Backpropagation Phase ---
        last_node_idx = len(path_indices) - 1
        
        # Iterate backwards from the leaf node up to the root
        for j in range(last_node_idx, -1, -1):
            cNode, cEdge_idx = path_indices[j]
            
            # Determine the value for W update (from cNode's perspective)
            # The perspective alternates at each level.
            # If depth difference is even, perspective is the same as the leaf.
            if (last_node_idx - j) % 2 == 0:
                value_for_W = backprop_start_value
            else:
                # Opposite perspective
                value_for_W = 1.0 - backprop_start_value
            
            # Determine virtual loss to clear.
            # VL was applied to the edge taken *from* this node (cEdge_idx).
            vl_to_clear = virtual_loss_value if cEdge_idx is not None else 0.0

            # Update stats (Locked operation)
            cNode.updateStatsAndClearVL(cEdge_idx, value_for_W, vl_to_clear)


    def parallelRolloutsOptimized(self, board, neuralNetwork, batch_size):
        """
        Executes a single batch of MCTS: Parallel Selection -> Batched Evaluation -> Parallel Updates.
        Includes Adaptive Virtual Loss Scaling to improve diversity when batch_size is large.
        """
        
        if batch_size == 0:
            return

        # --- Adaptive Virtual Loss Scaling ---
        # This helps reduce path collisions when batch_size is large. 
        # A simple logarithmic scaling based on the batch size.
        # VL = 1.0 (base) + log(BatchSize + 1) / 2.5 (Tuning factor)
        # E.g., B=64 -> VL ~ 2.66; B=512 -> VL ~ 3.49
        VIRTUAL_LOSS_SCALE_FACTOR = 2.5
        virtual_loss_value = 1.0 + math.log(batch_size + 1) / VIRTUAL_LOSS_SCALE_FACTOR

        # Prepare data structures for the batch
        boards = []
        path_indices_list = []
        histories = []
        
        # Setup the parallel tasks
        for i in range(batch_size):
            boards.append(board.copy()) # Copying the board is necessary as selectTask modifies it
            path_indices_list.append([])
            
            # Handle position history if needed (for specialized encoders)
            if self.use_enhanced_encoder and self.position_history and HISTORY_AVAILABLE:
                history_copy = PositionHistory(self.position_history.history_length)
                history_copy.history = self.position_history.history.copy()
                histories.append(history_copy)
            else:
                histories.append(None)

        # --- Phase 1: Selection (Parallel CPU) ---
        # Submit all selection tasks to the thread pool
        futures_selection = []
        for i in range(batch_size):
            future = self.thread_pool.submit(
                self.selectTask, boards[i], path_indices_list[i], virtual_loss_value
            )
            futures_selection.append(future)
        
        # Wait for all selections (Barrier 1)
        for future in futures_selection:
            future.result()
        
        # Update histories if applicable (must happen after selection, before evaluation)
        if self.use_enhanced_encoder and HISTORY_AVAILABLE and self.position_history:
            for i in range(batch_size):
                if histories[i]:
                   histories[i].add_position(boards[i])

        # --- Phase 2: Evaluation (Batched GPU) ---
        # Identify nodes needing NN evaluation (non-terminal leaves)
        eval_indices = []
        boards_to_eval = []
        histories_to_eval = []
        
        for i in range(batch_size):
            # Check the last element of the path: (Node, edge_index). 
            # If edge_index is None, the selection ended at a terminal node.
            if path_indices_list[i][-1][1] is not None: 
                eval_indices.append(i)
                boards_to_eval.append(boards[i])
                histories_to_eval.append(histories[i])

        # Initialize result containers
        values = np.zeros(batch_size, dtype=np.float32)
        move_probabilities_full_list = [None] * batch_size

        if boards_to_eval:
            # Batched NN evaluation (GPU intensive)
            nn_values, nn_move_probs_full = callNeuralNetworkBatchedOptimized(boards_to_eval, neuralNetwork, histories_to_eval)
            
            # Distribute results back to the corresponding tasks
            for i, idx in enumerate(eval_indices):
                values[idx] = nn_values[i]
                move_probabilities_full_list[idx] = nn_move_probs_full[i]

        # --- Phase 3: Expansion and Backpropagation (Parallel CPU) ---
        futures_update = []
        for i in range(batch_size):
            future = self.thread_pool.submit(
                self.updateTask, 
                boards[i], 
                path_indices_list[i], 
                values[i], 
                move_probabilities_full_list[i], 
                virtual_loss_value # Pass the VL value so it can be cleared correctly
            )
            futures_update.append(future)

        # Wait for all updates (Barrier 2)
        for future in futures_update:
            future.result()


    def parallelRollouts(self, board, neuralNetwork, total_rollouts):
        """
        Main entry point for running rollouts. Implements Progressive Batch Sizing (Ramped Batching).
        """
        remaining = total_rollouts
        
        # --- Progressive Batch Sizing Configuration ---
        # Tuned for balancing quality (low collisions) and throughput (high NPS) on high-end hardware.
        
        # Start small to encourage initial diversity when statistics are unstable.
        INITIAL_BATCH_SIZE = 32 
        
        # Maximum size to saturate the RTX 4080. 512 is often a good saturation point.
        MAX_BATCH_SIZE = 512 
        
        # How quickly to increase the batch size. 1.5x increase per iteration ramps up quickly.
        GROWTH_FACTOR = 1.5
        # ------------------------------------------
        
        current_batch_size = float(INITIAL_BATCH_SIZE)

        while remaining > 0:
            # Determine the size for this iteration, ensuring it's an integer and doesn't exceed remaining.
            batch_size_int = int(round(current_batch_size))
            actual_batch = min(remaining, batch_size_int)
            
            if actual_batch <= 0:
                break

            # Execute the batch
            self.parallelRolloutsOptimized(board, neuralNetwork, actual_batch)
            
            remaining -= actual_batch
            
            # Increase batch size for the next iteration, capping at MAX_BATCH_SIZE
            current_batch_size = min(current_batch_size * GROWTH_FACTOR, float(MAX_BATCH_SIZE))


    # Compatibility alias for the original interface
    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts, num_parallel_rollouts_ignored=None):
        self.parallelRollouts(board, neuralNetwork, total_rollouts)

    def maxNSelect(self):
        """
        Returns a wrapper object compatible with the original Edge interface for the best move.
        """
        idx = super().maxNSelect()
        if idx is None:
            return None
            
        # Lightweight wrapper class for compatibility with the existing engine interface
        class EdgeWrapper:
            def __init__(self, move, N, Q):
                self.move = move
                self.N = N
                self.Q = Q
            def getMove(self): return self.move
            def getN(self): return self.N
            def getQ(self): return self.Q

        move = self.getMove(idx)
        # Get stats (optimistic read)
        N = self.N[idx]
        Q = self.getQ(idx)
        
        return EdgeWrapper(move, N, Q)

    @staticmethod
    def cleanup_global_pool():
        """Clean up the shared thread pool when the engine shuts down."""
        with Root._pool_lock:
            if Root._shared_thread_pool is not None:
                Root._shared_thread_pool.shutdown(wait=True)
                Root._shared_thread_pool = None

    # Alias for compatibility with original cleanup call
    def cleanup(self):
        # Pool is managed globally, no per-search cleanup needed.
        pass 

# Register cleanup for the global thread pool when the script exits
atexit.register(Root.cleanup_global_pool)


# --- Neural Network Interface and Policy Handling ---
# These rely on the encoder module handling the actual inference efficiently.

def callNeuralNetworkOptimized(board, neuralNetwork, history=None):
    """
    Optimized single inference (delegates to encoder).
    """
    # Assumes encoder.callNeuralNetwork handles efficient CPU/GPU transfer
    return encoder.callNeuralNetwork(board, neuralNetwork, history)

def callNeuralNetworkBatchedOptimized(boards, neuralNetwork, histories=None):
    """
    Optimized batch inference (delegates to encoder).
    """
    # Assumes encoder.callNeuralNetworkBatched handles batching and efficient transfer
    return encoder.callNeuralNetworkBatched(boards, neuralNetwork, histories)

def extract_and_normalize_policy(board, move_probabilities_full, legal_moves):
    """
    Extracts the probabilities for the legal moves from the full policy vector,
    handles mirroring for Black's turn (if required by the encoder), and normalizes the result.
    """
    num_moves = len(legal_moves)
    if num_moves == 0:
        return np.array([], dtype=np.float32)

    # Optimization: Ensure move_probabilities_full is a numpy array 
    # (in case the encoder returns a torch tensor on CPU)
    if isinstance(move_probabilities_full, torch.Tensor):
        move_probabilities_full = move_probabilities_full.detach().cpu().numpy()

    probs = np.zeros(num_moves, dtype=np.float32)
    is_black = not board.turn
    
    # Handle mirroring if necessary (common in AlphaZero/LC0 style networks)
    mirrorMoveFunc = None
    if is_black:
        try:
            from encoder import mirrorMove
            mirrorMoveFunc = mirrorMove
        except ImportError:
            # Basic fallback (might not handle promotions correctly depending on encoder specifics)
            mirrorMoveFunc = lambda m: chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)

    # Extract probabilities using move indices defined by the encoder
    for i, move in enumerate(legal_moves):
        try:
            if is_black:
                move_to_encode = mirrorMoveFunc(move)
            else:
                move_to_encode = move
                
            planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move_to_encode)
            
            # Calculate index in the flattened policy vector (Assumes standard structure e.g. Plane*64 + Rank*8 + File)
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            
            if moveIdx < len(move_probabilities_full):
                probs[i] = move_probabilities_full[moveIdx]
        except Exception as e:
            # Handle potential errors during move encoding gracefully
            # print(f"Error extracting policy for move {move}: {e}")
            continue
        
    # Normalize the probabilities
    probs = np.nan_to_num(probs, nan=0.0) # Handle potential NaNs from NN output
    sum_P = np.sum(probs)
    
    if sum_P > 1e-6:
        # Normalize if the sum is non-zero
        return probs / sum_P
    else:
        # Fallback to uniform distribution if all probabilities are zero or invalid
        return np.full(num_moves, 1.0/num_moves, dtype=np.float32)

# --- Cache Management ---
def clear_caches():
    """Clear all caches to free memory."""
    global LEGAL_MOVES_CACHE
    with LEGAL_MOVES_CACHE_LOCK:
        LEGAL_MOVES_CACHE.clear()