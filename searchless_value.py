"""
Searchless Value-based Move Selection

This module provides a drop-in replacement for MCTS that selects moves
by evaluating all legal moves 1-ply deep using the neural network's value head.
No tree search is performed - just direct evaluation of resulting positions.
"""

import encoder
import chess
import torch
import numpy as np
from threading import Lock


class Edge:
    """
    Compatibility wrapper for move information.
    Mimics the MCTS Edge interface for seamless replacement.
    """
    
    def __init__(self, move, value, visits=1):
        """
        Args:
            move (chess.Move): The move this edge represents
            value (float): The value of the position after this move
            visits (int): Mock visit count for compatibility
        """
        self.move = move
        self.value = value
        self.visits = visits
        self.child = None
        self._lock = Lock()
    
    def getMove(self):
        """Returns the move."""
        return self.move
    
    def getN(self):
        """Returns mock visit count."""
        return self.visits
    
    def getQ(self):
        """Returns the value of this move."""
        return self.value
    
    def getP(self):
        """Returns a mock probability (not used in value selection)."""
        return 1.0 / 200.0


class Root:
    """
    Root node that performs value-based move selection.
    Compatible with MCTS.Root interface for drop-in replacement.
    """
    
    def __init__(self, board, neuralNetwork):
        """
        Create the root and immediately evaluate all legal moves.
        
        Args:
            board (chess.Board): The chess position
            neuralNetwork (torch.nn.Module): The neural network
        """
        self.board = board.copy()
        self.neuralNetwork = neuralNetwork
        self.edges = []
        self.best_edge = None
        self.N = 1
        self.same_paths = 0
        self.thread_pool = None
        
        # Evaluate current position for compatibility
        # Some code might expect these attributes
        value, move_probabilities = encoder.callNeuralNetwork(board, neuralNetwork)
        self.sum_Q = value / 2.0 + 0.5
        
        # Store initial evaluation
        self._evaluate_all_moves()
    
    def _evaluate_all_moves(self):
        """Evaluate all legal moves using the neural network's value head."""
        legal_moves = list(self.board.legal_moves)
        
        if not legal_moves:
            # Terminal position
            return
        
        # Create board positions for all legal moves
        boards = []
        for move in legal_moves:
            board_copy = self.board.copy()
            board_copy.push(move)
            boards.append(board_copy)
        
        # Batch evaluate all positions
        with torch.no_grad():
            values, _ = encoder.callNeuralNetworkBatched(boards, self.neuralNetwork)
        
        # Create edges for all moves with their values
        # Note: values are from the opponent's perspective after the move
        for i, move in enumerate(legal_moves):
            # Convert value to current player's perspective
            # Higher value for opponent = lower value for us
            move_value = 1.0 - (values[i] / 2.0 + 0.5)
            edge = Edge(move, move_value, visits=1)
            self.edges.append(edge)
        
        # Find best move based on value
        self._update_best_move()
    
    def _update_best_move(self):
        """Update the best move based on current evaluations."""
        if not self.edges:
            self.best_edge = None
            return
        
        # Select move with highest value for current player
        self.best_edge = max(self.edges, key=lambda e: e.getQ())
    
    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts, num_parallel_rollouts):
        """
        Compatibility method - doesn't actually do rollouts.
        Re-evaluates if called with significantly more rollouts.
        
        Args:
            board (chess.Board): The chess position (unused)
            neuralNetwork (torch.nn.Module): The neural network (unused)
            total_rollouts (int): Total number of rollouts (used for mock visit counts)
            num_parallel_rollouts (int): Parallel rollouts (unused)
        
        Returns:
            int: Number of rollouts "performed"
        """
        # Update mock visit counts for compatibility
        if self.edges and total_rollouts > 1:
            # Distribute visits proportionally based on values
            total_value = sum(max(0.01, edge.getQ()) for edge in self.edges)
            for edge in self.edges:
                edge.visits = max(1, int(total_rollouts * max(0.01, edge.getQ()) / total_value))
        
        self.N = total_rollouts
        return total_rollouts
    
    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts):
        """
        Compatibility wrapper for parallelRollouts.
        
        Args:
            board (chess.Board): The chess position
            neuralNetwork (torch.nn.Module): The neural network
            num_parallel_rollouts (int): Number of parallel rollouts
        """
        return self.parallelRolloutsTotal(board, neuralNetwork, num_parallel_rollouts, num_parallel_rollouts)
    
    def rollout(self, board, neuralNetwork):
        """
        Compatibility method for single rollout.
        Does nothing as evaluation is already complete.
        
        Args:
            board (chess.Board): The chess position
            neuralNetwork (torch.nn.Module): The neural network
        """
        pass
    
    def maxNSelect(self):
        """
        Returns the edge with the best value.
        
        Returns:
            Edge: The best move edge
        """
        return self.best_edge
    
    def UCTSelect(self):
        """
        Compatibility method - returns best value move.
        
        Returns:
            Edge: The best move edge
        """
        return self.best_edge
    
    def getN(self):
        """
        Returns mock total visit count.
        
        Returns:
            int: Total number of "visits"
        """
        return self.N
    
    def getQ(self):
        """
        Returns the value of the current position.
        
        Returns:
            float: Position value
        """
        return self.sum_Q / self.N
    
    def getStatisticsString(self):
        """
        Get a string containing move statistics.
        
        Returns:
            str: Statistics string for all moves
        """
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
                'move', 'value', 'visits', 'rank')
        
        # Sort edges by value
        sorted_edges = sorted(self.edges, key=lambda e: e.getQ(), reverse=True)
        
        for i, edge in enumerate(sorted_edges):
            move = edge.getMove()
            value = edge.getQ()
            visits = edge.getN()
            
            string += '|{: ^10}|{:10.4f}|{:10d}|{:10d}|\n'.format(
                str(move), value, visits, i+1)
        
        return string
    
    def getVisitCounts(self, board):
        """
        Get visit counts for all possible moves as a 4608-dimensional vector.
        For compatibility with training code.
        
        Args:
            board (chess.Board): Current board position
            
        Returns:
            numpy.array (4608,): Mock visit counts based on values
        """
        visit_counts = np.zeros(4608, dtype=np.float32)
        
        for edge in self.edges:
            move = edge.getMove()
            
            # Use encoder to get the move index
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
    
    def isTerminal(self):
        """
        Check if this position is terminal.
        
        Returns:
            bool: True if no legal moves
        """
        return len(self.edges) == 0
    
    def cleanup(self):
        """Compatibility method for cleanup - nothing to clean up."""
        pass


# For complete compatibility with MCTS module imports
Node = Root  # Alias for compatibility

def cleanup_engine():
    """Global cleanup function for compatibility."""
    pass