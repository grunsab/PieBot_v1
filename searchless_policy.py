"""
Searchless Policy-based Move Selection

This module provides a drop-in replacement for MCTS that selects moves
based directly on the neural network's policy output for the current position.
No tree search or lookahead is performed - just direct policy-based selection
from the current board state.
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
    
    def __init__(self, move, policy_prob, visits=1):
        """
        Args:
            move (chess.Move): The move this edge represents
            policy_prob (float): The policy probability for this move
            visits (int): Mock visit count for compatibility
        """
        self.move = move
        self.policy_prob = policy_prob
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
        """Returns the policy probability as a quality metric."""
        return self.policy_prob
    
    def getP(self):
        """Returns the policy probability."""
        return self.policy_prob


class Root:
    """
    Root node that performs policy-based move selection.
    Compatible with MCTS.Root interface for drop-in replacement.
    """
    
    def __init__(self, board, neuralNetwork):
        """
        Create the root and immediately get policy for all legal moves.
        
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
        
        # Get neural network evaluation for current position
        with torch.no_grad():
            value, move_probabilities = encoder.callNeuralNetwork(board, neuralNetwork)
        
        # Store value for compatibility
        self.sum_Q = value / 2.0 + 0.5
        
        # Create edges for all legal moves with their policy probabilities
        self._setup_edges(move_probabilities)
    
    def _setup_edges(self, move_probabilities):
        """
        Set up edges for all legal moves with their policy probabilities.
        The policy probabilities directly correspond to legal moves in order.
        
        Args:
            move_probabilities (numpy.array): Policy probabilities from neural network
        """
        legal_moves = list(self.board.legal_moves)
        
        if not legal_moves:
            # Terminal position
            return
        
        # Create edges for all moves with their policy probabilities
        # The neural network returns probabilities for each legal move in order
        for i, move in enumerate(legal_moves):
            policy_prob = move_probabilities[i] if i < len(move_probabilities) else 0.0
            edge = Edge(move, policy_prob, visits=1)
            self.edges.append(edge)
        
        # Find best move based on policy
        self._update_best_move()
    
    def _update_best_move(self):
        """Update the best move based on policy probabilities."""
        if not self.edges:
            self.best_edge = None
            return
        
        # Select move with highest policy probability
        self.best_edge = max(self.edges, key=lambda e: e.getP())
    
    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts, num_parallel_rollouts):
        """
        Compatibility method - doesn't actually do rollouts.
        Updates mock visit counts based on policy distribution.
        
        Args:
            board (chess.Board): The chess position (unused)
            neuralNetwork (torch.nn.Module): The neural network (unused)
            total_rollouts (int): Total number of rollouts (used for mock visit counts)
            num_parallel_rollouts (int): Parallel rollouts (unused)
        
        Returns:
            int: Number of rollouts "performed"
        """
        # Update mock visit counts proportionally to policy probabilities
        if self.edges and total_rollouts > 1:
            # Ensure all probabilities sum to 1 for proper distribution
            total_prob = sum(edge.getP() for edge in self.edges)
            if total_prob > 0:
                for edge in self.edges:
                    # Distribute visits according to policy probability
                    edge.visits = max(1, int(total_rollouts * edge.getP() / total_prob))
        
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
        Does nothing as policy is already computed.
        
        Args:
            board (chess.Board): The chess position
            neuralNetwork (torch.nn.Module): The neural network
        """
        pass
    
    def maxNSelect(self):
        """
        Returns the edge with the highest policy probability.
        
        Returns:
            Edge: The best move edge
        """
        return self.best_edge
    
    def UCTSelect(self):
        """
        Compatibility method - returns best policy move.
        
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
            float: Position value from neural network
        """
        return self.sum_Q / self.N
    
    def getStatisticsString(self):
        """
        Get a string containing move statistics.
        
        Returns:
            str: Statistics string for all moves
        """
        string = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format(
                'move', 'policy', 'visits', 'rank')
        
        # Sort edges by policy probability
        sorted_edges = sorted(self.edges, key=lambda e: e.getP(), reverse=True)
        
        for i, edge in enumerate(sorted_edges):
            move = edge.getMove()
            policy = edge.getP()
            visits = edge.getN()
            
            string += '|{: ^10}|{:10.4f}|{:10d}|{:10d}|\n'.format(
                str(move), policy, visits, i+1)
        
        return string
    
    def getVisitCounts(self, board):
        """
        Get visit counts for all possible moves as a 4608-dimensional vector.
        For compatibility with training code.
        
        Args:
            board (chess.Board): Current board position
            
        Returns:
            numpy.array (4608,): Mock visit counts based on policy
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