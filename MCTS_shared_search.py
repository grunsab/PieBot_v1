"""
Shared tree MCTS implementation with proper model handling for Windows.
"""

import chess
import math
import multiprocessing
from multiprocessing import Pool, Manager
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global variables for workers
worker_tree = None
worker_lock = None
worker_model = None
worker_C = None
worker_model_path = None

def worker_init(tree, lock, C, model_path):
    """Initialize worker process with shared data structures."""
    global worker_tree, worker_lock, worker_C, worker_model, worker_model_path
    worker_tree = tree
    worker_lock = lock
    worker_C = C
    worker_model_path = model_path
    
    # Load model in worker process to avoid pickling issues
    if model_path:
        import torch
        import AlphaZeroNetwork
        from device_utils import get_optimal_device, optimize_for_device
        
        device, _ = get_optimal_device()
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(model_path, map_location=device, weights_only=True)
        
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
        else:
            model.load_state_dict(weights)
        
        model = optimize_for_device(model, device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
            
        worker_model = model

def worker_run_simulation(board, sim_id):
    """Worker function that runs a single simulation."""
    global worker_tree, worker_lock, worker_model, worker_C
    
    path = []
    current_board = board.copy()
    
    # Selection phase
    current_fen = current_board.fen()
    
    while True:
        with worker_lock:
            if current_fen not in worker_tree:
                # Leaf node - expand and evaluate
                node_data = {
                    'visits': 0,
                    'value': 0,
                    'is_fully_expanded': False,
                    'untried_moves': [str(m) for m in current_board.legal_moves],
                    'children': {},
                    'parent': None,
                    'move': None
                }
                worker_tree[current_fen] = node_data
                break
            
            node = dict(worker_tree[current_fen])
            
            # Check if we need to expand
            if node['untried_moves']:
                # Expansion phase
                move_str = node['untried_moves'].pop()
                move = chess.Move.from_uci(move_str)
                node['is_fully_expanded'] = len(node['untried_moves']) == 0
                worker_tree[current_fen] = node
                
                # Create child node
                current_board.push(move)
                child_fen = current_board.fen()
                child_node_data = {
                    'visits': 0,
                    'value': 0,
                    'is_fully_expanded': False,
                    'untried_moves': [str(m) for m in current_board.legal_moves],
                    'children': {},
                    'parent': current_fen,
                    'move': move_str
                }
                worker_tree[child_fen] = child_node_data
                node['children'][move_str] = child_fen
                worker_tree[current_fen] = node
                
                path.append((current_fen, 1))
                current_fen = child_fen
                break
            elif node['children']:
                # Selection - choose best child
                best_move_str = select_best_child(current_fen, node, worker_tree, worker_C)
                if best_move_str is None:
                    break
                
                path.append((current_fen, 1))
                move = chess.Move.from_uci(best_move_str)
                current_board.push(move)
                current_fen = current_board.fen()
            else:
                # Terminal node
                break
    
    # Evaluation phase
    value = evaluate(current_board, worker_model)
    
    # Backpropagation phase
    path.append((current_fen, 1))
    backpropagate(path, value, worker_tree, worker_lock)

def select_best_child(parent_fen, parent_node, tree, C):
    """Select best child using UCB1."""
    best_move = None
    best_ucb = -float('inf')
    
    for move_str, child_fen in parent_node['children'].items():
        child_node = tree[child_fen]
        ucb = ucb1(child_node, parent_node, C)
        if ucb > best_ucb:
            best_ucb = ucb
            best_move = move_str
    
    return best_move

def evaluate(board, model):
    """Use the neural network to evaluate the position."""
    if model is None:
        # Random evaluation for testing without model
        return np.random.uniform(-1, 1)
    
    import encoder
    value, _ = encoder.callNeuralNetwork(board, model)
    return value

def backpropagate(path, value, tree, lock):
    """Backpropagate value through the path."""
    with lock:
        for fen, sign in reversed(path):
            if fen in tree:
                node = dict(tree[fen])
                node['visits'] += 1
                node['value'] += value * sign
                tree[fen] = node
                value = -value

def ucb1(node, parent_node, C):
    """Calculate UCB1 value."""
    if node['visits'] == 0:
        return float('inf')
    exploitation = node['value'] / node['visits']
    exploration = C * math.sqrt(math.log(parent_node['visits']) / node['visits'])
    return exploitation + exploration


class SharedTreeMCTS:
    def __init__(self, model_path="weights/AlphaZeroNet_20x256.pt", num_workers=4, C=1.5):
        self.model_path = model_path
        self.num_workers = num_workers
        self.C = C
        self.manager = Manager()
        self.tree = self.manager.dict()
        self.lock = self.manager.Lock()

    def search(self, board, num_rollouts):
        root_fen = board.fen()
        
        # Initialize root node if needed
        with self.lock:
            if root_fen not in self.tree:
                root_data = {
                    'visits': 0,
                    'value': 0,
                    'is_fully_expanded': False,
                    'untried_moves': [str(m) for m in board.legal_moves],
                    'children': {},
                    'parent': None,
                    'move': None
                }
                self.tree[root_fen] = root_data
        
        # Run parallel simulations
        with Pool(self.num_workers, initializer=worker_init, 
                  initargs=(self.tree, self.lock, self.C, self.model_path)) as pool:
            tasks = [(board.copy(), i) for i in range(num_rollouts)]
            pool.starmap(worker_run_simulation, tasks)
        
        # Get best move based on visit counts
        with self.lock:
            root_node = self.tree[root_fen]
            if not root_node['children']:
                return None
            
            best_move = None
            best_visits = -1
            for move_str, child_fen in root_node['children'].items():
                child_node = self.tree[child_fen]
                if child_node['visits'] > best_visits:
                    best_visits = child_node['visits']
                    best_move = chess.Move.from_uci(move_str)
            
            return best_move


# Compatibility wrapper to match original MCTS interface
class Root:
    _shared_mcts = None
    _lock = multiprocessing.Lock()
    
    def __init__(self, board, neuralNetwork=None):
        self.board = board.copy()
        self.neuralNetwork = neuralNetwork
        self.best_move = None
        
        # Initialize shared MCTS if needed
        with Root._lock:
            if Root._shared_mcts is None:
                # Use default model path
                Root._shared_mcts = SharedTreeMCTS(
                    model_path="weights/AlphaZeroNet_20x256.pt",
                    num_workers=4
                )
    
    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts, num_parallel):
        self.best_move = Root._shared_mcts.search(board, total_rollouts)
    
    def maxNSelect(self):
        if self.best_move is None:
            return None
        
        # Create compatibility wrapper
        class EdgeWrapper:
            def __init__(self, move):
                self._move = move
                self._n = 100
                self._q = 0.5
            
            def getMove(self):
                return self._move
            
            def getN(self):
                return self._n
            
            def getQ(self):
                return self._q
        
        return EdgeWrapper(self.best_move)
    
    @staticmethod
    def cleanup_engine():
        with Root._lock:
            Root._shared_mcts = None