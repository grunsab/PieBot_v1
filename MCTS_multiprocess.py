import encoder
import math
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import numpy as np
import chess
import torch
import pickle
from functools import partial
import MCTS

def worker_rollouts(args):
    """
    Worker function to perform rollouts in a separate process.
    
    Args:
        args: Tuple of (board_fen, model_path, num_rollouts, num_threads_per_rollout, device_str)
    
    Returns:
        Serialized statistics from the rollouts
    """
    board_fen, model_path, num_rollouts, num_threads_per_rollout, device_str = args
    
    # Set up the device in this process
    if device_str.startswith('cuda'):
        device = torch.device(device_str)
    else:
        device = torch.device('cpu')
    
    # Load the model in this process
    import AlphaZeroNetwork
    from device_utils import optimize_for_device
    
    # Load model weights
    weights = torch.load(model_path, map_location='cpu')
    
    # Check if it's FP16 model
    if isinstance(weights, dict) and weights.get('model_type') == 'fp16':
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        model.load_state_dict(weights['model_state_dict'])
        model = model.half()
    else:
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
        else:
            model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Create board from FEN
    board = chess.Board(board_fen)
    
    # Create local root node
    with torch.no_grad():
        root = MCTS.Root(board, model)
        
        # Perform rollouts
        for _ in range(num_rollouts):
            root.parallelRollouts(board.copy(), model, num_threads_per_rollout)
    
    # Collect statistics from the tree
    stats = {
        'N': root.N,
        'sum_Q': root.sum_Q,
        'edges': []
    }
    
    for edge in root.edges:
        edge_stats = {
            'move': edge.move.uci(),
            'P': edge.P,
            'child_N': edge.child.N if edge.has_child() else 0,
            'child_sum_Q': edge.child.sum_Q if edge.has_child() else 0,
            'has_child': edge.has_child()
        }
        stats['edges'].append(edge_stats)
    
    return stats


class MultiprocessRoot(MCTS.Root):
    """
    Extended Root class that supports multiprocessing.
    """
    
    def __init__(self, board, neuralNetwork):
        super().__init__(board, neuralNetwork)
        self.model_path = None  # Will be set by the caller
        self.device_str = str(next(neuralNetwork.parameters()).device)
    
    def merge_stats(self, other_stats):
        """
        Merge statistics from another process into this tree.
        
        Args:
            other_stats: Dictionary containing statistics from another process
        """
        # Update root node statistics
        self.N += other_stats['N'] - 1  # Subtract 1 because root starts with N=1
        self.sum_Q += other_stats['sum_Q'] - self.sum_Q / self.N  # Adjust for initial Q
        
        # Merge edge statistics
        for edge_idx, edge_stats in enumerate(other_stats['edges']):
            edge = self.edges[edge_idx]
            
            if edge_stats['has_child']:
                if not edge.has_child():
                    # Create child node if it doesn't exist
                    board_copy = self.board.copy()
                    board_copy.push(edge.move)
                    
                    # Calculate initial Q for the child
                    child_Q = edge_stats['child_sum_Q'] / edge_stats['child_N']
                    
                    # Create empty move probabilities (will be updated if needed)
                    move_probs = np.zeros(200, dtype=np.float32)
                    
                    edge.child = MCTS.Node(board_copy, child_Q, move_probs)
                    edge.child.N = edge_stats['child_N']
                    edge.child.sum_Q = edge_stats['child_sum_Q']
                else:
                    # Update existing child
                    edge.child.N += edge_stats['child_N']
                    edge.child.sum_Q += edge_stats['child_sum_Q']
    
    def multiprocess_rollouts(self, board, model_path, total_rollouts, num_processes, threads_per_rollout):
        """
        Perform rollouts using multiple processes.
        
        Args:
            board: Chess board
            model_path: Path to the model file
            total_rollouts: Total number of rollout iterations
            num_processes: Number of processes to use
            threads_per_rollout: Number of threads per rollout in each process
        """
        self.model_path = model_path
        
        # Calculate rollouts per process
        rollouts_per_process = total_rollouts // num_processes
        remaining = total_rollouts % num_processes
        
        # Prepare arguments for each process
        process_args = []
        board_fen = board.fen()
        
        for i in range(num_processes):
            process_rollouts = rollouts_per_process
            if i < remaining:
                process_rollouts += 1
            
            if process_rollouts > 0:
                args = (board_fen, model_path, process_rollouts, threads_per_rollout, self.device_str)
                process_args.append(args)
        
        # Use multiprocessing pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker_rollouts, process_args)
        
        # Merge results from all processes
        for stats in results:
            self.merge_stats(stats)
        
        # Update same_paths counter (approximate)
        self.same_paths = sum(1 for edge in self.edges if edge.has_child() and edge.child.N > 1)


def create_multiprocess_root(board, neuralNetwork, model_path):
    """
    Create a MultiprocessRoot instance with the model path set.
    
    Args:
        board: Chess board
        neuralNetwork: Neural network model
        model_path: Path to the model file
    
    Returns:
        MultiprocessRoot instance
    """
    root = MultiprocessRoot(board, neuralNetwork)
    root.model_path = model_path
    return root