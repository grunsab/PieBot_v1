import sys
sys.path.append('..')
import argparse
import chess
import torch
import torch.multiprocessing as mp
import AlphaZeroNetwork
import time
import copy
from device_utils import get_gpu_count, setup_distributed_device
import encoder
import math
import numpy as np
from queue import Queue
from threading import Thread, Lock
import MCTS

def tolist(move_generator):
    """
    Change an iterable object of moves to a list of moves.
    
    Args:
        move_generator (Mainline object) iterable list of moves

    Returns:
        moves (list of chess.Move) list version of the input moves
    """
    moves = []
    for move in move_generator:
        moves.append(move)
    return moves

class MultiGPUMCTS:
    """
    Multi-GPU MCTS implementation that distributes neural network evaluations across multiple GPUs.
    """
    
    def __init__(self, board, models, devices, batch_size=32):
        """
        Initialize Multi-GPU MCTS.
        
        Args:
            board: Initial chess board state
            models: List of neural network models (one per GPU)
            devices: List of devices (one per GPU)
            batch_size: Batch size for neural network evaluation per GPU
        """
        self.models = models
        self.devices = devices
        self.num_gpus = len(devices)
        self.batch_size = batch_size
        
        # Create the root node
        with torch.no_grad():
            value, move_probabilities = encoder.callNeuralNetwork(board, models[0])
        Q = value / 2. + 0.5
        self.root = MCTS.Node(board, Q, move_probabilities)
        self.root.same_paths = 0
        
        # Queues for batch processing
        self.eval_queues = [Queue() for _ in range(self.num_gpus)]
        self.result_queues = {}
        self.queue_lock = Lock()
        
        # Start GPU worker threads
        self.gpu_threads = []
        for gpu_id in range(self.num_gpus):
            thread = Thread(target=self._gpu_worker, args=(gpu_id,))
            thread.daemon = True
            thread.start()
            self.gpu_threads.append(thread)
    
    def _gpu_worker(self, gpu_id):
        """
        GPU worker thread that processes batches of positions.
        
        Args:
            gpu_id: GPU index
        """
        model = self.models[gpu_id]
        device = self.devices[gpu_id]
        
        while True:
            batch = []
            batch_ids = []
            
            # Collect batch
            while len(batch) < self.batch_size:
                try:
                    item = self.eval_queues[gpu_id].get(timeout=0.001)
                    if item is None:  # Shutdown signal
                        return
                    batch_id, board = item
                    batch.append(board)
                    batch_ids.append(batch_id)
                except:
                    if batch:  # Process partial batch if we have something
                        break
                    continue
            
            if not batch:
                continue
            
            # Batch evaluation
            with torch.no_grad():
                # Encode all positions
                encoded_batch = []
                mask_batch = []
                for board in batch:
                    position, mask = encoder.encodePositionForInference(board)
                    encoded = torch.from_numpy(position).unsqueeze(0)
                    encoded_batch.append(encoded)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                    mask_batch.append(mask_tensor)
                
                # Stack and move to GPU
                positions = torch.cat(encoded_batch, dim=0).to(device)
                masks = torch.cat(mask_batch, dim=0).to(device)
                
                # Neural network forward pass
                values, policies = model(positions, policyMask=masks)
                
                # Process results
                values = values.cpu().numpy()
                policies = policies.cpu().numpy()
                
                # Send results back
                for i, batch_id in enumerate(batch_ids):
                    value = float(values[i, 0])  # Extract scalar from (batch_size, 1) array
                    policy = policies[i]
                    move_probs = encoder.decodePolicyOutput(batch[i], policy)
                    
                    with self.queue_lock:
                        if batch_id in self.result_queues:
                            self.result_queues[batch_id].put((value, move_probs))
    
    def parallel_rollouts(self, board, num_rollouts):
        """
        Perform parallel rollouts using multiple GPUs.
        
        Args:
            board: Current board state
            num_rollouts: Number of rollouts to perform
        """
        print(f"Starting {num_rollouts} MCTS rollouts...")
        rollout_threads = []
        self.completed_rollouts = 0
        self.rollout_lock = Lock()
        
        # Distribute rollouts across threads
        rollouts_per_thread = max(1, num_rollouts // (self.num_gpus * 4))
        
        for i in range(0, num_rollouts, rollouts_per_thread):
            batch_size = min(rollouts_per_thread, num_rollouts - i)
            thread = Thread(target=self._rollout_batch, args=(board, batch_size, num_rollouts))
            thread.start()
            rollout_threads.append(thread)
        
        # Wait for all rollouts to complete
        for thread in rollout_threads:
            thread.join()
        
        print(f"Completed {num_rollouts} rollouts")
    
    def _rollout_batch(self, board, batch_size, total_rollouts):
        """
        Perform a batch of rollouts.
        
        Args:
            board: Initial board state
            batch_size: Number of rollouts in this batch
            total_rollouts: Total number of rollouts for progress reporting
        """
        for i in range(batch_size):
            self._single_rollout(board.copy())
            with self.rollout_lock:
                self.completed_rollouts += 1
                if self.completed_rollouts % 1000 == 0:
                    print(f"Progress: {self.completed_rollouts}/{total_rollouts} rollouts completed ({100*self.completed_rollouts/total_rollouts:.1f}%)")
    
    def _single_rollout(self, board):
        """
        Perform a single MCTS rollout.
        
        Args:
            board: Chess board
        """
        node_path = []
        edge_path = []
        current_node = self.root
        
        # Selection phase
        while True:
            node_path.append(current_node)
            
            if current_node.isTerminal():
                edge_path.append(None)
                break
            
            edge = current_node.UCTSelect()
            edge_path.append(edge)
            
            if edge is None:
                break
            
            edge.addVirtualLoss()
            board.push(edge.getMove())
            
            if not edge.has_child():
                break
            
            current_node = edge.getChild()
        
        # Evaluation phase
        edge = edge_path[-1]
        
        if edge is not None:
            # Get neural network evaluation
            batch_id = id(board)
            result_queue = Queue()
            
            with self.queue_lock:
                self.result_queues[batch_id] = result_queue
            
            # Send to least loaded GPU
            min_queue_size = float('inf')
            target_gpu = 0
            for i in range(self.num_gpus):
                size = self.eval_queues[i].qsize()
                if size < min_queue_size:
                    min_queue_size = size
                    target_gpu = i
            
            self.eval_queues[target_gpu].put((batch_id, board))
            
            # Wait for result
            value, move_probabilities = result_queue.get()
            
            with self.queue_lock:
                del self.result_queues[batch_id]
            
            new_Q = value / 2. + 0.5
            edge.expand(board, new_Q, move_probabilities)
            new_Q = 1. - new_Q
        else:
            # Terminal node
            winner = encoder.parseResult(board.result())
            if not board.turn:
                winner *= -1
            new_Q = float(winner) / 2. + 0.5
        
        # Backpropagation phase
        last_node_idx = len(node_path) - 1
        
        for i in range(last_node_idx, -1, -1):
            node = node_path[i]
            is_from_child = (last_node_idx - i) % 2 == 1
            node.updateStats(new_Q, is_from_child)
        
        for edge in edge_path:
            if edge is not None:
                edge.clearVirtualLoss()
    
    def get_best_move(self):
        """Get the best move based on visit counts."""
        return self.root.maxNSelect()
    
    def get_statistics_string(self):
        """Get statistics string for the root node."""
        return self.root.getStatisticsString()
    
    def get_metrics(self):
        """Get MCTS metrics."""
        return {
            'Q': self.root.getQ(),
            'N': self.root.getN(),
            'same_paths': self.root.same_paths
        }
    
    def shutdown(self):
        """Shutdown GPU worker threads."""
        for queue in self.eval_queues:
            queue.put(None)
        for thread in self.gpu_threads:
            thread.join()

def load_models_multi_gpu(model_file, num_gpus=None):
    """
    Load the model on multiple GPUs.
    
    Args:
        model_file: Path to model file
        num_gpus: Number of GPUs to use (None for all available)
    
    Returns:
        models: List of models (one per GPU)
        devices: List of devices
    """
    available_gpus = get_gpu_count()
    
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus == 0:
        print("No GPUs available, using CPU")
        device = torch.device('cpu')
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(model_file, map_location=device)
        model.load_state_dict(weights)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return [model], [device]
    
    models = []
    devices = []
    
    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        devices.append(device)
        
        # Load model
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(model_file, map_location=device)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
        
        print(f'Loaded model on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    
    return models, devices

def main(model_file, mode, color, num_rollouts, num_gpus, batch_size, fen, verbose):
    """
    Main chess playing loop with multi-GPU support.
    """
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Load models on multiple GPUs
    models, devices = load_models_multi_gpu(model_file, num_gpus)
    print(f'Using {len(devices)} device(s) for MCTS')
    
    # Create chess board
    if fen:
        board = chess.Board(fen)
    else:
        board = chess.Board()
    
    # Play chess moves
    while True:
        if board.is_game_over():
            print('Game over. Winner: {}'.format(board.result()))
            board.reset_board()
            c = input('Enter any key to continue ')
        
        # Print current state
        if board.turn:
            print('White\'s turn')
        else:
            print('Black\'s turn')
        print(board)
        
        if mode == 'h' and board.turn == color:
            # Human move
            move_list = tolist(board.legal_moves)
            idx = -1
            
            while not (0 <= idx and idx < len(move_list)):
                string = input('Choose a move ')
                for i, move in enumerate(move_list):
                    if str(move) == string:
                        idx = i
                        break
            
            board.push(move_list[idx])
        else:
            # AI move with multi-GPU MCTS
            starttime = time.perf_counter()
            
            # Create multi-GPU MCTS instance
            mcts = MultiGPUMCTS(board, models, devices, batch_size=batch_size)
            
            # Perform rollouts
            mcts.parallel_rollouts(board, num_rollouts)
            
            endtime = time.perf_counter()
            elapsed = endtime - starttime
            
            # Get best move
            edge = mcts.get_best_move()
            bestmove = edge.getMove()
            
            # Get metrics
            metrics = mcts.get_metrics()
            nps = metrics['N'] / elapsed
            
            if verbose:
                print(mcts.get_statistics_string())
                print('total rollouts {} Q {:0.3f} elapsed {:0.2f} nps {:0.2f}'.format(
                    int(metrics['N']), metrics['Q'], elapsed, nps))
            
            print('best move {}'.format(str(bestmove)))
            board.push(bestmove)
            
            # Shutdown MCTS workers
            mcts.shutdown()
        
        if mode == 'p':
            # Profile mode - exit after first move
            break

def parse_color(color_string):
    """Maps 'w' to True and 'b' to False."""
    if color_string == 'w' or color_string == 'W':
        return True
    elif color_string == 'b' or color_string == 'B':
        return False
    else:
        print('Unrecognized argument for color')
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Play chess with multi-GPU support.')
    parser.add_argument('--model', help='Path to model (.pt) file.', required=True)
    parser.add_argument('--mode', help='Operation mode: \'s\' self play, \'p\' profile, \'h\' human')
    parser.add_argument('--color', help='Your color w or b')
    parser.add_argument('--rollouts', type=int, help='The number of rollouts on computers turn')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use (default: all available)')
    parser.add_argument('--batch-size', type=int, help='Batch size per GPU for neural network evaluation')
    parser.add_argument('--verbose', help='Print search statistics', action='store_true')
    parser.add_argument('--fen', help='Starting fen')
    parser.set_defaults(
        verbose=False, 
        mode='p', 
        color='w', 
        rollouts=1000, 
        gpus=None,
        batch_size=32
    )
    args = parser.parse_args()
    
    main(
        args.model, 
        args.mode, 
        parse_color(args.color), 
        args.rollouts, 
        args.gpus,
        args.batch_size,
        args.fen, 
        args.verbose
    )