import chess
import threading
import torch

# --- Import the NEWLY COMPILED C++ engine ---
from mcts_cpp_engine import MCTS_CPP
# --- Import NNManager from its own dedicated file ---
from nn_manager import NNManager

class MCTS:
    """
    A Python wrapper for the high-performance C++ MCTS engine.
    This class now manages the NNManager thread.
    """
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Create the NNManager here in Python
        self.nn_manager = NNManager(self.model, self.device)
        
        # 2. Pass its queue to the C++ engine's constructor (using positional arguments)
        self.engine = MCTS_CPP(
            self.nn_manager.inference_queue,  # positional argument
            8,                                 # num_workers
            512,                              # batch_size
            1.5,                              # cpuct
            3,                                # virtual_loss
            0.3,                              # dirichlet_alpha
            0.25                              # dirichlet_epsilon
        )

    def search(self, board, num_simulations):
        """
        Starts the NNManager, delegates the search to the C++ backend,
        and then stops the NNManager.
        """
        
        # 3. Start the NNManager thread from Python before searching
        # Only start if it's not already running
        if not self.nn_manager.is_alive():
            self.nn_manager.start()
        
        # This call will block until the C++ search is complete
        best_move = self.engine.search(board, num_simulations)
        
        # Don't stop the thread - keep it running for next search
        # The thread will be cleaned up when the MCTS object is destroyed
        
        return best_move

