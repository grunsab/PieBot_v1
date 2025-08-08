import threading
from collections import namedtuple

# --- Configuration ---
class Config:
    """Hyperparameters for the MCTS search."""
    NUM_WORKERS = 8
    BATCH_SIZE = 512
    INFERENCE_TIMEOUT = 0.005
    CPUCT = 1.5
    VIRTUAL_LOSS = 3
    SIMULATIONS_PER_MOVE = 10000
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

# --- Data Structures ---
InferenceRequest = namedtuple('InferenceRequest',
    ['request_id', 'encoded_state', 'mask', 'completion_event', 'board_fen'])

class ThreadSafeCounter:
    """A simple counter that is safe to use across multiple threads."""
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    @property
    def value(self):
        with self._lock:
            return self._value
