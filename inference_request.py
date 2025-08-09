import threading

class InferenceRequest:
    """
    A Python-compatible inference request that can be passed between C++ and Python.
    """
    def __init__(self, board):
        self.board = board
        self.result = None
        self.event = threading.Event()
        self.exception = None
    
    def set_result(self, policy, value):
        """Called by NNManager to set the result."""
        self.result = {'policy': policy, 'value': value}
        self.event.set()
    
    def set_exception(self, exc):
        """Called by NNManager if there's an error."""
        self.exception = exc
        self.event.set()
    
    def wait_for_result(self, timeout=10.0):
        """Wait for the result to be available."""
        if self.event.wait(timeout):
            if self.exception:
                raise self.exception
            return self.result
        else:
            raise TimeoutError("Inference request timed out")