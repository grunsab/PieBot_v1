import threading
import queue
import chess
import numpy as np
import torch
import encoder

# This class is now a standard Python thread.
class NNManager(threading.Thread):
    def __init__(self, model, device):
        super().__init__(daemon=True)
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        # This queue is passed to the C++ engine
        self.inference_queue = queue.Queue()
        self._stop_event = threading.Event()

    def run(self):
        """The main loop for this thread."""
        while not self._stop_event.is_set():
            try:
                # Block until the first request arrives, with a short timeout
                batch_requests = [self.inference_queue.get(timeout=0.01)]
                
                # If more requests are waiting, grab them to build a larger batch
                while len(batch_requests) < 512 and not self.inference_queue.empty():
                    batch_requests.append(self.inference_queue.get_nowait())
                
                if batch_requests:
                    self._process_batch(batch_requests)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in NNManager: {e}")
                break

    def _process_batch(self, batch):
        requests_with_boards = []
        for req in batch:
            if req is None: continue # Sentinel value to unblock
            try:
                board = req.board
                requests_with_boards.append((req, board))
            except Exception as e:
                if hasattr(req, 'set_exception'):
                    req.set_exception(e)
                else:
                    print(f"Error processing request: {e}")

        if not requests_with_boards:
            return

        encoded_states = []
        masks = []
        for _, board in requests_with_boards:
            state, mask = encoder.encodePositionForInference(board)
            encoded_states.append(state)
            masks.append(mask)

        inputs = torch.from_numpy(np.stack(encoded_states)).to(self.device)
        policy_masks = torch.from_numpy(np.stack(masks)).view(len(batch), -1).to(self.device)

        with torch.no_grad():
            raw_values, raw_policies = self.model(inputs, policyMask=policy_masks)

        values = raw_values.cpu().numpy().flatten()
        policies = raw_policies.cpu().numpy()

        for i, (req, board) in enumerate(requests_with_boards):
            try:
                # decodePolicyOutput returns a numpy array of move probabilities
                move_probs = encoder.decodePolicyOutput(board, policies[i])
                
                # Convert to dictionary mapping moves to probabilities
                policy_dict = {}
                for idx, move in enumerate(board.legal_moves):
                    if idx < len(move_probs):
                        policy_dict[move] = float(move_probs[idx])
                
                value = float(values[i])
                if hasattr(req, 'set_result'):
                    req.set_result(policy_dict, value)
                else:
                    print(f"Warning: Request doesn't have set_result method")
            except Exception as e:
                if hasattr(req, 'set_exception'):
                    req.set_exception(e)
                else:
                    print(f"Error processing request: {e}")

    def stop(self):
        self._stop_event.set()
        # Put a sentinel value to unblock the queue.get() call
        self.inference_queue.put(None)
