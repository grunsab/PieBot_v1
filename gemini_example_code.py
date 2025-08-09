import threading
import time
import queue
import numpy as np
import math
import random
from collections import defaultdict

# --- Configuration ---
class Config:
    """Hyperparameters for the MCTS search."""
    NUM_WORKERS = 8  # Number of parallel search threads
    BATCH_SIZE = 16 # Max batch size for the neural network
    INFERENCE_TIMEOUT = 0.01  # Seconds to wait before running a smaller batch
    CPUCT = 1.41  # Exploration constant in PUCT formula
    VIRTUAL_LOSS = 3  # Value to add to a node's visit count during selection
    SIMULATIONS_PER_MOVE = 800 # Total simulations to run before making a move
    GAME_MOVE_LIMIT = 20 # To prevent infinite games in this example

# --- Mock Game Logic ---
class MockGameState:
    """
    A very simple game state to demonstrate the MCTS logic.
    The "game" is to reach the number 10. A move consists of adding 1 or 2.
    The state is represented by the current number.
    """
    def __init__(self, current_number=0, history=None):
        self.current_number = current_number
        self.history = history or [0]

    def get_legal_moves(self):
        """Legal moves are adding 1 or 2, unless it goes over 10."""
        moves = []
        if self.current_number + 1 <= 10:
            moves.append(1)
        if self.current_number + 2 <= 10:
            moves.append(2)
        return moves

    def is_game_over(self):
        """Game is over if we reach 10 or there are no legal moves."""
        return self.current_number == 10 or not self.get_legal_moves()

    def get_game_result(self):
        """The 'winner' is 1 if we land on 10, -1 otherwise (loss/draw)."""
        if self.current_number == 10:
            return 1.0  # Win
        return -1.0 # Loss

    def apply_move(self, move):
        """Return a new game state after applying a move."""
        new_number = self.current_number + move
        new_history = self.history + [new_number]
        return MockGameState(new_number, new_history)

    def __str__(self):
        """Use the history as a unique representation of the state."""
        return str(self.history)

# --- Mock Neural Network ---
class MockNNManager(threading.Thread):
    """
    Simulates a neural network manager that batches inference requests.
    It runs in its own thread, collecting requests and processing them in batches.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.inference_queue = queue.Queue()
        self.results_dict = {}
        self._stop_event = threading.Event()

    def run(self):
        """Main loop for the inference thread."""
        while not self._stop_event.is_set():
            try:
                # Wait for a batch of requests, with a timeout
                batch = self._get_batch()
                if batch:
                    self._process_batch(batch)
            except queue.Empty:
                continue

    def _get_batch(self):
        """Collects a batch of states from the queue."""
        batch = []
        while len(batch) < Config.BATCH_SIZE:
            try:
                # The timeout allows processing smaller batches if the queue is slow
                item = self.inference_queue.get(timeout=Config.INFERENCE_TIMEOUT)
                batch.append(item)
            except queue.Empty:
                # If timeout is reached and we have items, process them
                if batch:
                    break
        return batch

    def _process_batch(self, batch):
        """
        Simulates running batch inference.
        In a real implementation, this would call a PyTorch/TensorFlow model.
        """
        # print(f"[NN Manager] Processing batch of size {len(batch)}")
        for game_state in batch:
            # Generate random policy and value for demonstration
            legal_moves = game_state.get_legal_moves()
            if not legal_moves:
                policy = {}
                value = game_state.get_game_result()
            else:
                # Random policy, normalized
                move_priors = np.random.rand(len(legal_moves))
                move_priors /= np.sum(move_priors)
                policy = {move: prior for move, prior in zip(legal_moves, move_priors)}
                # Random value between -1 and 1
                value = np.random.uniform(-1, 1)

            # Store the result so worker threads can retrieve it
            self.results_dict[str(game_state)] = (policy, value)

    def stop(self):
        """Signals the thread to stop."""
        self._stop_event.set()

# --- MCTS Node and Search Logic ---
class Node:
    """A node in the Monte Carlo Tree."""
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # map from move to Node

        # --- Node Statistics ---
        self.visit_count = 0
        self.total_action_value = 0.0  # W(s,a)
        self.mean_action_value = 0.0   # Q(s,a)
        self.prior_probability = prior_p # P(s,a)

        # --- Threading ---
        self.lock = threading.Lock()

    def select_child(self):
        """
        Select the child with the highest PUCT score.
        PUCT(s,a) = Q(s,a) + U(s,a)
        U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None
        best_move = None

        parent_visits = self.visit_count
        for move, child in self.children.items():
            score = child.mean_action_value + \
                    Config.CPUCT * child.prior_probability * \
                    math.sqrt(parent_visits) / (1 + child.visit_count)

            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
        return best_move, best_child

    def expand(self, policy):
        """Expand the node by creating children for all legal moves."""
        for move, prior in policy.items():
            if move not in self.children:
                self.children[move] = Node(parent=self, prior_p=prior)

    def backpropagate(self, value):
        """Update node statistics back up to the root."""
        self.visit_count += 1
        self.total_action_value += value
        self.mean_action_value = self.total_action_value / self.visit_count


class MCTSWorker(threading.Thread):
    """A worker thread that runs MCTS simulations."""
    def __init__(self, worker_id, root_node, initial_state, nn_manager, simulations_done_counter):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.root = root_node
        self.current_state = initial_state
        self.nn_manager = nn_manager
        self.simulations_done = simulations_done_counter
        self._stop_event = threading.Event()

    def run(self):
        """Main simulation loop for the worker."""
        while not self._stop_event.is_set():
            # Check if we have completed enough simulations
            with self.simulations_done.get_lock():
                if self.simulations_done.value >= Config.SIMULATIONS_PER_MOVE:
                    break

            self.run_one_simulation()

            # Increment the shared counter
            with self.simulations_done.get_lock():
                self.simulations_done.value += 1

    def run_one_simulation(self):
        """Performs one full MCTS simulation from selection to backpropagation."""
        path = []
        node = self.root
        game_state = self.current_state

        # 1. SELECTION: Traverse the tree until a leaf node is found.
        while True:
            node.lock.acquire()
            if not node.children: # It's a leaf node
                path.append(node)
                # Apply virtual loss to the leaf before releasing the lock
                node.visit_count += Config.VIRTUAL_LOSS
                node.lock.release()
                break

            # It's not a leaf, so apply virtual loss and select the best child
            path.append(node)
            node.visit_count += Config.VIRTUAL_LOSS
            
            # Select best child based on PUCT
            move, next_node = node.select_child()
            
            node.lock.release() # Release lock on parent before moving to child

            game_state = game_state.apply_move(move)
            node = next_node

        leaf_node = node
        
        # Check if the game ended during selection
        if game_state.is_game_over():
            value = game_state.get_game_result()
        else:
            # 2. EXPANSION & EVALUATION: Request NN evaluation for the leaf node.
            # This is a blocking call. The worker waits for the NN result.
            self.nn_manager.inference_queue.put(game_state)
            
            # Wait for the result to appear in the dictionary
            policy, value = self.wait_for_nn_result(str(game_state))
            
            # Expand the leaf node with the policy from the NN
            leaf_node.lock.acquire()
            leaf_node.expand(policy)
            leaf_node.lock.release()

        # 3. BACKPROPAGATION: Update statistics back up the tree.
        for node in reversed(path):
            node.lock.acquire()
            # Undo the virtual loss and apply the real result
            node.visit_count -= Config.VIRTUAL_LOSS
            node.backpropagate(value)
            node.lock.release()

    def wait_for_nn_result(self, state_key):
        """Spins until the NN result for the given state is available."""
        while state_key not in self.nn_manager.results_dict:
            time.sleep(1e-5) # Sleep for a tiny duration to yield CPU
        result = self.nn_manager.results_dict.pop(state_key)
        return result

    def stop(self):
        """Signals the worker thread to stop."""
        self._stop_event.set()


def main():
    """Main function to run the MCTS search."""
    game_state = MockGameState()
    move_count = 0

    while not game_state.is_game_over() and move_count < Config.GAME_MOVE_LIMIT:
        print(f"\n--- Move {move_count+1}: Current State = {game_state.current_number} ---")
        
        # Start the neural network manager
        nn_manager = MockNNManager()
        nn_manager.start()

        # Create the root node for the current search
        root_node = Node(parent=None, prior_p=1.0)

        # Shared counter for simulations
        simulations_done = threading.Value('i', 0)

        # Create and start worker threads
        workers = []
        for i in range(Config.NUM_WORKERS):
            worker = MCTSWorker(i, root_node, game_state, nn_manager, simulations_done)
            workers.append(worker)
            worker.start()

        # Wait for all workers to finish their simulations
        for worker in workers:
            worker.join()

        # Stop the NN manager thread for this move
        nn_manager.stop()
        nn_manager.join()

        # Choose the best move based on the most visited child node
        if not root_node.children:
            print("No moves found. Game over.")
            break
            
        best_move = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
        
        print(f"Search complete. Total simulations: {simulations_done.value}")
        print("Root node stats:")
        for move, child in sorted(root_node.children.items()):
            print(f"  Move: {move}, Visits: {child.visit_count}, Value: {child.mean_action_value:.3f}")
        
        print(f"\n>>> Best move chosen: {best_move}\n")

        # Apply the move to the game state
        game_state = game_state.apply_move(best_move)
        move_count += 1
    
    print("--- Game Over ---")
    print(f"Final State: {game_state.current_number}")
    print(f"Final Result: {game_state.get_game_result()}")


if __name__ == "__main__":
    main()
