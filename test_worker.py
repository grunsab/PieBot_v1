#!/usr/bin/env python3
"""
Test MCTSWorker in isolation.
"""

import chess
import torch
import threading
import multiprocessing as mp
import uuid
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import Config, NNManager, MCTSWorker, SharedTree, SharedTreeClient
import MCTS_profiling_speedups_v2 as MCTS

print("Setup...")
Config.SIMULATIONS_PER_MOVE = 2  # Small number for testing

board = chess.Board()
model = AlphaZeroNet(num_blocks=10, num_filters=128)
model.eval()
device = torch.device('cpu')

# Create shared tree
tree_name = f"mcts_tree_{uuid.uuid4().hex[:8]}"
shared_tree = SharedTree(tree_name)
shared_tree.alloc_lock = threading.Lock()

# Create root
with torch.no_grad():
    value, move_probabilities = MCTS.callNeuralNetworkOptimized(board, model)
root_idx = shared_tree.create_root(board, value, move_probabilities)
print(f"Root created at index {root_idx}")

# Start NN Manager
nn_manager = NNManager(model, device)
nn_manager.start()
print("NNManager started")

# Create shared tree client
client = SharedTreeClient(tree_name)
client.alloc_lock = shared_tree.alloc_lock

# Create simulation counter
simulations_done = mp.Value('i', 0)

print(f"\nCreating worker thread...")
worker = MCTSWorker(0, root_idx, board, nn_manager, client, simulations_done)

print("Starting worker...")
worker.start()

print("Waiting for worker to complete...")
worker.join(timeout=5.0)

if worker.is_alive():
    print("WARNING: Worker still alive after 5 seconds!")
    worker.stop()
else:
    print(f"Worker completed! Simulations done: {simulations_done.value}")

# Check tree state
root_node = client.get_node(root_idx)
print(f"Root N after search: {root_node.get_N()}")
print(f"Root Q after search: {root_node.get_Q()}")

# Cleanup
nn_manager.stop()
shared_tree.cleanup()
client.cleanup()
print("\nTest completed!")