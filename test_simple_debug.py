#!/usr/bin/env python3
"""
Simple debug to test the basic components.
"""

import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import Config, NNManager, SharedTree, SharedTreeClient
import MCTS_profiling_speedups_v2 as MCTS
import uuid

print("Test 1: Basic setup")
board = chess.Board()
model = AlphaZeroNet(num_blocks=10, num_filters=128)
model.eval()

print("Test 2: Create shared tree")
tree_name = f"mcts_tree_{uuid.uuid4().hex[:8]}"
shared_tree = SharedTree(tree_name)

print("Test 3: Evaluate root position")
with torch.no_grad():
    value, move_probabilities = MCTS.callNeuralNetworkOptimized(board, model)
print(f"Root value: {value:.3f}")
print(f"Move probabilities shape: {move_probabilities}")

print("Test 4: Create root node")
import threading
shared_tree.alloc_lock = threading.Lock()
root_idx = shared_tree.create_root(board, value, move_probabilities)
print(f"Root index: {root_idx}")

print("Test 5: Get root node via client")
client = SharedTreeClient(tree_name)
client.alloc_lock = shared_tree.alloc_lock
root_node = client.get_node(root_idx)
print(f"Root N: {root_node.get_N()}")
print(f"Root Q: {root_node.get_Q()}")

print("Test 6: Check edges")
edges = root_node.get_edges(client.edges_shm)
print(f"Number of edges: {len(edges)}")
if edges:
    print(f"First edge move: {edges[0].get_move()}")
    print(f"First edge P: {edges[0].get_P()}")

print("\nAll tests passed!")
shared_tree.cleanup()
client.cleanup()