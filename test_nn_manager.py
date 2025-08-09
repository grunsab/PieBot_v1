#!/usr/bin/env python3
"""
Test NNManager threading.
"""

import chess
import torch
import time
import queue
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import Config, NNManager

print("Test 1: Create model and NNManager")
model = AlphaZeroNet(num_blocks=10, num_filters=128)
model.eval()
device = torch.device('cpu')

nn_manager = NNManager(model, device)
print("NNManager created")

print("\nTest 2: Start NNManager thread")
nn_manager.start()
print("NNManager started")

print("\nTest 3: Send a request")
board = chess.Board()
request_id = "test_001"
nn_manager.inference_queue.put((request_id, board))
print(f"Request sent: {request_id}")

print("\nTest 4: Wait for result")
start_time = time.time()
timeout = 5.0
while request_id not in nn_manager.results_dict:
    if time.time() - start_time > timeout:
        print("TIMEOUT waiting for result!")
        break
    time.sleep(0.01)

if request_id in nn_manager.results_dict:
    policy, value = nn_manager.results_dict[request_id]
    print(f"Got result! Value: {value:.3f}, Policy keys: {len(policy)}")
else:
    print("No result received")

print("\nTest 5: Stop NNManager")
nn_manager.stop()
nn_manager.join(timeout=2.0)
if nn_manager.is_alive():
    print("WARNING: NNManager thread still alive!")
else:
    print("NNManager stopped successfully")

print("\nTest completed!")