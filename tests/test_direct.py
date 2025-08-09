#!/usr/bin/env python3
"""
Direct test without Root wrapper.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from MCTS_multiprocess import MultiprocessMCTS

print("1. Creating board and model...")
board = chess.Board()
model = AlphaZeroNet(num_blocks=10, num_filters=128)
model.eval()

print("2. Creating MultiprocessMCTS...")
mcts = MultiprocessMCTS(model, num_workers=1)

print("3. Running parallel rollouts...")
try:
    root_idx = mcts.run_parallel_rollouts(board, 2)
    print(f"4. Rollouts completed! Root index: {root_idx}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("5. Cleaning up...")
mcts.cleanup()
print("Done!")