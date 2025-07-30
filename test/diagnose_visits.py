#!/usr/bin/env python3
"""Diagnose why visit counts are lower than requested"""

import sys
sys.path.append('..')
import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from experiments.MCTS.MCTS_ultra_performance import UltraPerformanceMCTSEngine

def diagnose_visits():
    board = chess.Board()
    model = AlphaZeroNet(10, 128)
    model.eval()
    
    for requested in [100, 500, 1000]:
        engine = UltraPerformanceMCTSEngine(
            model,
            device=torch.device('cpu'),
            batch_size=64,
            num_workers=8,
            verbose=False
        )
        
        engine.start()
        
        # Run search
        import time
        start = time.time()
        best_move = engine.search(board, requested)
        elapsed = time.time() - start
        
        # Check actual visits
        root_hash = board.fen()
        if root_hash in engine.node_lookup:
            root_idx = engine.node_lookup[root_hash]
            root = engine.nodes[root_idx]
            
            print(f"\nRequested: {requested} simulations")
            print(f"Root visits: {root.visits}")
            print(f"Total simulations counter: {engine.total_simulations}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Efficiency: {root.visits / requested * 100:.1f}%")
            
            # Check children
            if root_idx in engine.children:
                children = engine.children[root_idx]
                total_child_visits = sum(
                    engine.nodes[c].visits for c in children 
                    if c < len(engine.nodes)
                )
                print(f"Total child visits: {total_child_visits}")
                print(f"Visit accounting error: {root.visits - total_child_visits - 1}")
        
        engine.stop()

if __name__ == "__main__":
    diagnose_visits()