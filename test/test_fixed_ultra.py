#!/usr/bin/env python3
"""Test the fixed Ultra-Performance MCTS implementation"""

import sys
sys.path.append('..')
import chess
import torch
from AlphaZeroNetwork import AlphaZeroNet
from experiments.MCTS.MCTS_ultra_performance_fixed import UltraPerformanceMCTSEngine
import MCTS

def test_fixed_ultra():
    print("Testing fixed Ultra-Performance MCTS...")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load model
    model = AlphaZeroNet(10, 128)
    model = model.to(device)
    model.eval()
    
    # Test 1: Basic functionality
    print("\n1. Testing basic search functionality...")
    board = chess.Board()
    engine = UltraPerformanceMCTSEngine(
        model,
        device=device,
        batch_size=64,
        num_workers=8,
        verbose=True
    )
    engine.start()
    
    # Run search
    move = engine.search(board, 100)
    print(f"Selected move: {move}")
    
    # Check root visits
    root_hash = board.fen()
    if root_hash in engine.node_lookup:
        root_idx = engine.node_lookup[root_hash]
        root = engine.nodes[root_idx]
        print(f"Root visits: {root.visits}")
        
        # Check children
        if root_idx in engine.children:
            children = engine.children[root_idx]
            visited_children = 0
            zero_visit_children = 0
            for child_idx in children:
                if child_idx < len(engine.nodes):
                    child = engine.nodes[child_idx]
                    if child.visits > 0:
                        visited_children += 1
                    else:
                        zero_visit_children += 1
            print(f"Visited children: {visited_children}/{len(children)}")
            print(f"Zero-visit children: {zero_visit_children}")
    
    # Test 2: Tree reuse
    print("\n2. Testing tree reuse...")
    board.push(move)
    move2 = engine.search(board, 50)
    print(f"Move after reuse: {move2}")
    
    # Check if we reused the tree
    root_hash2 = board.fen()
    if root_hash2 in engine.node_lookup:
        root_idx2 = engine.node_lookup[root_hash2]
        root2 = engine.nodes[root_idx2]
        print(f"Root visits after reuse: {root2.visits}")
        
    # Test 3: Compare with original MCTS
    print("\n3. Comparing move selection with original MCTS...")
    board = chess.Board()
    
    # Original MCTS
    root_orig = MCTS.Root(board, model)
    for _ in range(5):  # 100 rollouts total
        root_orig.parallelRollouts(board.copy(), model, 20)
    
    best_edge = root_orig.maxNSelect()
    if best_edge:
        orig_move = best_edge.getMove()
        print(f"Original MCTS selected: {orig_move}")
    
    # Ultra-Performance
    move_ultra = engine.search(board, 100)
    print(f"Ultra-Performance selected: {move_ultra}")
    
    engine.stop()
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_fixed_ultra()