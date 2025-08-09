#!/usr/bin/env python3
"""
Test script for v2 shared tree MCTS implementation.
"""

import chess
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MCTS_shared_search_v2 as MCTS

def test_search(board, rollouts=10):
    """Test a single search operation."""
    print(f"\nTesting search with {rollouts} rollouts...")
    print(f"Position: {board.fen()[:50]}...")
    
    mcts_engine = MCTS.Root(board, None)
    
    start_time = time.time()
    mcts_engine.parallelRolloutsTotal(board, None, rollouts, 4)
    edge = mcts_engine.maxNSelect()
    
    elapsed = time.time() - start_time
    
    if edge:
        best_move = edge.getMove()
        print(f"Best move: {best_move}")
        print(f"Search took {elapsed:.2f} seconds")
        return best_move
    else:
        print("No move found")
        return None

def main():
    """Main test function."""
    print("Testing v2 shared tree MCTS...")
    
    # Test 1: Initial position
    board = chess.Board()
    print("\n" + "="*50)
    print("Test 1: Initial position")
    move = test_search(board, rollouts=20)
    
    if move:
        # Test 2: After first move
        board.push(move)
        print("\n" + "="*50)
        print("Test 2: After first move")
        move = test_search(board, rollouts=20)
    
    if move:
        # Test 3: After second move
        board.push(move)
        print("\n" + "="*50)
        print("Test 3: After second move")
        move = test_search(board, rollouts=30)
    
    # Cleanup
    MCTS.Root.cleanup_engine()
    print("\n" + "="*50)
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()