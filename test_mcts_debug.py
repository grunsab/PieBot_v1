#!/usr/bin/env python3
"""
Test script to debug MCTS chess engine blunders and timeouts.
This script simulates a chess game and monitors for issues.
"""

import chess
import chess.pgn
import sys
import os
import time
import torch
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AlphaZeroNetwork
import MCTS_root_parallel as MCTS
from device_utils import get_optimal_device, optimize_for_device
import encoder

def load_model(model_path="weights/AlphaZeroNet_20x256.pt"):
    """Load the chess model."""
    device, device_str = get_optimal_device()
    print(f"Loading model on {device_str}...")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(model_path, map_location=device)
    
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, device

def evaluate_position(board, model):
    """Evaluate a position using the neural network."""
    value, move_probs = encoder.callNeuralNetwork(board, model)
    return value

def run_mcts_search(board, model, rollouts=100, verbose=True):
    """Run MCTS search on a position."""
    start_time = time.time()
    
    # Create a fresh MCTS engine for each search
    mcts_engine = MCTS.Root(board.copy(), model)
    
    try:
        # Run the search
        mcts_engine.parallelRolloutsTotal(board.copy(), model, rollouts, 64)
        
        # Get best move
        edge = mcts_engine.maxNSelect()
        if edge:
            best_move = edge.getMove()
            visits = edge.getN()
            q_value = edge.getQ()
        else:
            best_move = None
            visits = 0
            q_value = 0.5
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Search took {elapsed:.2f}s for {rollouts} rollouts")
            if best_move:
                print(f"  Best move: {best_move} (visits: {visits}, Q: {q_value:.3f})")
        
        return best_move, elapsed, q_value
        
    except Exception as e:
        print(f"ERROR during MCTS search: {e}")
        traceback.print_exc()
        return None, 0, 0
    # Note: Don't cleanup here - let the main function handle cleanup at the end

def detect_blunder(board, move, model, threshold=0.3):
    """
    Detect if a move is a blunder by comparing position evaluations.
    """
    # Evaluate position before move
    eval_before = evaluate_position(board, model)
    
    # Make the move
    board_after = board.copy()
    board_after.push(move)
    
    # Evaluate position after move (from opponent's perspective)
    eval_after = evaluate_position(board_after, model)
    
    # Calculate evaluation drop (accounting for perspective change)
    eval_drop = eval_before - (-eval_after)
    
    is_blunder = eval_drop > threshold
    
    return is_blunder, eval_drop, eval_before, eval_after

def simulate_game(model, max_moves=100, rollouts_per_move=100):
    """
    Simulate a game and monitor for blunders and timeouts.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    
    blunder_count = 0
    timeout_count = 0
    move_count = 0
    
    print("\n" + "="*60)
    print("Starting chess game simulation...")
    print("="*60)
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        turn = "White" if board.turn else "Black"
        
        print(f"\nMove {move_count} ({turn}):")
        print(f"FEN: {board.fen()}")
        
        # Check if game is over due to repetition
        if board.can_claim_threefold_repetition():
            print("  Game ending: Draw by threefold repetition")
            break
        if board.is_game_over():
            print(f"  Game is over! Result: {board.result()}")
            break
        
        # Search for best move
        best_move, search_time, q_value = run_mcts_search(
            board, model, rollouts=rollouts_per_move, verbose=True
        )
        
        # Check for timeout
        if search_time > 10.0:  # More than 10 seconds is concerning
            timeout_count += 1
            print(f"  WARNING: Search took {search_time:.1f}s (potential timeout issue)")
        
        if best_move is None:
            # No move found - this can happen in drawn positions
            if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                print("  No move selected (drawn position detected)")
                break
            else:
                print("  WARNING: No move found in non-drawn position")
                # Try to recover with a random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    best_move = legal_moves[0]
                    print(f"  Using fallback move: {best_move}")
                else:
                    break
        
        # Check if move is a blunder
        is_blunder, eval_drop, eval_before, eval_after = detect_blunder(
            board, best_move, model
        )
        
        if is_blunder:
            blunder_count += 1
            print(f"  BLUNDER DETECTED! Eval drop: {eval_drop:.3f}")
            print(f"     Before: {eval_before:.3f}, After: {-eval_after:.3f}")
        
        # Make the move
        board.push(best_move)
        node = node.add_variation(best_move)
        
        # Print board position every 10 moves
        if move_count % 10 == 0:
            print("\nCurrent position:")
            print(board)
    
    # Game summary
    print("\n" + "="*60)
    print("GAME SUMMARY")
    print("="*60)
    print(f"Total moves: {move_count}")
    print(f"Blunders detected: {blunder_count}")
    print(f"Timeout warnings: {timeout_count}")
    print(f"Final position: {board.fen()}")
    
    if board.is_game_over():
        print(f"Game result: {board.result()}")
    
    # Save PGN
    pgn_file = "debug_game.pgn"
    with open(pgn_file, "w") as f:
        print(game, file=f)
    print(f"\nGame saved to {pgn_file}")
    
    return blunder_count, timeout_count

def main():
    """Main test function."""
    # Load model
    model, device = load_model()
    print(f"Model loaded successfully on {device}")
    
    # Run test game
    print("\nRunning test game to detect blunders and timeouts...")
    
    try:
        blunder_count, timeout_count = simulate_game(
            model, 
            max_moves=300,  # Play 100 moves (50 per side) for faster testing
            rollouts_per_move=500  # Use 5000 rollouts for good quality
        )
        
        # Analyze results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        if blunder_count > 2:
            print(f"ISSUE DETECTED: {blunder_count} blunders found!")
            print("   This suggests the MCTS tree may be corrupted.")
        else:
            print(f"Blunder count normal: {blunder_count}")
        
        if timeout_count > 0:
            print(f"ISSUE DETECTED: {timeout_count} timeout warnings!")
            print("   This suggests blocking or deadlock issues.")
        else:
            print("No timeout issues detected")
        
        if blunder_count <= 2 and timeout_count == 0:
            print("\nAll tests passed! The engine appears to be working correctly.")
        else:
            print("\nIssues detected. Review the debug output above.")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        traceback.print_exc()
    finally:
        # Ensure cleanup
        MCTS.Root.cleanup_engine()
        print("\nCleanup completed.")

if __name__ == "__main__":
    main()