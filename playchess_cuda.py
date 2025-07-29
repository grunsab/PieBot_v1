#!/usr/bin/env python3
"""
Play chess against the AI - Windows/CUDA Optimized Version

High-performance interactive chess playing interface optimized for Windows
systems with NVIDIA GPUs (RTX 4080 and similar).
"""

import argparse
import chess
import torch
import AlphaZeroNetwork
import time
import os
import sys
from MCTS_cuda import MCTSEngineCUDA
from async_neural_net_server_cuda import NeuralNetworkPoolCUDA

# Windows-specific imports
if sys.platform == 'win32':
    import msvcrt
    import colorama
    colorama.init()  # Enable ANSI colors on Windows


def clear_screen():
    """Clear the console screen (cross-platform)."""
    if sys.platform == 'win32':
        os.system('cls')
    else:
        os.system('clear')


def print_board_fancy(board):
    """Print chess board with Unicode pieces and colors."""
    piece_symbols = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
    }
    
    print("\n    a   b   c   d   e   f   g   h")
    print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    
    for rank in range(7, -1, -1):
        print(f"{rank+1} │", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            # Alternating colors for squares
            if (rank + file) % 2 == 0:
                bg = '\033[47m'  # Light background
            else:
                bg = '\033[100m'  # Dark background
                
            if piece:
                symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                if piece.color:  # White
                    print(f"{bg}\033[97m {symbol} \033[0m│", end="")
                else:  # Black
                    print(f"{bg}\033[30m {symbol} \033[0m│", end="")
            else:
                print(f"{bg}   \033[0m│", end="")
                
        print(f" {rank+1}")
        
        if rank > 0:
            print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
    
    print("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    print("    a   b   c   d   e   f   g   h\n")


def main(modelFile, mode, color, num_rollouts, num_threads, fen, verbose, 
         batch_size=512, device_id=0, use_unicode=True):
    
    # Set process priority on Windows
    if sys.platform == 'win32':
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except:
            pass
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This version requires an NVIDIA GPU.")
        print("Please use the CPU version or install CUDA drivers.")
        return
        
    device_count = torch.cuda.device_count()
    if device_id >= device_count:
        print(f"ERROR: CUDA device {device_id} not available. Found {device_count} devices.")
        device_id = 0
        
    # Load model and create MCTS engine
    device = torch.device(f'cuda:{device_id}')
    device_name = torch.cuda.get_device_name(device_id)
    device_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    
    print(f"Using CUDA device {device_id}: {device_name}")
    print(f"GPU Memory: {device_memory:.1f}GB")
    print(f"Loading model from: {modelFile}")
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(modelFile, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    
    # Create CUDA-optimized MCTS engine
    print(f"\nInitializing CUDA MCTS engine")
    print(f"Workers: {num_threads}, Batch size: {batch_size}")
    
    mcts_engine = MCTSEngineCUDA(
        model,
        device=device,
        max_batch_size=batch_size,
        num_workers=num_threads,
        verbose=verbose
    )
    mcts_engine.start()
   
    # Create chess board
    if fen:
        board = chess.Board(fen)
    else:
        board = chess.Board()

    # Game loop
    print("\n" + "="*60)
    print("   AlphaZero Chess Engine - CUDA High Performance Edition")
    print("="*60)
    print("\nCommands:")
    print("  - Enter move in UCI format (e.g., e2e4, g1f3)")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'undo' to take back last move")
    print("  - 'new' to start a new game")
    print("  - 'fen' to show current FEN")
    print("  - 'eval' to see position evaluation")
    print("-" * 60)
    
    move_history = []
    
    while True:
        # Check game over
        if board.is_game_over():
            print(f"\nGame over! Result: {board.result()}")
            
            # Show game termination reason
            if board.is_checkmate():
                print("Checkmate!")
            elif board.is_stalemate():
                print("Stalemate!")
            elif board.is_insufficient_material():
                print("Insufficient material!")
            elif board.can_claim_draw():
                print("Draw by repetition or 50-move rule!")
                
            # Show final position
            if use_unicode:
                print_board_fancy(board)
            else:
                print(board)
            
            response = input("\nPress Enter to start new game or 'quit' to exit: ").strip()
            if response.lower() in ['quit', 'exit']:
                break
            board.reset()
            move_history = []
            clear_screen()
            print("New game started!")
            print("-" * 60)

        # Display board
        print()
        if board.turn:
            print("WHITE to move")
        else:
            print("BLACK to move")
            
        if use_unicode:
            print_board_fancy(board)
        else:
            print(board)

        # Human's turn
        if mode == 'h' and board.turn == color:
            # Show legal moves
            legal_moves = list(board.legal_moves)
            print(f"\nLegal moves ({len(legal_moves)}):")
            
            # Group moves by piece
            moves_by_piece = {}
            for move in legal_moves:
                piece = board.piece_at(move.from_square)
                if piece:
                    key = f"{piece.symbol().upper()} on {chess.square_name(move.from_square)}"
                    if key not in moves_by_piece:
                        moves_by_piece[key] = []
                    moves_by_piece[key].append(str(move))
            
            for piece, moves in sorted(moves_by_piece.items()):
                print(f"  {piece}: {', '.join(moves)}")

            while True:
                user_input = input("\nYour move: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    mcts_engine.stop()
                    return
                    
                if user_input.lower() == 'undo' and len(move_history) >= 2:
                    # Undo last two moves
                    board.pop()
                    board.pop()
                    move_history = move_history[:-2]
                    print("Move undone!")
                    break
                    
                if user_input.lower() == 'new':
                    board.reset()
                    move_history = []
                    clear_screen()
                    print("New game started!")
                    break
                    
                if user_input.lower() == 'fen':
                    print(f"FEN: {board.fen()}")
                    continue
                    
                if user_input.lower() == 'eval':
                    # Quick evaluation
                    print("Evaluating position...")
                    root = mcts_engine.search(board, 1000, return_root=True)
                    print(f"Evaluation: {root.getQ():.3f} (from White's perspective)")
                    root.cleanup()
                    continue
                
                # Try to parse move
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_history.append(move)
                        break
                    else:
                        print(f"Illegal move: {user_input}")
                except:
                    print(f"Invalid move format: {user_input}")
                    print("Use UCI format, e.g., e2e4, g1f3")

        else:
            # AI's turn
            print("\nAI thinking", end="")
            sys.stdout.flush()
            
            start_time = time.perf_counter()
            
            # Progress indicator
            def progress_indicator():
                while not hasattr(progress_indicator, 'stop'):
                    print(".", end="")
                    sys.stdout.flush()
                    time.sleep(0.5)
            
            import threading
            progress_thread = threading.Thread(target=progress_indicator)
            progress_thread.start()
            
            # Search for best move
            best_move = mcts_engine.search(board, num_rollouts)
            
            # Stop progress indicator
            progress_indicator.stop = True
            progress_thread.join()
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            if best_move:
                # Display move info
                print(f"\n\nAI plays: {best_move}")
                print(f"Time: {elapsed:.2f}s ({num_rollouts/elapsed:.0f} nodes/sec)")
                
                # Show principal variation if verbose
                if verbose:
                    root = mcts_engine.search(board, 100, return_root=True)
                    print("\nTop moves:")
                    edges = sorted(root.edges, key=lambda e: e.getN(), reverse=True)[:5]
                    for edge in edges:
                        if edge.getN() > 0:
                            print(f"  {edge.getMove()}: N={edge.getN():.0f}, "
                                  f"Q={edge.getQ():.3f}")
                    root.cleanup()
                
                board.push(best_move)
                move_history.append(best_move)
            else:
                print("\nAI error: No legal move found!")
                break

        # Profile mode - exit after first move
        if mode == 'p':
            break

    # Cleanup
    mcts_engine.stop()
    print("\nThanks for playing!")


def parseColor(colorString):
    """Parse color string to boolean."""
    if colorString.lower() in ['w', 'white']:
        return True
    elif colorString.lower() in ['b', 'black']:
        return False
    else:
        print('Invalid color. Use "w" or "b"')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Play chess against the AI - CUDA optimized for Windows'
    )
    parser.add_argument('--model', help='Path to model (.pt) file', required=True)
    parser.add_argument('--rollouts', type=int, help='Number of MCTS rollouts', 
                        default=10000)
    parser.add_argument('--threads', type=int, help='Number of worker threads', 
                        default=64)
    parser.add_argument('--mode', type=str, 
                        help='Play mode: h (human), s (self-play), p (profile)', 
                        default='h')
    parser.add_argument('--color', type=str, 
                        help='Human color: w (white) or b (black)', 
                        default='w')
    parser.add_argument('--fen', type=str, help='Starting position in FEN notation', 
                        default=None)
    parser.add_argument('--verbose', action='store_true', 
                        help='Verbose output with search statistics')
    parser.add_argument('--batch-size', type=int, help='Neural network batch size', 
                        default=512)
    parser.add_argument('--device', type=int, help='CUDA device ID', 
                        default=0)
    parser.add_argument('--no-unicode', action='store_true', 
                        help='Disable Unicode chess pieces')
    
    args = parser.parse_args()
    
    main(
        modelFile=args.model,
        mode=args.mode,
        color=parseColor(args.color),
        num_rollouts=args.rollouts,
        num_threads=args.threads,
        fen=args.fen,
        verbose=args.verbose,
        batch_size=args.batch_size,
        device_id=args.device,
        use_unicode=not args.no_unicode
    )