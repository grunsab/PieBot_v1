import argparse
import chess
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
from MCTS_async import MCTSEngine
from async_neural_net_server import NeuralNetworkPool


def tolist( move_generator ):
    """
    Change an iterable object of moves to a list of moves.
    
    Args:
        move_generator (Mainline object) iterable list of moves

    Returns:
        moves (list of chess.Move) list version of the input moves
    """
    moves = []
    for move in move_generator:
        moves.append( move )
    return moves


def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, 
         batch_size=256, gpu_ids=None ):
    
    # Load model and create MCTS engine
    device, device_str = get_optimal_device()
    print(f'Using device: {device_str}')
    
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    weights = torch.load(modelFile, map_location=device)
    model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Create MCTS engine with async neural network
    print(f"Initializing async MCTS engine")
    print(f"Workers: {num_threads}, Batch size: {batch_size}")
    
    if gpu_ids and len(gpu_ids) > 1:
        # Multi-GPU setup
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        nn_pool = NeuralNetworkPool(
            model, 
            num_gpus=len(gpu_ids),
            max_batch_size=batch_size,
            verbose=verbose
        )
        nn_pool.start()
        # Use first GPU's server for simplicity in interactive play
        mcts_engine = MCTSEngine(
            model,
            device=nn_pool.servers[0].device,
            max_batch_size=batch_size,
            num_workers=num_threads,
            verbose=verbose
        )
        mcts_engine.nn_server = nn_pool.servers[0]
    else:
        # Single device setup
        mcts_engine = MCTSEngine(
            model,
            device=device,
            max_batch_size=batch_size,
            num_workers=num_threads,
            verbose=verbose
        )
        mcts_engine.start()
   
    # Create chess board object
    if fen:
        board = chess.Board( fen )
    else:
        board = chess.Board()

    # Play chess moves
    print("\nStarting chess game...")
    print("Commands: 'quit' to exit, 'undo' to take back last move")
    print("-" * 50)
    
    while True:

        if board.is_game_over():
            # If the game is over, output the winner and wait for user input to continue
            print( f'\nGame over. Result: {board.result()}' )
            
            # Show final position
            print("\nFinal position:")
            print( board )
            
            c = input( '\nPress Enter to start a new game or "quit" to exit: ' )
            if c.lower() == 'quit':
                break
            board.reset()
            print("\nNew game started!")
            print("-" * 50)

        # Print the current state of the board
        print()
        if board.turn:
            print( 'White to move' )
        else:
            print( 'Black to move' )
        print( board )

        if mode == 'h' and board.turn == color:
            # If we are in human mode and it is the human's turn, play the move specified from stdin
            move_list = tolist( board.legal_moves )

            # Show legal moves
            print("\nLegal moves:")
            moves_per_line = 8
            for i in range(0, len(move_list), moves_per_line):
                move_strs = [str(move) for move in move_list[i:i+moves_per_line]]
                print("  " + " ".join(f"{m:6}" for m in move_strs))

            while True:
                string = input( '\nEnter your move: ' ).strip()
                
                if string.lower() == 'quit':
                    mcts_engine.stop()
                    if 'nn_pool' in locals():
                        nn_pool.stop()
                    return
                    
                if string.lower() == 'undo' and len(board.move_stack) > 0:
                    board.pop()
                    if len(board.move_stack) > 0:
                        board.pop()  # Also undo computer's move
                    print("\nMove undone!")
                    break
                
                # Try to parse the move
                move_found = False
                for move in move_list:
                    if str(move) == string:
                        board.push(move)
                        move_found = True
                        break
                
                if move_found:
                    break
                else:
                    print(f"Invalid move: '{string}'. Please try again.")

        else:
            # In all other cases the AI selects the next move
            print("\nAI thinking...")
            
            starttime = time.perf_counter()

            # Use async MCTS engine
            best_move = mcts_engine.search(board, num_rollouts)

            endtime = time.perf_counter()
            elapsed = endtime - starttime

            if best_move:
                print( f'\nAI plays: {best_move}' )
                print( f'Time: {elapsed:.2f}s ({num_rollouts/elapsed:.0f} nodes/sec)' )
                board.push( best_move )
            else:
                print("AI error: No legal move found!")
                break

        if mode == 'p':
            # In profile mode, exit after the first move
            break

    # Cleanup
    mcts_engine.stop()
    if 'nn_pool' in locals():
        nn_pool.stop()
    
    print("\nThanks for playing!")


def parseColor( colorString ):
    """
    Maps 'w' to True and 'b' to False.

    Args:
        colorString (string) a string representing white or black

    """

    if colorString == 'w' or colorString == 'W':
        return True
    elif colorString == 'b' or colorString == 'B':
        return False
    else:
        print( 'Unrecognized argument for color' )
        exit()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Play chess against the AI using async MCTS engine for high performance.'
    )
    parser.add_argument( '--model', help='Path to model (.pt) file.', required=True )
    parser.add_argument( '--rollouts', type=int, help='Number of rollouts.', default=1000 )
    parser.add_argument( '--threads', type=int, help='Number of worker threads.', default=32 )
    parser.add_argument( '--mode', type=str, help='Play mode: h (human), s (self-play), p (profile).', 
                        default='h' )
    parser.add_argument( '--color', type=str, help='Color to play as human: w (white) or b (black).', 
                        default='w' )
    parser.add_argument( '--fen', type=str, help='Starting position in FEN notation.', default=None )
    parser.add_argument( '--verbose', action='store_true', help='Verbose output with search statistics.' )
    parser.add_argument( '--batch-size', type=int, help='Neural network batch size.', default=256 )
    parser.add_argument( '--gpus', type=int, nargs='+', help='GPU IDs to use (e.g., --gpus 0 1)' )
    
    args = parser.parse_args()
    
    if args.mode == 'h':
        print("\n" + "="*60)
        print("   AlphaZero Chess Engine - Async High Performance Edition")
        print("="*60)
        
    main( 
        modelFile=args.model,
        mode=args.mode,
        color=parseColor(args.color),
        num_rollouts=args.rollouts,
        num_threads=args.threads,
        fen=args.fen,
        verbose=args.verbose,
        batch_size=args.batch_size,
        gpu_ids=args.gpus
    )