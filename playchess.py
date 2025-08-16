
import argparse
import chess
#import MCTS_profiling_speedups_v2 as MCTS
#import mcts_batched as MCTS
import mcts_gemini as MCTS
# import searchless_policy as MCTS
import torch
import time
from model_utils import load_model

# Import enhanced encoder for position history tracking
try:
    from encoder_enhanced import PositionHistory
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    print("Warning: encoder_enhanced not available, history tracking disabled")

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



def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, use_enhanced_encoder=False ):
    
    # Load model using model_utils (single GPU only)
    alphaZeroNet, device, is_quantized = load_model(modelFile)
    
    if verbose:
        print(f"DEBUG: Model device: {device}")
        print(f"DEBUG: Model type: {type(alphaZeroNet)}")
        print(f"DEBUG: Is quantized: {is_quantized}")
        import sys
        sys.stdout.flush()
   
    #create chess board object
    if fen:
        board = chess.Board( fen )
    else:
        board = chess.Board()
    
    # Create position history for enhanced encoding (only if requested and available)
    position_history = None
    if use_enhanced_encoder and HISTORY_AVAILABLE:
        # Check if model is TitanMini by checking its attributes
        is_titanmini = hasattr(alphaZeroNet, 'chess_positional_encoding') or \
                      'TitanMini' in str(type(alphaZeroNet))
        if is_titanmini:
            position_history = PositionHistory(history_length=8)
            # Add the initial position to history
            position_history.add_position(board)
            if verbose:
                print("DEBUG: Position history tracking enabled for TitanMini with enhanced encoder")
    elif use_enhanced_encoder and not HISTORY_AVAILABLE:
        if verbose:
            print("DEBUG: Enhanced encoder requested but encoder_enhanced module not available")

    #play chess moves
    while True:

        if board.is_game_over(claim_draw=True):
            #If the game is over, output the winner and wait for user input to continue
            print( 'Game over. Winner: {}'.format( board.result(claim_draw=True) ) )
            board.reset_board()
            
            # Reset position history for new game (only if using enhanced encoder)
            if use_enhanced_encoder and position_history:
                position_history.history = []
                position_history.add_position(board)
            
            c = input( 'Enter any key to continue ' )

        #Print the current state of the board
        if board.turn:
            print( 'White\'s turn' )
        else:
            print( 'Black\'s turn' )
        print( board )

        if mode == 'h' and board.turn == color:
            #If we are in human mode and it is the humans turn, play the move specified from stdin
            move_list = tolist( board.legal_moves )

            idx = -1

            while not (0 <= idx and idx < len( move_list ) ):
            
                string = input( 'Choose a move ' )

                for i, move in enumerate( move_list ):
                    if str( move ) == string:
                        idx = i
                        break
            
            board.push( move_list[ idx ] )
            
            # Update position history after human move (only if using enhanced encoder)
            if use_enhanced_encoder and position_history:
                position_history.add_position(board)

        else:
            #In all other cases the AI selects the next move
            
            if verbose:
                print(f"DEBUG: Starting AI move calculation")
                print(f"DEBUG: Device for neural network: {device}")
                import sys
                sys.stdout.flush()
            
            starttime = time.perf_counter()

            with torch.no_grad():
                total_simulations = num_threads * num_rollouts
                root = MCTS.Root(board, alphaZeroNet, position_history, use_enhanced_encoder)
                root.parallelRolloutsTotal(board.copy(), alphaZeroNet, total_simulations, num_threads)
                if verbose:
                    print(f"DEBUG: Starting {num_rollouts} rollouts with {num_threads} threads")


            endtime = time.perf_counter()

            elapsed = endtime - starttime

            edge = root.maxNSelect()
            best_move = edge.getMove()

            total_nodes_explored = root.getN()

            print(f"Total nodes explored {total_nodes_explored}")

            print( 'best move {}'.format( str( best_move ) ) )

            print(f"Elapsed time is {elapsed}")

            NPS = total_nodes_explored/elapsed

            print(f"NPS is {NPS}")
        
            board.push( best_move )
            
            # Update position history after AI move (only if using enhanced encoder)
            if use_enhanced_encoder and position_history:
                position_history.add_position(board)

        if mode == 'p':
            #In profile mode, exit after the first move
            break

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
    parser = argparse.ArgumentParser(usage='Play chess against the computer or watch self play games.')
    parser.add_argument( '--model', help='Path to model (.pt) file.' )
    parser.add_argument( '--mode', help='Operation mode: \'s\' self play, \'p\' profile, \'h\' human' )
    parser.add_argument( '--color', help='Your color w or b' )
    parser.add_argument( '--rollouts', type=int, help='The number of rollouts on computers turn' )
    parser.add_argument( '--threads', type=int, help='Number of threads used per rollout' )
    parser.add_argument( '--verbose', help='Print search statistics', action='store_true' )
    parser.add_argument( '--fen', help='Starting fen' )
    parser.add_argument( '--enhanced-encoder', action='store_true', help='Use enhanced encoder with position history (requires encoder_enhanced module)' )
    parser.set_defaults( verbose=False, mode='p', color='w', rollouts=100, threads=100, enhanced_encoder=False )
    parser = parser.parse_args()

    try:
        main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose, parser.enhanced_encoder )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

