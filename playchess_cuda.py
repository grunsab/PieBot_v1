import argparse
import chess
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
from quantization_utils import load_quantized_model

# Import the appropriate MCTS implementation based on availability
try:
    import MCTS_cuda_optimized as MCTS
    print("Using CUDA-optimized MCTS implementation")
    USING_CUDA_MCTS = True
except ImportError:
    try:
        import MCTS_advanced_optimizations as MCTS
        print("Using advanced MCTS implementation (CUDA not available)")
        USING_CUDA_MCTS = False
    except ImportError:
        import MCTS
        print("Using original MCTS implementation")
        USING_CUDA_MCTS = False

# Import helper functions from original playchess
from playchess import tolist, load_model_multi_gpu, parseColor

def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, gpu_ids=None ):
    
    # Load models (supports multi-GPU)
    models, devices = load_model_multi_gpu(modelFile, gpu_ids)
    
    # For backward compatibility, use first model/device for single GPU mode
    alphaZeroNet = models[0]
    device = devices[0]
    
    if verbose:
        print(f"DEBUG: Model device: {device}")
        print(f"DEBUG: Model type: {type(alphaZeroNet)}")
        if USING_CUDA_MCTS:
            print("DEBUG: Using CUDA-OPTIMIZED MCTS implementation")
            if hasattr(MCTS, 'CPP_AVAILABLE'):
                print(f"DEBUG: C++ extension available: {MCTS.CPP_AVAILABLE}")
            if hasattr(MCTS, 'CUDA_AVAILABLE'):
                print(f"DEBUG: CUDA extension available: {MCTS.CUDA_AVAILABLE}")
        import sys
        sys.stdout.flush()
    
    # Print GPU usage info
    if len(models) > 1:
        print(f'Using {len(models)} GPUs for neural network evaluation')
        print('Note: Multi-GPU support requires playchess_multigpu.py for full parallelization')
   
    #create chess board object
    if fen:
        board = chess.Board( fen )
    else:
        board = chess.Board()

    #play chess moves
    move_count = 0
    total_time = 0
    total_nodes = 0
    
    while True:

        if board.is_game_over():
            #If the game is over, output the winner and wait for user input to continue
            print( 'Game over. Winner: {}'.format( board.result() ) )
            
            # Clear caches if using CUDA MCTS
            if USING_CUDA_MCTS and hasattr(MCTS, 'clear_caches'):
                MCTS.clear_caches()
                if verbose:
                    print("Cleared MCTS caches")
            
            board.reset_board()
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

        else:
            #In all other cases the AI selects the next move
            
            if verbose:
                print(f"DEBUG: Starting AI move calculation")
                print(f"DEBUG: Device for neural network: {device}")
                import sys
                sys.stdout.flush()
            
            starttime = time.perf_counter()

            with torch.no_grad():
                if verbose:
                    print(f"DEBUG: Creating MCTS root")
                
                root = MCTS.Root( board, alphaZeroNet )
                
                if verbose:
                    print(f"DEBUG: Starting {num_rollouts} rollouts with {num_threads} threads")
            
                for i in range( num_rollouts ):
                    root.parallelRollouts( board.copy(), alphaZeroNet, num_threads )

            endtime = time.perf_counter()

            elapsed = endtime - starttime

            Q = root.getQ()
            # Ensure Q is a scalar value
            if hasattr(Q, 'item'):
                Q = Q.item()
            elif hasattr(Q, '__len__'):
                Q = float(Q)

            N = root.getN()

            nps = N / elapsed

            same_paths = getattr(root, 'same_paths', 0)
       
            if verbose:
                #In verbose mode, print some statistics
                print( root.getStatisticsString() )
                print( 'total rollouts {} Q {:0.3f} duplicate paths {} elapsed {:0.2f} nps {:0.2f}'.format( int( N ), Q, same_paths, elapsed, nps ) )
                
                # Print cache statistics if available
                if USING_CUDA_MCTS and hasattr(MCTS, 'position_cache'):
                    print(f'Cache stats - Positions: {len(MCTS.position_cache)}, Moves: {len(MCTS.legal_move_cache)}')
                
                # Performance tracking
                total_time += elapsed
                total_nodes += N
                print(f'Session average NPS: {total_nodes/total_time:.1f}')
     
            edge = root.maxNSelect()

            bestmove = edge.getMove()

            print( 'best move {}'.format( str( bestmove ) ) )
        
            board.push( bestmove )
            
            # Cleanup if available
            if hasattr(root, 'cleanup'):
                root.cleanup()
            
            move_count += 1
            
            # Clear caches periodically to prevent memory issues
            if USING_CUDA_MCTS and move_count % 50 == 0:
                if hasattr(MCTS, 'clear_caches'):
                    MCTS.clear_caches()
                    if verbose:
                        print("Cleared caches after 50 moves")
                if hasattr(MCTS, 'clear_batch_queue'):
                    MCTS.clear_batch_queue()

        if mode == 'p':
            #In profile mode, exit after the first move
            
            # Print final statistics
            if verbose and total_time > 0:
                print(f"\nFinal Statistics:")
                print(f"Total nodes searched: {int(total_nodes)}")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average NPS: {total_nodes/total_time:.1f}")
            
            break

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Play chess against the computer or watch self play games (CUDA-OPTIMIZED VERSION).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Notes:
- This version automatically uses CUDA optimizations if available
- For best performance on Windows CUDA systems:
  * Use high rollout counts (500-2000)
  * Use many threads (16-32)
  * Ensure CUDA extensions are built (see CUDA_OPTIMIZATION_README.md)
  
Examples:
  # Standard play
  python playchess_cuda.py --model weights/model.pt --rollouts 1000 --threads 16
  
  # Profile mode with verbose output
  python playchess_cuda.py --model weights/model.pt --mode p --rollouts 1000 --threads 32 --verbose
  
  # Human vs AI
  python playchess_cuda.py --model weights/model.pt --mode h --color w --rollouts 1500 --threads 16
        """
    )
    parser.add_argument( '--model', help='Path to model (.pt) file.' )
    parser.add_argument( '--mode', help='Operation mode: \'s\' self play, \'p\' profile, \'h\' human' )
    parser.add_argument( '--color', help='Your color w or b' )
    parser.add_argument( '--rollouts', type=int, help='The number of rollouts on computers turn' )
    parser.add_argument( '--threads', type=int, help='Number of threads used per rollout' )
    parser.add_argument( '--verbose', help='Print search statistics', action='store_true' )
    parser.add_argument( '--fen', help='Starting fen' )
    parser.add_argument( '--gpus', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.set_defaults( verbose=False, mode='p', color='w', rollouts=10, threads=1 )
    parser = parser.parse_args()
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if parser.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parser.gpus.split(',')]

    try:
        main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose, gpu_ids )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()