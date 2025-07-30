
import argparse
import chess
import MCTS
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
from quantization_utils import load_quantized_model

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

def load_model_multi_gpu(model_file, gpu_ids=None):
    """
    Load model on specified GPUs or automatically select GPUs.
    
    Args:
        model_file: Path to model file
        gpu_ids: List of GPU IDs to use, or None for auto-selection
    
    Returns:
        models: List of models (one per GPU)
        devices: List of devices
    """
    available_gpus = get_gpu_count()
    
    if gpu_ids is None:
        # Auto-select: use first GPU for single GPU, or all for multi-GPU
        if available_gpus > 0:
            gpu_ids = [0]  # Default to single GPU for compatibility
        else:
            gpu_ids = []
    
    if not gpu_ids or available_gpus == 0:
        device, device_str = get_optimal_device()
        print(f'Using device: {device_str}')
        
        # Always load to CPU first to check model type
        weights = torch.load(model_file, map_location='cpu')
        
        # Debug: print model type
        if isinstance(weights, dict):
            print(f"Model type in weights: {weights.get('model_type', 'not specified')}")
            print(f"Keys in weights dict: {list(weights.keys())[:5]}...")  # Print first 5 keys
        
        # Check if it's a static quantized model first
        if isinstance(weights, dict) and weights.get('model_type') == 'static_quantized':
            # Static quantized models run on CPU
            cpu_device = torch.device('cpu')
            try:
                # Try loading as TorchScript
                model = torch.jit.load(model_file, map_location=cpu_device)
                model.eval()
                print(f"Loaded static quantized model (TorchScript) on CPU")
            except:
                # Load using quantization_utils
                model = load_quantized_model(model_file, cpu_device, 20, 256)
                print(f"Loaded static quantized model on CPU")
            device = cpu_device
            device_str = 'CPU (static quantized model)'
        else:
            # Create regular model
            model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
            
            # Handle different model formats
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                # FP16 model format
                model.load_state_dict(weights['model_state_dict'])
                if weights.get('model_type') == 'fp16':
                    model = model.half()
                    print(f"Loaded FP16 model on {device_str}")
            else:
                # Regular model format
                model.load_state_dict(weights)
            
            model = optimize_for_device(model, device)
            model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
            
        return [model], [device]
    
    # Multi-GPU setup
    models = []
    devices = []
    
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            print(f"Warning: GPU {gpu_id} not available, skipping")
            continue
            
        device = torch.device(f'cuda:{gpu_id}')
        devices.append(device)
        
        # Always load to CPU first to check model type
        weights = torch.load(model_file, map_location='cpu')
        
        # Check if it's a static quantized model first
        if isinstance(weights, dict) and weights.get('model_type') == 'static_quantized':
            # Static quantized models run on CPU
            cpu_device = torch.device('cpu')
            try:
                # Try loading as TorchScript
                model = torch.jit.load(model_file, map_location=cpu_device)
                model.eval()
                print(f"Loaded static quantized model (TorchScript) on CPU (GPU {gpu_id} requested but quantized models run on CPU)")
            except:
                # Load using quantization_utils
                model = load_quantized_model(model_file, cpu_device, 20, 256)
                print(f"Loaded static quantized model on CPU (GPU {gpu_id} requested but quantized models run on CPU)")
            device = cpu_device
            devices[len(models)] = cpu_device
        else:
            # Create regular model
            model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
            
            # Handle different model formats
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                # FP16 model format
                model.load_state_dict(weights['model_state_dict'])
                if weights.get('model_type') == 'fp16':
                    model = model.half()
                    print(f"Loaded FP16 model on GPU {gpu_id}")
            else:
                # Regular model format
                model.load_state_dict(weights)
        
        # Only move to device if not static quantized (already on CPU)
        if not (isinstance(weights, dict) and weights.get('model_type') == 'static_quantized'):
            model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
        print(f'Loaded model on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    
    return models, devices

def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, gpu_ids=None ):
    
    # Load models (supports multi-GPU)
    models, devices = load_model_multi_gpu(modelFile, gpu_ids)
    
    # For backward compatibility, use first model/device for single GPU mode
    alphaZeroNet = models[0]
    device = devices[0]
    
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
    while True:

        if board.is_game_over():
            #If the game is over, output the winner and wait for user input to continue
            print( 'Game over. Winner: {}'.format( board.result() ) )
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
            
            starttime = time.perf_counter()

            with torch.no_grad():

                root = MCTS.Root( board, alphaZeroNet )
            
                for i in range( num_rollouts ):
                    root.parallelRollouts( board.copy(), alphaZeroNet, num_threads )

            endtime = time.perf_counter()

            elapsed = endtime - starttime

            Q = root.getQ()

            N = root.getN()

            nps = N / elapsed

            same_paths = root.same_paths
       
            if verbose:
                #In verbose mode, print some statistics
                print( root.getStatisticsString() )
                print( 'total rollouts {} Q {:0.3f} duplicate paths {} elapsed {:0.2f} nps {:0.2f}'.format( int( N ), Q, same_paths, elapsed, nps ) )
     
            edge = root.maxNSelect()

            bestmove = edge.getMove()

            print( 'best move {}'.format( str( bestmove ) ) )
        
            board.push( bestmove )

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
    parser.add_argument( '--gpus', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.set_defaults( verbose=False, mode='p', color='w', rollouts=10, threads=1 )
    parser = parser.parse_args()
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if parser.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parser.gpus.split(',')]

    main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose, gpu_ids )

