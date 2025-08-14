
import argparse
import chess
import MCTS_profiling_speedups_v2 as MCTS
# import searchless_policy as MCTS
import torch
import AlphaZeroNetwork
import PieBotNetwork
import PieNanoNetwork
import PieNanoNetwork_v2
import TitanMiniNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
try:
    from quantization_tools.quantization_utils_titanmini import load_quantized_model
except ImportError:
    load_quantized_model = None

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


def detect_model_type(weights, model_path=None):
    """Detect whether this is an AlphaZeroNet, PieBotNet, PieNano, PieNanoV2, or TitanMini model."""
    # Check if it's a quantized model by filename
    if model_path and ('quantized' in model_path.lower() or 'quant' in model_path.lower() or '_q' in model_path.lower()):
        # Try to determine the base model type from filename
        if 'titan' in model_path.lower() or 'titanmini' in model_path.lower():
            return 'TitanMini_Quantized'
        elif 'pienano' in model_path.lower() or 'pie_nano' in model_path.lower():
            return 'PieNano_Quantized'
        # Could be a quantized model without clear naming
        return 'Unknown_Quantized'
    
    if isinstance(weights, dict):
        # Check state dict keys to determine model type
        state_dict = weights.get('model_state_dict', weights)
        
        # TitanMini has specific modules like chess_positional_encoding and relative_position_bias
        has_chess_pos_encoding = any('chess_positional_encoding' in key for key in state_dict.keys())
        has_relative_pos_bias = any('relative_position_bias' in key for key in state_dict.keys())
        has_geglu = any('geglu' in key.lower() for key in state_dict.keys())
        has_cls_token = any('cls_token' in key for key in state_dict.keys())
        
        # PieBotNet has specific modules like positional_encoding and transformer_blocks
        has_positional_encoding = any('positional_encoding' in key for key in state_dict.keys())
        has_transformer = any('transformer_blocks' in key for key in state_dict.keys())
        
        # PieNano models have SE (Squeeze-Excitation) modules and depthwise convolutions
        has_se = any('se.' in key or 'squeeze' in key or 'excitation' in key for key in state_dict.keys())
        has_depthwise = any('depthwise' in key for key in state_dict.keys())
        has_wdl_value = any('value_head.fc2' in key for key in state_dict.keys())
        
        # PieNanoV2 has the improved policy head with fc1 and fc2 in policy_head
        has_improved_policy = any('policy_head.fc1' in key or 'policy_head.fc2' in key for key in state_dict.keys())
        
        if has_chess_pos_encoding or has_relative_pos_bias or has_geglu or has_cls_token:
            return 'TitanMini'
        elif has_positional_encoding or has_transformer:
            return 'PieBotNet'
        elif has_improved_policy and (has_se or has_depthwise):
            return 'PieNanoV2'
        elif (has_se or has_depthwise) and has_wdl_value:
            return 'PieNano'
    return 'AlphaZeroNet'

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
        
        # Try loading as TorchScript first (for quantized models)
        is_quantized = False
        try:
            # Attempt to load as TorchScript
            model = torch.jit.load(model_file, map_location='cpu')
            is_quantized = True
            model.eval()
            print(f"Loaded quantized model (TorchScript) on CPU")
            device = torch.device('cpu')  # Quantized models run on CPU
            for param in model.parameters():
                param.requires_grad = False
            return [model], [device]
        except:
            # Not a TorchScript file, load normally
            pass
        
        # Load weights to check model type
        weights = torch.load(model_file, map_location='cpu', weights_only=False)
        
        # Detect and create the appropriate model type
        model_type = detect_model_type(weights, model_file)
        
        if model_type == 'TitanMini_Quantized':
            # Handle quantized TitanMini
            try:
                # Use model_utils function for loading quantized TitanMini
                from model_utils import load_quantized_titanmini
                model = load_quantized_titanmini(model_file)
                is_quantized = True
                print(f"Loaded quantized TitanMini model on CPU")
                device = torch.device('cpu')  # Quantized models run on CPU
            except Exception as e:
                print(f"Warning: Failed to load as quantized TitanMini: {e}")
                # Fallback: create regular TitanMini using model_utils
                from model_utils import create_titanmini_from_weights
                model = create_titanmini_from_weights(weights)
                print(f"Loading TitanMini model (dequantized fallback) on {device_str}")
        elif model_type == 'TitanMini':
            # Use model_utils to create TitanMini with detected configuration
            from model_utils import create_titanmini_from_weights
            model = create_titanmini_from_weights(weights)
            print(f"Loading TitanMini model on {device_str}")
        elif model_type == 'PieNano_Quantized':
            # Handle quantized PieNano saved as state dict
            try:
                # Try using quantization_utils
                model = load_quantized_model(model_file)
                is_quantized = True
                print(f"Loaded quantized PieNano model on CPU")
                device = torch.device('cpu')  # Quantized models run on CPU
            except:
                # Fallback: create regular PieNano and load state dict
                model = PieNanoNetwork.PieNano(num_blocks=8, num_filters=128)
                print(f"Loading PieNano model (dequantized fallback) on {device_str}")
        elif model_type == 'PieBotNet':
            # Use default PieBotNet configuration
            model = PieBotNetwork.PieBotNet()
            print(f"Loading PieBotNet model on {device_str}")
        elif model_type == 'PieNanoV2':
            # Use PieNanoV2 configuration matching the saved weights
            # The saved weights use policy_hidden_dim=768 instead of default 256
            model = PieNanoNetwork_v2.PieNanoV2(num_blocks=16, num_filters=256, policy_hidden_dim=768)
            print(f"Loading PieNanoV2 model on {device_str}")
        elif model_type == 'PieNano':
            # Use default PieNano configuration (8 blocks, 128 filters)
            model = PieNanoNetwork.PieNano(num_blocks=8, num_filters=128)
            print(f"Loading PieNano model on {device_str}")
        else:
            # Get state dict for AlphaZeroNet configuration detection
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                check_dict = weights['model_state_dict']
            else:
                check_dict = weights
            
            # Try to detect AlphaZeroNet configuration from state dict size
            if 'residualBlocks.19.conv1.weight' in check_dict:
                model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
                print(f"Loading AlphaZeroNet 20x256 model on {device_str}")
            elif 'residualBlocks.9.conv1.weight' in check_dict:
                # Check filter size to distinguish 10x128 from 10x256
                conv_shape = check_dict['convBlock1.conv1.weight'].shape
                num_filters = conv_shape[0]
                model = AlphaZeroNetwork.AlphaZeroNet(10, num_filters)
                print(f"Loading AlphaZeroNet 10x{num_filters} model on {device_str}")
            else:
                # Default to 20x256
                model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
                print(f"Loading AlphaZeroNet model on {device_str}")
        
        # Only load state dict if not quantized
        if not is_quantized:
            # Handle different model formats
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                state_dict = weights['model_state_dict']
            else:
                state_dict = weights
            
            # Handle _orig_mod prefix from torch.compile
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('_orig_mod.'):
                        new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            
            # Handle FP16 models
            if isinstance(weights, dict) and weights.get('model_type') == 'fp16':
                model = model.half()
                print(f"Loaded FP16 model on {device_str}")
            
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
        
        # Load weights to check model type
        weights = torch.load(model_file, map_location='cpu', weights_only=False)
        
        # Detect and create the appropriate model type
        model_type = detect_model_type(weights, model_file)
        if model_type == 'TitanMini':
            # Use model_utils to create TitanMini with detected configuration
            from model_utils import create_titanmini_from_weights
            model = create_titanmini_from_weights(weights)
            print(f"Loading TitanMini model on GPU {gpu_id}")
        elif model_type == 'PieBotNet':
            # Use default PieBotNet configuration
            model = PieBotNetwork.PieBotNet()
            print(f"Loading PieBotNet model on GPU {gpu_id}")
        elif model_type == 'PieNanoV2':
            # Use default PieNanoV2 configuration (16 blocks, 256 filters)
            model = PieNanoNetwork_v2.PieNanoV2(num_blocks=16, num_filters=256)
            print(f"Loading PieNanoV2 model on GPU {gpu_id}")
        elif model_type == 'PieNano':
            # Use default PieNano configuration (8 blocks, 128 filters)
            model = PieNanoNetwork.PieNano(num_blocks=8, num_filters=128)
            print(f"Loading PieNano model on GPU {gpu_id}")
        else:
            model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
            print(f"Loading AlphaZeroNet model on GPU {gpu_id}")
        
        # Handle different model formats
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            state_dict = weights['model_state_dict']
        else:
            state_dict = weights
        
        # Handle _orig_mod prefix from torch.compile
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        
        # Handle FP16 models
        if isinstance(weights, dict) and weights.get('model_type') == 'fp16':
            model = model.half()
            print(f"Loaded FP16 model on GPU {gpu_id}")
        
        model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
        print(f'Loaded model on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    
    return models, devices

def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, gpu_ids=None, num_processes=1, use_enhanced_encoder=False ):
    
    # Load models (supports multi-GPU)
    models, devices = load_model_multi_gpu(modelFile, gpu_ids)
    
    # For backward compatibility, use first model/device for single GPU mode
    alphaZeroNet = models[0]
    device = devices[0]
    
    if verbose:
        print(f"DEBUG: Model device: {device}")
        print(f"DEBUG: Model type: {type(alphaZeroNet)}")
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
    
    # Create position history for enhanced encoding (only if requested and available)
    position_history = None
    if use_enhanced_encoder and HISTORY_AVAILABLE and isinstance(alphaZeroNet, TitanMiniNetwork.TitanMini):
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
    parser.add_argument( '--gpus', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument( '--processes', type=int, help='Number of processes for multiprocessing (default: 1)' )
    parser.add_argument( '--enhanced-encoder', action='store_true', help='Use enhanced encoder with position history (requires encoder_enhanced module)' )
    parser.set_defaults( verbose=False, mode='p', color='w', rollouts=100, threads=100, processes=1, enhanced_encoder=False )
    parser = parser.parse_args()
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if parser.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parser.gpus.split(',')]

    try:
        main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose, gpu_ids, parser.processes, parser.enhanced_encoder )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

