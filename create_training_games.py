
import argparse
import os
import sys
import chess
import chess.pgn
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
from RLDataset import SelfPlayDataCollector
import numpy as np
import encoder
import MCTS

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
        
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(model_file, map_location=device)
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
        
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(model_file, map_location=device)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
        print(f'Loaded model on GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    
    return models, devices


def main( modelFile, mode, color, num_rollouts, num_threads, fen, verbose, output_directory, offset, file_base, games_to_play, gpu_ids=None, save_format='pgn', temperature=1.0, iteration=0 ):
    def get_fileName(games_played, offset, file_base, extension='pgn'):
        return f'{output_directory}/{file_base}_{games_played + offset}.{extension}'

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


    games_played = 0
    total_positions = 0
    
    
    # Initialize data collector if saving in HDF5 format
    if save_format == 'h5':
        file_name = get_fileName(games_played, offset, file_base, 'h5')
        current_collector = SelfPlayDataCollector(file_name, iteration=iteration)
    else:
        current_collector = None
    
    #play chess moves
    while True and games_played < games_to_play:
        if board.is_game_over():
            print( 'Game over. Winner: {}'.format( board.result() ) )
            
            if save_format == 'pgn':
                # Save as PGN
                game = chess.pgn.Game()
                game.add_variation(board.move_stack[0])          
                node = game.add_variation(board.move_stack[1])
        
                for i in range(2,len(board.move_stack)):
                    node = node.add_variation(board.move_stack[i])
                
                game.headers["Event"] = "PieBot vs. PieBot"
                game.headers["Site"] = "Local Machine"
                game.headers["Date"] = "2025.07.29"
                game.headers["Round"] = "1"
                game.headers["White"] = "PieBot"
                game.headers["Black"] = "PieBot"
                game.headers["Result"] = board.result()

                file_name = get_fileName(games_played, offset, file_base)
                with open(file_name, "w+") as f:
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
            
            elif save_format == 'h5' and current_collector:
                # Finalize and save HDF5 data
                current_collector.end_game(board.result())
                current_collector.save()
                current_collector = None
            
            games_played += 1
            board.reset()
            
            # Start new data collection for next game
            if save_format == 'h5':
                file_name = get_fileName(games_played, offset, file_base, 'h5')
                current_collector = SelfPlayDataCollector(file_name, iteration=iteration)

        #Print the current state of the board
        if board.turn:
            print( 'White\'s turn' )
        else:
            print( 'Black\'s turn' )
        print( board )

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
    
        # Get visit counts for training data
        if save_format == 'h5' and current_collector:
            visit_counts = root.getVisitCounts(board)
            legal_move_mask = encoder.getLegalMoveMask(board)
            current_collector.add_position(board, visit_counts, legal_move_mask)
            total_positions += 1
        
        # Select move based on visit counts (with temperature for exploration)
        if temperature > 0:
            # Temperature-based selection for training
            visits = np.array([edge.getN() for edge in root.edges])
            if visits.sum() > 0:
                # Apply temperature
                visits_temp = np.power(visits, 1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
                move_idx = np.random.choice(len(root.edges), p=probs)
                edge = root.edges[move_idx]
            else:
                edge = root.maxNSelect()
        else:
            # Greedy selection (temperature = 0)
            edge = root.maxNSelect()

        bestmove = edge.getMove()
        print( 'best move {}'.format( str( bestmove ) ) )
    
        board.push( bestmove )
        
        # Count position for all formats
        if save_format == 'pgn':
            total_positions += 1
    
    # Print total positions at the end
    print(f"TOTAL_POSITIONS: {total_positions}")


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
    parser = argparse.ArgumentParser(usage='Create self play games.')
    parser.add_argument( '--mode', help='Operation mode: \'u\' unlimited self-play , \'s\' stop self-play creation at 1MM games,')
    parser.add_argument( '--model', default="AlphaZeroNet_20x256_distributed.pt", help='Path to model (.pt) file.' )
    parser.add_argument( '--rollouts', type=int, help='The number of rollouts on computers turn. Total rollouts is this times number of threads.' )
    parser.add_argument( '--games-to-play', type=int, default=25000, help='The number of games to play' )
    parser.add_argument( '--offset', type=int, default=0, help='The offset to use for the file number in the PGN' )
    parser.add_argument( '--file-base', default='self_play', help='The file base name for the PGN (changing this will allow you to avoid looking up offset)' )
    parser.add_argument( '--threads', type=int, help='Number of threads used per rollout' )
    parser.add_argument( '--verbose', help='Print search statistics', action='store_true' )
    parser.add_argument( '--fen', help='Starting fen' )
    parser.add_argument( '--gpus', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--output-dir', default='games_training_data/selfplay', help='Output directory (default: games_training_data/selfplay)')
    parser.add_argument('--save-format', choices=['pgn', 'h5'], default='pgn', help='Output format: pgn for games, h5 for training data (default: pgn)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection (0 for greedy, >0 for exploration, default: 1.0)')
    parser.add_argument('--iteration', type=int, default=0, help='Training iteration number for metadata (default: 0)')
    parser.set_defaults( verbose=False, mode='s', color='w', rollouts=10, threads=20)
    parser = parser.parse_args()
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if parser.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parser.gpus.split(',')]

    main( parser.model, parser.mode, parseColor( parser.color ), parser.rollouts, parser.threads, parser.fen, parser.verbose, parser.output_dir, 
        parser.offset, parser.file_base, parser.games_to_play, gpu_ids, parser.save_format, parser.temperature, parser.iteration )

