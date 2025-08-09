import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import chess
import chess.pgn
import MCTS
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device
from quantization_utils import load_quantized_model
from datetime import datetime
import io
from chess_openings import get_opening, get_total_openings

def load_model(model_file, device_hint=None):
    """
    Load a model from file with automatic device selection.
    
    Args:
        model_file: Path to model file
        device_hint: Optional device hint (for multi-GPU setups)
    
    Returns:
        model: Loaded model
        device: Device the model is on
    """
    if device_hint is None:
        device, device_str = get_optimal_device()
    else:
        device = device_hint
        device_str = str(device)
    
    # Always load to CPU first to check model type
    weights = torch.load(model_file, map_location='cpu')
    
    # Check if it's a static quantized model
    is_static_quantized = (isinstance(weights, dict) and 
                         'quant.scale' in weights and 
                         any(k.startswith('base_model.') for k in weights.keys()))
    
    if is_static_quantized or (isinstance(weights, dict) and weights.get('model_type') == 'static_quantized'):
        # Static quantized models run on CPU
        cpu_device = torch.device('cpu')
        try:
            # Try loading as TorchScript
            model = torch.jit.load(model_file, map_location=cpu_device)
            model.eval()
            print(f"Loaded static quantized model (TorchScript) on CPU for {model_file}")
            device = cpu_device
        except Exception as e:
            print(f"Warning: Could not load as TorchScript: {e}")
            try:
                # Try using quantization_utils
                model = load_quantized_model(model_file, cpu_device, 20, 256)
                print(f"Loaded static quantized model on CPU for {model_file}")
                device = cpu_device
            except Exception as e2:
                print(f"Warning: Static quantization not supported: {e2}")
                print("Falling back to regular model...")
                # Fall back to loading as regular model
                model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
                new_state_dict = {}
                for key, value in weights.items():
                    if key.startswith('base_model.'):
                        new_key = key.replace('base_model.', '')
                        if any(x in new_key for x in ['.scale', '.zero_point', '_packed_params']):
                            continue
                        if hasattr(value, 'dequantize'):
                            new_state_dict[new_key] = value.dequantize()
                        else:
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
                model.to(device)
                model.eval()
                print(f"Loaded dequantized model on {device_str} for {model_file}")
    else:
        # Create regular model
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        
        # Handle different model formats
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            # FP16 model format
            model.load_state_dict(weights['model_state_dict'])
            if weights.get('model_type') == 'fp16':
                model = model.half()
                print(f"Loaded FP16 model on {device_str} for {model_file}")
        else:
            # Regular model format
            model.load_state_dict(weights)
            print(f"Loaded model on {device_str} for {model_file}")
        
        model = optimize_for_device(model, device)
        model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, device

def play_game(model1, device1, model2, device2, rollouts=100, threads=1, verbose=False, opening_index=None):
    """
    Play a single game between two models.
    
    Args:
        model1: First model (plays as white)
        device1: Device for first model
        model2: Second model (plays as black)
        device2: Device for second model
        rollouts: Number of MCTS rollouts per move
        threads: Number of threads for MCTS
        verbose: Print move details
        opening_index: Index of the opening to use (None for no opening)
    
    Returns:
        result: Game result ('1-0', '0-1', '1/2-1/2')
        pgn_game: chess.pgn.Game object
        move_count: Number of moves in the game
    """
    board = chess.Board()
    
    # Create PGN game
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Model vs Model Test"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["White"] = "Model 1"
    pgn_game.headers["Black"] = "Model 2"
    
    # Add opening information if using one
    opening_moves = []
    opening_name = None
    if opening_index is not None:
        opening_data = get_opening(opening_index)
        opening_moves = opening_data["moves"]
        opening_name = opening_data["name"]
        pgn_game.headers["Opening"] = opening_name
        if verbose:
            print(f"Using opening: {opening_name}")
    
    node = pgn_game
    move_count = 0
    opening_move_index = 0
    
    while not board.is_game_over():
        move_count += 1
        
        # Select the model based on whose turn it is
        if board.turn:  # White's turn
            current_model = model1
            current_device = device1
            player_name = "Model 1 (White)"
        else:  # Black's turn
            current_model = model2
            current_device = device2
            player_name = "Model 2 (Black)"
        
        if verbose:
            print(f"\nMove {move_count}. {player_name}'s turn")
            print(board)
        
        # Check if we should use opening moves
        if opening_moves and opening_move_index < len(opening_moves):
            # Use opening move
            best_move = chess.Move.from_uci(opening_moves[opening_move_index])
            opening_move_index += 1
            
            if verbose:
                print(f"Opening move: {best_move}")
                if opening_move_index == len(opening_moves):
                    print("Opening sequence complete. Switching to MCTS.")
        else:
            # Use MCTS to find the best move
            start_time = time.perf_counter()
            
            with torch.no_grad():
                root = MCTS.Root(board, current_model)
                
                for i in range(rollouts):
                    root.parallelRollouts(board.copy(), current_model, threads)
            
            elapsed = time.perf_counter() - start_time
            
            # Get the best move
            edge = root.maxNSelect()
            best_move = edge.getMove()
            
            if verbose:
                Q = root.getQ()
                N = root.getN()
                nps = N / elapsed if elapsed > 0 else 0
                print(f"Best move: {best_move}, Q: {Q:.3f}, N: {int(N)}, Time: {elapsed:.2f}s, NPS: {nps:.1f}")
        
        # Make the move
        board.push(best_move)
        node = node.add_variation(best_move)
    
    # Get the result
    result = board.result()
    pgn_game.headers["Result"] = result
    
    if verbose:
        print(f"\nGame over! Result: {result}")
        print(f"Total moves: {move_count}")
    
    return result, pgn_game, move_count

def main():
    parser = argparse.ArgumentParser(description='Test two chess models against each other')
    parser.add_argument('--model1', required=True, help='Path to first model (.pt) file')
    parser.add_argument('--model2', required=True, help='Path to second model (.pt) file')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play (default: 10)')
    parser.add_argument('--rollouts', type=int, default=50, help='Number of MCTS rollouts per thread (default: 50)')
    parser.add_argument('--threads', type=int, default=60, help='Number of threads for MCTS (default: 60)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed move information')
    parser.add_argument('--pgn', help='Save games to PGN file')
    parser.add_argument('--alternate', action='store_true', help='Alternate which model plays white')
    parser.add_argument('--json', help='Output results in JSON format to specified file')
    parser.add_argument('--no-openings', action='store_true', help='Disable chess openings and play from starting position')
    
    args = parser.parse_args()
    
    print(f"Loading models...")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    
    # Load both models
    model1, device1 = load_model(args.model1)
    model2, device2 = load_model(args.model2)
    
    print(f"\nStarting tournament: {args.games} games")
    print(f"Rollouts per move: {args.rollouts}")
    print(f"Threads per rollout: {args.threads}")
    if args.alternate:
        print("Models will alternate playing as white")
    if not args.no_openings:
        print(f"Using {get_total_openings()} different chess openings (cycling through)")
    else:
        print("Playing from standard starting position (no openings)")
    print("-" * 60)
    
    # Track results
    results = {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0,
        'total_moves': 0,
        'games': []
    }
    
    # Play the games
    start_time = time.time()
    
    for game_num in range(args.games):
        print(f"\nGame {game_num + 1}/{args.games}")
        
        # Determine which model plays white
        if args.alternate and game_num % 2 == 1:
            # Swap models for odd-numbered games
            opening_idx = None if args.no_openings else game_num
            result, pgn_game, move_count = play_game(
                model2, device2, model1, device1, 
                args.rollouts, args.threads, args.verbose, opening_index=opening_idx
            )
            # Adjust result interpretation since models are swapped
            if result == '1-0':
                results['model2_wins'] += 1
                winner = "Model 2"
            elif result == '0-1':
                results['model1_wins'] += 1
                winner = "Model 1"
            else:
                results['draws'] += 1
                winner = "Draw"
            
            # Update PGN headers for swapped game
            pgn_game.headers["White"] = "Model 2"
            pgn_game.headers["Black"] = "Model 1"
        else:
            # Normal game order
            opening_idx = None if args.no_openings else game_num
            result, pgn_game, move_count = play_game(
                model1, device1, model2, device2, 
                args.rollouts, args.threads, args.verbose, opening_index=opening_idx
            )
            if result == '1-0':
                results['model1_wins'] += 1
                winner = "Model 1"
            elif result == '0-1':
                results['model2_wins'] += 1
                winner = "Model 2"
            else:
                results['draws'] += 1
                winner = "Draw"
        
        results['total_moves'] += move_count
        results['games'].append(pgn_game)
        
        print(f"Result: {result} ({winner} wins)" if winner != "Draw" else f"Result: {result}")
        print(f"Moves: {move_count}")
        
        # Print running score
        print(f"Score: Model 1: {results['model1_wins']} | Model 2: {results['model2_wins']} | Draws: {results['draws']}")
    
    total_time = time.time() - start_time
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total games: {args.games}")
    print(f"Model 1 wins: {results['model1_wins']} ({results['model1_wins']/args.games*100:.1f}%)")
    print(f"Model 2 wins: {results['model2_wins']} ({results['model2_wins']/args.games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/args.games*100:.1f}%)")
    print(f"\nAverage game length: {results['total_moves']/args.games:.1f} moves")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per game: {total_time/args.games:.1f} seconds")
    
    # Determine overall winner
    if results['model1_wins'] > results['model2_wins']:
        print(f"\nModel 1 wins the match! (+{results['model1_wins'] - results['model2_wins']})")
    elif results['model2_wins'] > results['model1_wins']:
        print(f"\nModel 2 wins the match! (+{results['model2_wins'] - results['model1_wins']})")
    else:
        print(f"\nThe match is a draw!")
    
    # Save PGN if requested
    if args.pgn:
        with open(args.pgn, 'w') as f:
            for game in results['games']:
                print(game, file=f)
                print("", file=f)  # Empty line between games
        print(f"\nGames saved to {args.pgn}")
    
    # Save JSON results if requested
    if args.json:
        json_results = {
            'model1': args.model1,
            'model2': args.model2,
            'games': args.games,
            'model1_wins': results['model1_wins'],
            'model2_wins': results['model2_wins'],
            'draws': results['draws'],
            'model1_win_rate': results['model1_wins'] / args.games,
            'model2_win_rate': results['model2_wins'] / args.games,
            'draw_rate': results['draws'] / args.games,
            'average_game_length': results['total_moves'] / args.games,
            'total_time': total_time,
            'time_per_game': total_time / args.games,
            'winner': 'model1' if results['model1_wins'] > results['model2_wins'] else ('model2' if results['model2_wins'] > results['model1_wins'] else 'draw')
        }
        import json
        with open(args.json, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results saved to {args.json}")

if __name__ == '__main__':
    main()