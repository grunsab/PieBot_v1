import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import chess
import chess.pgn
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device
from datetime import datetime
import importlib.util
from chess_openings import get_opening, get_total_openings

def load_mcts_module(mcts_file):
    """
    Load an MCTS module from a Python file.
    
    Args:
        mcts_file: Path to MCTS implementation file
    
    Returns:
        mcts_module: The loaded MCTS module
    """
    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(mcts_file))[0]
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, mcts_file)
    if spec is None:
        raise ImportError(f"Could not load module spec from {mcts_file}")
    
    mcts_module = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules with the proper module name for pickle compatibility
    sys.modules[module_name] = mcts_module
    
    # Execute the module
    spec.loader.exec_module(mcts_module)
    
    # Verify the module has the required components
    required_attrs = ['Root', ]
    for attr in required_attrs:
        if not hasattr(mcts_module, attr):
            raise AttributeError(f"MCTS module {mcts_file} missing required attribute: {attr}")
    
    print(f"Loaded MCTS module from {mcts_file}")
    return mcts_module

def load_model(model_file):
    """
    Load a chess model for use with MCTS.
    
    Args:
        model_file: Path to model file
    
    Returns:
        model: Loaded model
        device: Device the model is on
    """
    device, device_str = get_optimal_device()
    
    # Create model
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    
    # Load weights
    weights = torch.load(model_file, map_location=device)
    
    # Handle different model formats
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Loaded model from {model_file} on {device_str}")
    return model, device

def play_game(mcts1_module, mcts2_module, model, device, rollouts=100, threads=1, verbose=False, opening_index=None):
    """
    Play a single game between two MCTS implementations.
    
    Args:
        mcts1_module: First MCTS module (plays as white)
        mcts2_module: Second MCTS module (plays as black)
        model: Neural network model
        device: Device for model
        rollouts: Number of MCTS rollouts per move
        threads: Number of threads for MCTS
        verbose: Print move details
        opening_index: Index of the opening to use (None for no opening)
    
    Returns:
        result: Game result ('1-0', '0-1', '1/2-1/2')
        pgn_game: chess.pgn.Game object
        move_count: Number of moves in the game
        time_stats: Dictionary with timing statistics
    """
    board = chess.Board()
    
    # Create PGN game
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "MCTS vs MCTS Test"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["White"] = "MCTS 1"
    pgn_game.headers["Black"] = "MCTS 2"
    
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
    
    # Track timing statistics
    time_stats = {
        'mcts1_total': 0.0,
        'mcts2_total': 0.0,
        'mcts1_moves': 0,
        'mcts2_moves': 0
    }
    
    while not board.is_game_over(claim_draw=True):
        move_count += 1
        
        # Select the MCTS module based on whose turn it is
        if board.turn:  # White's turn
            current_mcts = mcts1_module
            player_name = "MCTS 1 (White)"
            time_key = 'mcts1'
        else:  # Black's turn
            current_mcts = mcts2_module
            player_name = "MCTS 2 (Black)"
            time_key = 'mcts2'
        
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
                root = current_mcts.Root(board, model)
                root.parallelRolloutsTotal(board.copy(), model, threads*rollouts, threads)
            
            elapsed = time.perf_counter() - start_time
            time_stats[f'{time_key}_total'] += elapsed
            time_stats[f'{time_key}_moves'] += 1
            
            # Get the best move
            edge = root.maxNSelect()
            best_move = edge.getMove()

            if verbose:
                Q = root.getQ()
                # Ensure Q is a scalar value
                if hasattr(Q, 'item'):
                    Q = Q.item()
                elif hasattr(Q, '__len__'):
                    Q = float(Q)
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
    
    return result, pgn_game, move_count, time_stats

def main():
    parser = argparse.ArgumentParser(description='Test two MCTS implementations against each other')
    parser.add_argument('--mcts1', required=True, help='Path to first MCTS implementation (.py) file')
    parser.add_argument('--mcts2', required=True, help='Path to second MCTS implementation (.py) file')
    parser.add_argument('--model', required=True, help='Path to model (.pt) file to use for both MCTS')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play (default: 20)')
    parser.add_argument('--rollouts', type=int, default=50, help='Number of MCTS rollouts per move (default: 50)')
    parser.add_argument('--threads', type=int, default=60, help='Number of threads for MCTS (default: 60)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed move information')
    parser.add_argument('--pgn', help='Save games to PGN file')
    parser.add_argument('--alternate', action='store_true', help='Alternate which MCTS plays white')
    parser.add_argument('--json', help='Output results in JSON format to specified file')
    parser.add_argument('--no-openings', action='store_true', help='Disable chess openings and play from starting position')
    
    args = parser.parse_args()
    
    print(f"Loading MCTS implementations...")
    print(f"MCTS 1: {args.mcts1}")
    print(f"MCTS 2: {args.mcts2}")
    
    # Load both MCTS modules
    mcts1_module = load_mcts_module(args.mcts1)
    mcts2_module = load_mcts_module(args.mcts2)
    
    # Load the model
    print(f"\nLoading model: {args.model}")
    model, device = load_model(args.model)
    print(f"Model loaded on device: {device}")
    
    print(f"\nStarting tournament: {args.games} games")
    print(f"Rollouts per move: {args.rollouts}")
    print(f"Threads per rollout: {args.threads}")
    if args.alternate:
        print("MCTS implementations will alternate playing as white")
    if not args.no_openings:
        print(f"Using {get_total_openings()} different chess openings (cycling through)")
    else:
        print("Playing from standard starting position (no openings)")
    print("-" * 60)
    
    # Track results
    results = {
        'mcts1_wins': 0,
        'mcts2_wins': 0,
        'draws': 0,
        'total_moves': 0,
        'games': [],
        'mcts1_total_time': 0.0,
        'mcts2_total_time': 0.0,
        'mcts1_total_moves': 0,
        'mcts2_total_moves': 0
    }
    
    # Play the games
    start_time = time.time()
    
    for game_num in range(args.games):
        print(f"\nGame {game_num + 1}/{args.games}")
        
        # Determine which MCTS plays white
        if args.alternate and game_num % 2 == 1:
            # Swap MCTS for odd-numbered games
            opening_idx = None if args.no_openings else game_num
            result, pgn_game, move_count, time_stats = play_game(
                mcts2_module, mcts1_module, model, device,
                args.rollouts, args.threads, args.verbose, opening_index=opening_idx
            )
            # Adjust result interpretation since MCTS are swapped
            if result == '1-0':
                results['mcts2_wins'] += 1
                winner = "MCTS 2"
            elif result == '0-1':
                results['mcts1_wins'] += 1
                winner = "MCTS 1"
            else:
                results['draws'] += 1
                winner = "Draw"
            
            # Update time stats (swapped)
            results['mcts1_total_time'] += time_stats['mcts2_total']
            results['mcts2_total_time'] += time_stats['mcts1_total']
            results['mcts1_total_moves'] += time_stats['mcts2_moves']
            results['mcts2_total_moves'] += time_stats['mcts1_moves']
            
            # Update PGN headers for swapped game
            pgn_game.headers["White"] = "MCTS 2"
            pgn_game.headers["Black"] = "MCTS 1"
        else:
            # Normal game order
            opening_idx = None if args.no_openings else game_num
            result, pgn_game, move_count, time_stats = play_game(
                mcts1_module, mcts2_module, model, device,
                args.rollouts, args.threads, args.verbose, opening_index=opening_idx
            )
            if result == '1-0':
                results['mcts1_wins'] += 1
                winner = "MCTS 1"
            elif result == '0-1':
                results['mcts2_wins'] += 1
                winner = "MCTS 2"
            else:
                results['draws'] += 1
                winner = "Draw"
            
            # Update time stats
            results['mcts1_total_time'] += time_stats['mcts1_total']
            results['mcts2_total_time'] += time_stats['mcts2_total']
            results['mcts1_total_moves'] += time_stats['mcts1_moves']
            results['mcts2_total_moves'] += time_stats['mcts2_moves']
        
        results['total_moves'] += move_count
        results['games'].append(pgn_game)
        
        print(f"Result: {result} ({winner} wins)" if winner != "Draw" else f"Result: {result}")
        print(f"Moves: {move_count}")
        
        # Print running score
        print(f"Score: MCTS 1: {results['mcts1_wins']} | MCTS 2: {results['mcts2_wins']} | Draws: {results['draws']}")
    
    total_time = time.time() - start_time
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total games: {args.games}")
    print(f"MCTS 1 wins: {results['mcts1_wins']} ({results['mcts1_wins']/args.games*100:.1f}%)")
    print(f"MCTS 2 wins: {results['mcts2_wins']} ({results['mcts2_wins']/args.games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/args.games*100:.1f}%)")
    print(f"\nAverage game length: {results['total_moves']/args.games:.1f} moves")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per game: {total_time/args.games:.1f} seconds")
    
    # Performance comparison
    print("\n" + "-" * 60)
    print("PERFORMANCE COMPARISON")
    print("-" * 60)
    if results['mcts1_total_moves'] > 0:
        avg_time_mcts1 = results['mcts1_total_time'] / results['mcts1_total_moves']
        print(f"MCTS 1 average time per move: {avg_time_mcts1:.3f} seconds")
    if results['mcts2_total_moves'] > 0:
        avg_time_mcts2 = results['mcts2_total_time'] / results['mcts2_total_moves']
        print(f"MCTS 2 average time per move: {avg_time_mcts2:.3f} seconds")
    
    if results['mcts1_total_moves'] > 0 and results['mcts2_total_moves'] > 0:
        speed_ratio = avg_time_mcts1 / avg_time_mcts2
        if speed_ratio > 1:
            print(f"MCTS 2 is {speed_ratio:.2f}x faster than MCTS 1")
        else:
            print(f"MCTS 1 is {1/speed_ratio:.2f}x faster than MCTS 2")
    
    # Determine overall winner
    print("\n" + "=" * 60)
    if results['mcts1_wins'] > results['mcts2_wins']:
        print(f"MCTS 1 wins the match! (+{results['mcts1_wins'] - results['mcts2_wins']})")
    elif results['mcts2_wins'] > results['mcts1_wins']:
        print(f"MCTS 2 wins the match! (+{results['mcts2_wins'] - results['mcts1_wins']})")
    else:
        print(f"The match is a draw!")
    
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
            'mcts1': args.mcts1,
            'mcts2': args.mcts2,
            'model': args.model,
            'games': args.games,
            'mcts1_wins': results['mcts1_wins'],
            'mcts2_wins': results['mcts2_wins'],
            'draws': results['draws'],
            'mcts1_win_rate': results['mcts1_wins'] / args.games,
            'mcts2_win_rate': results['mcts2_wins'] / args.games,
            'draw_rate': results['draws'] / args.games,
            'average_game_length': results['total_moves'] / args.games,
            'total_time': total_time,
            'time_per_game': total_time / args.games,
            'mcts1_avg_time_per_move': results['mcts1_total_time'] / results['mcts1_total_moves'] if results['mcts1_total_moves'] > 0 else 0,
            'mcts2_avg_time_per_move': results['mcts2_total_time'] / results['mcts2_total_moves'] if results['mcts2_total_moves'] > 0 else 0,
            'winner': 'mcts1' if results['mcts1_wins'] > results['mcts2_wins'] else ('mcts2' if results['mcts2_wins'] > results['mcts1_wins'] else 'draw')
        }
        import json
        with open(args.json, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results saved to {args.json}")

if __name__ == '__main__':
    main()