import argparse
import os
import chess
import chess.pgn
import torch
import AlphaZeroNetwork
import time
from device_utils import get_optimal_device, optimize_for_device, get_gpu_count
from RLDataset import SelfPlayDataCollector
import numpy as np
import encoder
from MCTS.MCTS_async import MCTSEngine, AsyncRoot
from async_neural_net_server import NeuralNetworkPool
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


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


def play_single_game(game_id, mcts_engine, num_rollouts, temperature, verbose=False):
    """
    Play a single self-play game using the async MCTS engine.
    
    Args:
        game_id: Unique identifier for this game
        mcts_engine: The MCTSEngine instance
        num_rollouts: Number of rollouts per move
        temperature: Temperature for move selection
        verbose: Whether to print game progress
        
    Returns:
        Game data for training
    """
    board = chess.Board()
    game_data = []
    move_count = 0
    
    if verbose:
        print(f"Starting game {game_id}")
    
    while not board.is_game_over():
        # Get root node with MCTS search
        root = mcts_engine.search(board, num_rollouts, return_root=True)
        
        # Get visit counts for training
        visit_counts = root.getVisitCounts(board)
        
        # Apply temperature to visit counts for move selection
        if temperature > 0:
            visit_probs = visit_counts / (np.sum(visit_counts) + 1e-8)
            visit_probs = np.power(visit_probs, 1.0 / temperature)
            visit_probs = visit_probs / np.sum(visit_probs)
        else:
            # Temperature 0 means deterministic (argmax)
            visit_probs = np.zeros_like(visit_counts)
            visit_probs[np.argmax(visit_counts)] = 1.0
        
        # Store position for training
        game_data.append({
            'board': board.copy(),
            'visit_counts': visit_counts,
            'move_probs': visit_probs
        })
        
        # Select move based on visit probabilities
        legal_moves = list(board.legal_moves)
        move_indices = []
        move_probs = []
        
        for move in legal_moves:
            if not board.turn:
                mirrored_move = encoder.mirrorMove(move)
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(mirrored_move)
            else:
                planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
            
            moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
            if visit_probs[moveIdx] > 0:
                move_indices.append(len(move_indices))
                move_probs.append(visit_probs[moveIdx])
        
        if move_probs:
            move_probs = np.array(move_probs)
            move_probs = move_probs / np.sum(move_probs)
            selected_idx = np.random.choice(move_indices, p=move_probs)
            selected_move = legal_moves[selected_idx]
        else:
            # Fallback to best move by visit count
            best_edge = root.maxNSelect()
            selected_move = best_edge.getMove() if best_edge else legal_moves[0]
        
        board.push(selected_move)
        move_count += 1
        
        # Clean up root
        root.cleanup()
        
        if verbose and move_count % 10 == 0:
            print(f"Game {game_id}: {move_count} moves played")
    
    # Get game result
    result = board.result()
    winner = encoder.parseResult(result)
    
    if verbose:
        print(f"Game {game_id} finished: {result} after {move_count} moves")
    
    # Add winner information to all positions
    for data in game_data:
        data['winner'] = winner
    
    return game_data


def main(modelFile, mode, color, num_rollouts, num_threads, fen, verbose, 
         output_directory, offset, file_base, games_to_play, gpu_ids=None, 
         save_format='pgn', temperature=1.0, iteration=0, batch_size=256,
         games_per_worker=10):
    """
    Generate self-play training games using async MCTS engine.
    
    Args:
        modelFile: Path to model file
        num_rollouts: Rollouts per move
        num_threads: Worker threads per game
        output_directory: Where to save games
        games_to_play: Number of games to generate
        batch_size: Neural network batch size
        games_per_worker: Games to play in parallel
    """
    
    def get_fileName(games_played, offset, file_base, extension='pgn'):
        return f'{output_directory}/{file_base}_{games_played + offset}.{extension}'
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
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
    print(f"Initializing async MCTS engine with {num_threads} workers per game")
    print(f"Max batch size: {batch_size}")
    
    # For multi-GPU setups, use NeuralNetworkPool
    if gpu_ids and len(gpu_ids) > 1:
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        nn_pool = NeuralNetworkPool(
            model, 
            num_gpus=len(gpu_ids),
            max_batch_size=batch_size,
            verbose=verbose
        )
        nn_pool.start()
        # Create multiple MCTS engines, one per GPU
        mcts_engines = []
        for server in nn_pool.servers:
            engine = MCTSEngine(
                model,
                device=server.device,
                max_batch_size=batch_size,
                num_workers=num_threads,
                verbose=False
            )
            engine.nn_server = server  # Use existing server
            mcts_engines.append(engine)
    else:
        # Single GPU/device setup
        mcts_engine = MCTSEngine(
            model,
            device=device,
            max_batch_size=batch_size,
            num_workers=num_threads,
            verbose=verbose
        )
        mcts_engine.start()
        mcts_engines = [mcts_engine]
    
    # Play games in parallel
    games_played = 0
    total_positions = 0
    start_time = time.time()
    
    if save_format == 'npz':
        # Initialize data collector for NPZ format
        collector = SelfPlayDataCollector(output_directory, file_base, iteration)
    
    # Use process pool for game generation
    with ThreadPoolExecutor(max_workers=games_per_worker) as executor:
        while games_played < games_to_play:
            # Submit batch of games
            futures = []
            games_in_batch = min(games_per_worker, games_to_play - games_played)
            
            for i in range(games_in_batch):
                game_id = games_played + i + offset
                # Round-robin engine selection for multi-GPU
                engine = mcts_engines[i % len(mcts_engines)]
                
                future = executor.submit(
                    play_single_game,
                    game_id,
                    engine,
                    num_rollouts,
                    temperature,
                    verbose and i == 0  # Only verbose for first game
                )
                futures.append((future, game_id))
            
            # Collect completed games
            for future, game_id in futures:
                try:
                    game_data = future.result()
                    
                    if save_format == 'pgn':
                        # Save as PGN
                        pgn_file = get_fileName(game_id, 0, file_base, 'pgn')
                        game = chess.pgn.Game()
                        node = game
                        
                        for position_data in game_data:
                            board = position_data['board']
                            if board.move_stack:
                                node = node.add_variation(board.peek())
                        
                        game.headers["Result"] = board.result()
                        game.headers["Event"] = "Self-play training game"
                        game.headers["Round"] = str(game_id)
                        
                        with open(pgn_file, 'w') as f:
                            print(game, file=f)
                    
                    elif save_format == 'npz':
                        # Add to collector
                        for position_data in game_data:
                            collector.add_position(
                                position_data['board'],
                                position_data['visit_counts'],
                                position_data['winner']
                            )
                    
                    total_positions += len(game_data)
                    games_played += 1
                    
                    if games_played % 10 == 0:
                        elapsed = time.time() - start_time
                        games_per_sec = games_played / elapsed
                        positions_per_sec = total_positions / elapsed
                        print(f"Progress: {games_played}/{games_to_play} games "
                              f"({games_per_sec:.2f} games/sec, "
                              f"{positions_per_sec:.1f} positions/sec)")
                        
                except Exception as e:
                    print(f"Error in game {game_id}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save final NPZ file if using that format
    if save_format == 'npz' and total_positions > 0:
        collector.save()
        print(f"Saved {total_positions} positions to {collector.current_file}")
    
    # Cleanup
    for engine in mcts_engines:
        if hasattr(engine, 'stop'):
            engine.stop()
    
    if 'nn_pool' in locals():
        nn_pool.stop()
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\nCompleted {games_played} games with {total_positions} positions")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Games per second: {games_played/elapsed:.2f}")
    print(f"Positions per second: {total_positions/elapsed:.1f}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Generate self-play training games using async MCTS engine'
    )
    parser.add_argument('--model', help='Path to model (.pt) file.', required=True)
    parser.add_argument('--rollouts', type=int, help='Number of rollouts.', default=800)
    parser.add_argument('--threads', type=int, help='Number of threads per game.', default=32)
    parser.add_argument('--output', type=str, help='Output directory.', required=True)
    parser.add_argument('--offset', type=int, help='Start counting at the given value, and add to game number to ensure unique .pgn file names.', default=0)
    parser.add_argument('--file_base', type=str, help='Base name for output files', default='selfplay')
    parser.add_argument('--games', type=int, help='Number of games to play', required=True)
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--temperature', type=float, help='Temperature for move selection', default=1.0)
    parser.add_argument('--format', choices=['pgn', 'npz'], default='pgn', help='Output format')
    parser.add_argument('--iteration', type=int, default=0, help='Training iteration number')
    parser.add_argument('--batch-size', type=int, default=256, help='Neural network batch size')
    parser.add_argument('--parallel-games', type=int, default=10, help='Number of games to play in parallel')
    parser.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to use (e.g., --gpus 0 1 2)')
    
    # Unused arguments for compatibility
    parser.add_argument('--mode', type=str, help='Play mode. Ignored.', default='s')
    parser.add_argument('--color', type=str, help='Color to play. Ignored.', default='w')
    parser.add_argument('--fen', type=str, help='Starting position. Ignored.', default=None)
    
    args = parser.parse_args()
    
    main(
        modelFile=args.model,
        mode=args.mode,
        color=args.color,
        num_rollouts=args.rollouts,
        num_threads=args.threads,
        fen=args.fen,
        verbose=args.verbose,
        output_directory=args.output,
        offset=args.offset,
        file_base=args.file_base,
        games_to_play=args.games,
        gpu_ids=args.gpus,
        save_format=args.format,
        temperature=args.temperature,
        iteration=args.iteration,
        batch_size=args.batch_size,
        games_per_worker=args.parallel_games
    )