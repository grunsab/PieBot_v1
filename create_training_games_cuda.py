#!/usr/bin/env python3
"""
Generate self-play training games - Windows/CUDA Optimized Version

High-performance training game generation optimized for Windows systems
with NVIDIA GPUs (RTX 4080 and similar). Capable of generating thousands
of games per hour with proper hardware.
"""

import argparse
import os
import sys
import chess
import chess.pgn
import torch
import AlphaZeroNetwork
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import encoder
from MCTS_cuda import MCTSEngineCUDA, AsyncRootCUDA
from async_neural_net_server_cuda import NeuralNetworkPoolCUDA
from RLDataset import SelfPlayDataCollector
import queue
import threading

# Windows-specific optimizations
if sys.platform == 'win32':
    # Prevent Windows from sleeping during long runs
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


class GameGenerator:
    """Manages game generation with multiple workers."""
    
    def __init__(self, model_path, device_id, num_rollouts, batch_size, temperature, verbose):
        self.model_path = model_path
        self.device_id = device_id
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size
        self.temperature = temperature
        self.verbose = verbose
        self.mcts_engine = None
        
    def initialize(self):
        """Initialize the MCTS engine for this worker."""
        if self.mcts_engine is not None:
            return
            
        # Load model
        device = torch.device(f'cuda:{self.device_id}')
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        weights = torch.load(self.model_path, map_location=device)
        model.load_state_dict(weights)
        model.eval()
        
        # Create MCTS engine
        self.mcts_engine = MCTSEngineCUDA(
            model,
            device=device,
            max_batch_size=self.batch_size,
            num_workers=32,  # Fewer workers per game for better concurrency
            verbose=False
        )
        self.mcts_engine.start()
        
    def generate_game(self, game_id):
        """Generate a single self-play game."""
        # Ensure engine is initialized
        self.initialize()
        
        board = chess.Board()
        game_data = []
        move_count = 0
        
        if self.verbose and game_id % 10 == 0:
            print(f"Starting game {game_id}")
        
        start_time = time.time()
        
        while not board.is_game_over() and move_count < 300:  # Cap at 300 moves
            # Get root node with MCTS search
            root = self.mcts_engine.search(board, self.num_rollouts, return_root=True)
            
            # Get visit counts for training
            visit_counts = root.getVisitCounts(board)
            
            # Apply temperature to visit counts for move selection
            if self.temperature > 0 and move_count < 30:  # Apply temperature for first 30 moves
                visit_probs = visit_counts / (np.sum(visit_counts) + 1e-8)
                visit_probs = np.power(visit_probs, 1.0 / self.temperature)
                visit_probs = visit_probs / np.sum(visit_probs)
            else:
                # Temperature 0 after move 30
                visit_probs = np.zeros_like(visit_counts)
                if np.max(visit_counts) > 0:
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
            
            for i, move in enumerate(legal_moves):
                if not board.turn:
                    mirrored_move = encoder.mirrorMove(move)
                    planeIdx, rankIdx, fileIdx = encoder.moveToIdx(mirrored_move)
                else:
                    planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
                
                moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
                if visit_probs[moveIdx] > 0:
                    move_indices.append(i)
                    move_probs.append(visit_probs[moveIdx])
            
            if move_probs:
                move_probs = np.array(move_probs)
                move_probs = move_probs / np.sum(move_probs)
                selected_idx = np.random.choice(len(move_indices), p=move_probs)
                selected_move = legal_moves[move_indices[selected_idx]]
            else:
                # Fallback to best move by visit count
                best_edge = root.maxNSelect()
                selected_move = best_edge.getMove() if best_edge else legal_moves[0]
            
            board.push(selected_move)
            move_count += 1
            
            # Clean up root
            root.cleanup()
        
        # Get game result
        result = board.result()
        if board.is_game_over():
            winner = encoder.parseResult(result)
        else:
            winner = 0  # Draw if we hit move limit
            result = "1/2-1/2"
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"Game {game_id} finished: {result} after {move_count} moves ({elapsed:.1f}s)")
        
        # Add winner information to all positions
        for data in game_data:
            data['winner'] = winner
        
        return game_data, result, move_count
    
    def cleanup(self):
        """Clean up resources."""
        if self.mcts_engine:
            self.mcts_engine.stop()


def worker_process(worker_args):
    """Worker process for game generation."""
    worker_id, games_to_generate, model_path, device_id, num_rollouts, batch_size, temperature, verbose, offset = worker_args
    
    # Set process affinity on Windows for better performance
    if sys.platform == 'win32':
        try:
            import psutil
            p = psutil.Process()
            # Distribute processes across CPU cores
            cpu_count = psutil.cpu_count()
            affinity_mask = list(range(worker_id % cpu_count, cpu_count, max(1, cpu_count // 4)))
            p.cpu_affinity(affinity_mask[:4])  # Use up to 4 cores per worker
        except:
            pass
    
    generator = GameGenerator(model_path, device_id, num_rollouts, batch_size, temperature, verbose)
    results = []
    
    try:
        for i in range(games_to_generate):
            game_id = offset + worker_id * games_to_generate + i
            game_data, result, move_count = generator.generate_game(game_id)
            results.append((game_id, game_data, result, move_count))
    finally:
        generator.cleanup()
    
    return results


def main(modelFile, num_rollouts, num_threads, output_directory, offset, file_base, 
         games_to_play, temperature=1.0, iteration=0, batch_size=512, 
         num_workers=4, device_ids=None, save_format='npz'):
    """
    Generate self-play training games using CUDA-optimized engine.
    
    Args:
        modelFile: Path to model file
        num_rollouts: Rollouts per move
        num_threads: Worker threads per MCTS instance
        output_directory: Where to save games
        games_to_play: Number of games to generate
        batch_size: Neural network batch size
        num_workers: Number of parallel game generation processes
        device_ids: List of CUDA device IDs to use
    """
    
    print("="*60)
    print("Self-Play Game Generation - CUDA Optimized")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This version requires an NVIDIA GPU.")
        return
        
    # Auto-detect devices if not specified
    if device_ids is None:
        device_count = torch.cuda.device_count()
        device_ids = list(range(device_count))
        
    print(f"Using CUDA devices: {device_ids}")
    for device_id in device_ids:
        name = torch.cuda.get_device_name(device_id)
        memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        print(f"  Device {device_id}: {name} ({memory:.1f}GB)")
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Adjust worker count based on available GPUs
    if num_workers > len(device_ids) * 2:
        num_workers = len(device_ids) * 2
        print(f"Adjusted worker count to {num_workers} (2 per GPU)")
    
    print(f"\nGenerating {games_to_play} games:")
    print(f"  Rollouts per move: {num_rollouts}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Output format: {save_format}")
    
    # Initialize data collector for NPZ format
    if save_format == 'npz':
        collector = SelfPlayDataCollector(output_directory, file_base, iteration)
    
    # Statistics
    total_positions = 0
    total_moves = 0
    start_time = time.time()
    
    # Prepare worker arguments
    games_per_worker = games_to_play // num_workers
    remainder = games_to_play % num_workers
    
    worker_args_list = []
    for i in range(num_workers):
        games = games_per_worker
        if i < remainder:
            games += 1
            
        # Assign GPU in round-robin fashion
        device_id = device_ids[i % len(device_ids)]
        
        args = (i, games, modelFile, device_id, num_rollouts, batch_size, 
                temperature, i == 0, offset)
        worker_args_list.append(args)
    
    # Generate games in parallel
    print("\nStarting game generation...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(worker_process, args): i 
                  for i, args in enumerate(worker_args_list)}
        
        games_completed = 0
        
        # Process results as they complete
        for future in as_completed(futures):
            worker_id = futures[future]
            
            try:
                results = future.result()
                
                for game_id, game_data, result, move_count in results:
                    if save_format == 'pgn':
                        # Save as PGN
                        pgn_file = f"{output_directory}/{file_base}_{game_id}.pgn"
                        game = chess.pgn.Game()
                        node = game
                        
                        for position_data in game_data:
                            board = position_data['board']
                            if board.move_stack:
                                node = node.add_variation(board.peek())
                        
                        game.headers["Event"] = "Self-play training game"
                        game.headers["Site"] = "CUDA Engine"
                        game.headers["Result"] = result
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
                    total_moves += move_count
                    games_completed += 1
                    
                    if games_completed % 10 == 0:
                        elapsed = time.time() - start_time
                        games_per_hour = games_completed / elapsed * 3600
                        positions_per_sec = total_positions / elapsed
                        avg_game_length = total_moves / games_completed
                        
                        print(f"Progress: {games_completed}/{games_to_play} games "
                              f"({games_per_hour:.0f} games/hour, "
                              f"{positions_per_sec:.0f} pos/sec, "
                              f"avg length: {avg_game_length:.1f} moves)")
                        
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save final NPZ file if using that format
    if save_format == 'npz' and total_positions > 0:
        collector.save()
        print(f"\nSaved {total_positions} positions to {collector.current_file}")
    
    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Total games: {games_completed}")
    print(f"Total positions: {total_positions}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Games per hour: {games_completed/elapsed*3600:.0f}")
    print(f"Positions per second: {total_positions/elapsed:.0f}")
    print(f"Average game length: {total_moves/games_completed:.1f} moves")


if __name__ == '__main__':
    # Multiprocessing setup for Windows
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description='Generate self-play training games - CUDA optimized for Windows'
    )
    parser.add_argument('--model', help='Path to model (.pt) file', required=True)
    parser.add_argument('--rollouts', type=int, help='Number of rollouts per move', 
                        default=1600)
    parser.add_argument('--threads', type=int, help='Threads per MCTS instance', 
                        default=32)
    parser.add_argument('--output', type=str, help='Output directory', required=True)
    parser.add_argument('--offset', type=int, help='Starting game number', default=0)
    parser.add_argument('--file-base', type=str, help='Base name for output files', 
                        default='selfplay')
    parser.add_argument('--games', type=int, help='Number of games to generate', 
                        required=True)
    parser.add_argument('--temperature', type=float, help='Temperature for move selection', 
                        default=1.0)
    parser.add_argument('--format', choices=['pgn', 'npz'], default='npz', 
                        help='Output format')
    parser.add_argument('--iteration', type=int, default=0, 
                        help='Training iteration number')
    parser.add_argument('--batch-size', type=int, default=512, 
                        help='Neural network batch size')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel workers')
    parser.add_argument('--gpus', type=int, nargs='+', 
                        help='GPU IDs to use (e.g., --gpus 0 1)')
    
    args = parser.parse_args()
    
    main(
        modelFile=args.model,
        num_rollouts=args.rollouts,
        num_threads=args.threads,
        output_directory=args.output,
        offset=args.offset,
        file_base=args.file_base,
        games_to_play=args.games,
        temperature=args.temperature,
        iteration=args.iteration,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_ids=args.gpus,
        save_format=args.format
    )