#!/usr/bin/env python3
"""
Neural Network Benchmark Script

This script benchmarks the raw neural network evaluation performance
without MCTS overhead to establish a theoretical maximum nodes per second.
"""

import argparse
import time
import chess
import torch
import numpy as np
import random
from AlphaZeroNetwork import AlphaZeroNet
from encoder import encodePositionForInference, callNeuralNetworkBatched, callNeuralNetworkBatchedMP
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device


from multiprocessing import Pool, set_start_method

# Set start method for multiprocessing
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

def generate_diverse_positions(num_positions=10000):
    """
    Generate diverse chess positions by playing random games from various openings.
    
    Args:
        num_positions: Number of positions to generate
        
    Returns:
        List of chess.Board objects
    """
    positions = []
    
    # Common chess openings to ensure diversity
    openings = [
        [],  # Starting position
        ["e2e4"],  # King's pawn
        ["d2d4"],  # Queen's pawn
        ["g1f3"],  # Nf3
        ["c2c4"],  # English
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Ruy Lopez
        ["e2e4", "c7c5"],  # Sicilian
        ["d2d4", "g8f6", "c2c4"],  # Indian defense
        ["e2e4", "e7e6"],  # French defense
        ["e2e4", "c7c6"],  # Caro-Kann
    ]
    
    print(f"Generating {num_positions} diverse chess positions...")
    
    positions_per_opening = num_positions // len(openings)
    
    for opening_moves in openings:
        for _ in range(positions_per_opening):
            board = chess.Board()
            
            # Apply opening moves
            for move in opening_moves:
                try:
                    board.push(chess.Move.from_uci(move))
                except:
                    break
            
            # Play random moves to create diverse positions
            num_random_moves = random.randint(0, 40)  # Vary game phase
            
            for _ in range(num_random_moves):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                    
                # Sometimes prefer captures to create imbalanced positions
                if random.random() < 0.3:
                    capture_moves = [m for m in legal_moves if board.is_capture(m)]
                    if capture_moves:
                        move = random.choice(capture_moves)
                    else:
                        move = random.choice(legal_moves)
                else:
                    move = random.choice(legal_moves)
                
                board.push(move)
                
                # Stop if game is over
                if board.is_game_over():
                    break
            
            positions.append(board.copy())
            
            if len(positions) >= num_positions:
                break
                
        if len(positions) >= num_positions:
            break
    
    # Fill remaining positions if needed
    while len(positions) < num_positions:
        board = chess.Board()
        num_moves = random.randint(0, 50)
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
            
        positions.append(board)
    
    return positions[:num_positions]


def worker_init():
    """Initialize worker process for multiprocessing."""
    # Each worker needs its own device context
    import torch
    torch.set_num_threads(1)  # Prevent thread oversubscription


def benchmark_neural_network(model_path, num_positions=10000, batch_size=None):
    """
    Benchmark neural network evaluation performance.
    
    Args:
        model_path: Path to the neural network model
        num_positions: Number of positions to evaluate
        batch_size: Batch size for evaluation (None for auto-detection)
    """
    # Get optimal device
    device, device_str = get_optimal_device()
    print(f"Using device: {device_str}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model configuration
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        num_blocks = checkpoint['model_config']['num_blocks']
        num_filters = checkpoint['model_config']['num_filters']
    else:
        # Try to infer from filename or use default
        if '20x256' in model_path:
            num_blocks, num_filters = 20, 256
        elif '10x128' in model_path:
            num_blocks, num_filters = 10, 128
        else:
            # Default configuration
            num_blocks = 20
            num_filters = 256
    
    # Create model
    model = AlphaZeroNet(num_blocks, num_filters)
    
    # Load weights - handle both checkpoint and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint is directly the state_dict
        model.load_state_dict(checkpoint)
    
    model = optimize_for_device(model, device)
    model.eval()
    
    # Determine batch size
    if batch_size is None:
        batch_size = get_batch_size_for_device(base_batch_size=256)
    print(f"Using batch size: {batch_size}")
    
    # Generate positions
    positions = generate_diverse_positions(num_positions)
    print(f"Generated {len(positions)} positions")
    
    # Pre-encode positions (not included in timing)
    print("Pre-encoding positions...")
    encoded_positions = []
    masks = []
    
    with torch.no_grad():
        for board in positions:
            position, mask = encodePositionForInference(board)
            encoded_positions.append(position)
            masks.append(mask)
    
    # Convert to tensors
    encoded_positions = np.array(encoded_positions)
    masks = np.array(masks)
    
    # Warm-up run
    print("Performing warm-up run...")
    with torch.no_grad():
        _ = callNeuralNetworkBatched(positions[:batch_size], model)
    
    # Benchmark
    print(f"\nStarting benchmark of {num_positions} positions...")
    
    start_time = time.perf_counter()
    
    num_batches = (num_positions + batch_size - 1) // batch_size
    total_evaluations = 0
    
    with torch.no_grad():
        # For multiprocessing with MPS, we need to move model to CPU.
        # Hence do not test the performance with multiprocessing on MPS devices, since it'll be radically slower.
        if 'mps' in str(device).lower():
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_positions)
                batch_positions = positions[batch_start:batch_end]
                values, policies = callNeuralNetworkBatched(batch_positions, model)
                total_evaluations += len(batch_positions)
        else:
            chunks = []
            num_processes = min(10, num_batches)  # Don't create more processes than batches
            for i in range(num_processes):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_positions)
                batch_positions = positions[batch_start:batch_end]
                chunks.append((batch_positions, model))

            with Pool(num_processes) as pool:
                results = pool.map(callNeuralNetworkBatchedMP, chunks)
            
            # Calculate total evaluations from results
            total_evaluations = sum(len(result[0]) for result in results)

    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    evaluations_per_second = total_evaluations / elapsed_time
    avg_time_per_eval = elapsed_time / total_evaluations * 1000  # in milliseconds
    
    # Report results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Device: {device_str}")
    print(f"Model: {num_blocks} blocks x {num_filters} filters")
    print(f"Batch size: {batch_size}")
    print(f"Total positions evaluated: {total_evaluations}")
    print(f"Total time: {elapsed_time:.3f} seconds")
    print(f"Evaluations per second: {evaluations_per_second:.1f}")
    print(f"Average time per evaluation: {avg_time_per_eval:.3f} ms")
    print(f"Theoretical maximum nodes/second: {evaluations_per_second:.0f}")
    print("="*60)
    
    # Additional timing breakdown
    print("\nTiming breakdown (estimated):")
    print(f"  - Per batch: {elapsed_time / num_batches * 1000:.1f} ms")
    print(f"  - Batch throughput: {batch_size / (elapsed_time / num_batches):.1f} positions/sec")
    
    return evaluations_per_second


def main():
    parser = argparse.ArgumentParser(description='Benchmark neural network evaluation performance')
    parser.add_argument('--model', type=str, default='weights/AlphaZeroNet_20x256.pt',
                        help='Path to the neural network model')
    parser.add_argument('--positions', type=int, default=10000,
                        help='Number of positions to evaluate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for evaluation (auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for position generation')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run benchmark
    benchmark_neural_network(args.model, args.positions, args.batch_size)


if __name__ == '__main__':
    main()