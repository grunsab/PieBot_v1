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
import multiprocessing as mp
from functools import partial
from AlphaZeroNetwork import AlphaZeroNet
import PieBotNetwork
import PieNanoNetwork
import PieNanoNetwork_v2
import TitanMiniNetwork
from encoder import encodePositionForInference, callNeuralNetworkBatched, decodePolicyOutput
from device_utils import get_optimal_device, optimize_for_device, get_batch_size_for_device
from model_utils import detect_model_type, create_pienano_from_weights, create_titanmini_from_weights, clean_state_dict
import sys

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


def encode_positions_worker(positions_chunk):
    """
    Worker function to encode a chunk of positions in parallel.
    
    Args:
        positions_chunk: List of chess.Board positions to encode
        
    Returns:
        Tuple of (encoded_positions, masks) as numpy arrays
    """
    encoded = []
    masks = []
    
    for board in positions_chunk:
        position, mask = encodePositionForInference(board)
        encoded.append(position)
        masks.append(mask)
    
    return np.array(encoded), np.array(masks)


def decode_policies_worker(args):
    """
    Worker function to decode policy outputs in parallel.
    
    Args:
        args: Tuple of (boards_chunk, policies_chunk)
        
    Returns:
        Array of move probabilities
    """
    boards_chunk, policies_chunk = args
    move_probs = []
    
    for board, policy in zip(boards_chunk, policies_chunk):
        move_prob = decodePolicyOutput(board, policy)
        # Pad to 200 to match expected output shape
        padded = np.zeros(200, dtype=np.float32)
        padded[:move_prob.shape[0]] = move_prob
        move_probs.append(padded)
    
    return np.array(move_probs)


def benchmark_neural_network(model_path, num_positions=10000, batch_size=None, skip_compile=False, num_processes=1):
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model type using model_utils
    model_type = detect_model_type(checkpoint, model_path)
    
    # Create appropriate model
    if model_type == 'PieBotNet':
        model = PieBotNetwork.PieBotNet()
        print(f"Detected PieBotNet model")
        num_blocks = "N/A"
        num_filters = "N/A"
    elif model_type == 'TitanMini':
        model = create_titanmini_from_weights(checkpoint)
        print(f"Detected TitanMini model")
        # Extract architecture info for display
        num_blocks = model.num_layers if hasattr(model, 'num_layers') else "N/A"
        num_filters = model.d_model if hasattr(model, 'd_model') else "N/A"
    elif model_type == 'PieNanoV2' or model_type == 'PieNano':
        # Use model_utils to create the model with correct architecture
        model = create_pienano_from_weights(checkpoint)
        print(f"Detected {model_type} model")
        # Extract architecture info from the created model
        if hasattr(model, 'residual_tower'):
            num_blocks = len(model.residual_tower)
        else:
            num_blocks = "N/A"
        
        if hasattr(model, 'conv_block') and len(model.conv_block) > 0:
            # Get num_filters from the first conv layer
            conv_layer = model.conv_block[0]
            if hasattr(conv_layer, 'out_channels'):
                num_filters = conv_layer.out_channels
            else:
                num_filters = "N/A"
        else:
            num_filters = "N/A"
    else:
        # Extract model configuration for AlphaZeroNet
        if 'model_config' in checkpoint:
            num_blocks = checkpoint['model_config']['num_blocks']
            num_filters = checkpoint['model_config']['num_filters']
        else:
            # Default configuration
            num_blocks = 20
            num_filters = 256
        model = AlphaZeroNet(num_blocks, num_filters)
        print(f"Detected AlphaZeroNet model")
    # Load model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Clean state dict (remove _orig_mod prefix etc.)
    state_dict = clean_state_dict(state_dict)
    
    # Load the cleaned state dict
    model.load_state_dict(state_dict)
    
    # Handle FP16 models
    if isinstance(checkpoint, dict) and checkpoint.get('model_type') == 'fp16':
        model = model.half()
        print(f"Loaded FP16 model on {device_str}")

    model = optimize_for_device(model, device)
    model.eval()
    # Basically if we're using Linux with a CUDA device
    if hasattr(torch, 'compile') and sys.platform != 'win32' and device.type != "mps" and not skip_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for additional speedup")
        except:
            print("torch.compile not available or failed, using eager mode")
    elif sys.platform == 'win32':
        print("Skipping torch.compile on Windows (Triton not fully supported)")

    # Determine batch size
    if batch_size is None:
        batch_size = get_batch_size_for_device(base_batch_size=1024)
    print(f"Using batch size: {batch_size}")
    
    # Generate positions
    positions = generate_diverse_positions(num_positions)
    print(f"Generated {len(positions)} positions")
    
    # Pre-encode positions (not included in timing)
    print(f"Pre-encoding positions using {num_processes} process(es)...")
    
    if num_processes > 1:
        # Split positions into chunks for parallel processing
        chunk_size = len(positions) // num_processes
        position_chunks = []
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processes - 1 else len(positions)
            position_chunks.append(positions[start_idx:end_idx])
        
        # Encode positions in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(encode_positions_worker, position_chunks)
        
        # Combine results
        encoded_positions = np.vstack([r[0] for r in results])
        masks = np.vstack([r[1] for r in results])
    else:
        # Single process encoding (original behavior)
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
    print(f"Using {num_processes} process(es) for CPU operations")
    
    if num_processes > 1:
        # Multi-process benchmark with custom batching
        start_time = time.perf_counter()
        
        num_batches = (num_positions + batch_size - 1) // batch_size
        total_evaluations = 0
        
        # Setup multiprocessing pool
        pool = mp.Pool(processes=num_processes)
        
        with torch.no_grad():
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_positions)
                batch_positions = positions[batch_start:batch_end]
                batch_encoded = encoded_positions[batch_start:batch_end]
                batch_masks = masks[batch_start:batch_end]
                
                # Convert to tensors
                inputs = torch.from_numpy(batch_encoded).float()
                masks_tensor = torch.from_numpy(batch_masks).float()
                
                # Move to device
                model_device = next(model.parameters()).device
                inputs = inputs.to(model_device)
                masks_tensor = masks_tensor.to(model_device)
                
                # Convert to half precision if model is FP16
                if next(model.parameters()).dtype == torch.float16:
                    inputs = inputs.half()
                    masks_tensor = masks_tensor.half()
                
                # Flatten masks
                masks_flat = masks_tensor.view(masks_tensor.shape[0], -1)
                
                # GPU inference
                value, policy = model(inputs, policyMask=masks_flat)
                
                # Move results back to CPU
                values_cpu = value.cpu().numpy().reshape(-1)
                policies_cpu = policy.cpu().numpy()
                
                # Parallel policy decoding
                chunk_size = len(batch_positions) // num_processes
                decode_args = []
                for j in range(num_processes):
                    start_idx = j * chunk_size
                    end_idx = start_idx + chunk_size if j < num_processes - 1 else len(batch_positions)
                    decode_args.append((
                        batch_positions[start_idx:end_idx],
                        policies_cpu[start_idx:end_idx]
                    ))
                
                # Decode policies in parallel
                if len(batch_positions) >= num_processes:
                    decoded_results = pool.map(decode_policies_worker, decode_args)
                    move_probabilities = np.vstack(decoded_results)
                else:
                    # For small batches, use single process
                    move_probabilities = decode_policies_worker((batch_positions, policies_cpu))
                
                total_evaluations += len(batch_positions)
        
        pool.close()
        pool.join()
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    else:
        # Single process benchmark (original behavior)
        start_time = time.perf_counter()
        
        num_batches = (num_positions + batch_size - 1) // batch_size
        total_evaluations = 0
        
        with torch.no_grad():
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_positions)
                batch_positions = positions[batch_start:batch_end]
                
                # Evaluate batch
                values, policies = callNeuralNetworkBatched(batch_positions, model)
                total_evaluations += len(batch_positions)
        
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
    if isinstance(num_blocks, int) and isinstance(num_filters, int):
        print(f"Model: {num_blocks} blocks x {num_filters} filters")
    else:
        print(f"Model: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"CPU processes: {num_processes}")
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
    parser.add_argument("--skip-compile", type=bool, default=False,
                        help="Should we skip the compilation of the neural network")
    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of processes for parallel encoding/decoding (default: 1)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run benchmark
    benchmark_neural_network(args.model, args.positions, args.batch_size, args.skip_compile, args.num_processes)


if __name__ == '__main__':
    main()