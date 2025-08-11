"""
Quantization utilities for AlphaZero neural network models.

This module provides functions to quantize AlphaZero models for improved CPU inference performance,
reducing model size and increasing throughput while maintaining accuracy.
Optimized specifically for CNN architectures on CPU.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
from typing import List, Tuple, Optional
import encoder
from AlphaZeroNetwork import AlphaZeroNet, ConvBlock, ResidualBlock, ValueHead, PolicyHead
from device_utils import get_optimal_device
import MCTS_profiling_speedups_v2 as MCTS


class QuantizableResidualBlock(nn.Module):
    """
    Quantization-ready residual block that handles skip connections properly.
    """
    def __init__(self, orig_block):
        super().__init__()
        self.conv1 = orig_block.conv1
        self.bn1 = orig_block.bn1
        self.relu1 = orig_block.relu1
        self.conv2 = orig_block.conv2
        self.bn2 = orig_block.bn2
        self.relu2 = orig_block.relu2
        # Add skip connection for quantization
        self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Use quantized-friendly addition
        x = self.skip_add.add(x, residual)
        x = self.relu2(x)
        return x


class QuantizableAlphaZero(nn.Module):
    """
    Quantization-ready version of AlphaZeroNet with quant/dequant stubs for static quantization.
    """
    
    def __init__(self, base_model: AlphaZeroNet):
        super().__init__()
        self.quant = QuantStub()
        self.dequant_value = DeQuantStub()
        self.dequant_policy = DeQuantStub()
        
        # Copy the base model components
        self.convBlock1 = base_model.convBlock1
        
        # Replace residual blocks with quantizable versions
        self.residualBlocks = nn.ModuleList([
            QuantizableResidualBlock(block) for block in base_model.residualBlocks
        ])
        
        self.valueHead = base_model.valueHead
        self.policyHead = base_model.policyHead
        
        # Copy other attributes
        self.policy_weight = base_model.policy_weight
        
    def forward(self, x, policyMask=None):
        """
        Forward pass with quantization stubs.
        """
        # Quantize input
        x = self.quant(x)
        
        # Initial convolution block
        x = self.convBlock1(x)
        
        # Residual blocks
        for block in self.residualBlocks:
            x = block(x)
        
        # Compute heads
        value = self.valueHead(x)
        policy = self.policyHead(x)
        
        # Dequantize outputs
        value = self.dequant_value(value)
        policy = self.dequant_policy(policy)
        
        # Apply softmax and policy mask for inference (matching AlphaZeroNet behavior)
        if not self.training and policyMask is not None:
            policyMask = policyMask.view(policyMask.shape[0], -1)
            policy_exp = torch.exp(policy)
            policy_exp *= policyMask.type(torch.float32)
            policy_exp_sum = torch.sum(policy_exp, dim=1, keepdim=True)
            policy = policy_exp / policy_exp_sum
        
        # Return in correct order: value, policy (not policy, value!)
        return value, policy


def prepare_alphazero_for_quantization(model: AlphaZeroNet) -> QuantizableAlphaZero:
    """
    Prepare an AlphaZeroNet model for static quantization.
    
    Args:
        model: Original AlphaZeroNet model
        
    Returns:
        QuantizableAlphaZero ready for quantization
    """
    # Create quantizable model
    quant_model = QuantizableAlphaZero(model)
    
    # Copy all parameters
    quant_model.load_state_dict(model.state_dict(), strict=False)
    
    return quant_model


def create_calibration_dataset_selfplay(model: AlphaZeroNet, 
                                       num_positions: int = 2000,
                                       rollouts_per_move: int = 100,
                                       threads: int = 10,
                                       input_planes: int = 16) -> List[torch.Tensor]:
    """
    Create a high-quality calibration dataset from self-play games using MCTS.
    
    Args:
        model: The AlphaZeroNet model to use for self-play
        num_positions: Target number of positions to generate
        rollouts_per_move: Number of MCTS rollouts per move (default 100)
        threads: Number of threads for MCTS
        input_planes: Number of input planes (16 for AlphaZero)
        
    Returns:
        List of input tensors for calibration from realistic game positions
    """
    calibration_data = []
    model.eval()
    
    # Calculate how many games we need
    moves_per_game = 20  # Collect first 20 moves of each game
    num_games = (num_positions + moves_per_game - 1) // moves_per_game
    
    print(f"Generating {num_positions} calibration positions from {num_games} self-play games...")
    print(f"Using {rollouts_per_move} MCTS rollouts per move with {threads} threads")
    
    positions_collected = 0
    
    for game_idx in range(num_games):
        if positions_collected >= num_positions:
            break
            
        # Start a new game
        board = chess.Board()
        
        # Play moves and collect positions
        for move_idx in range(moves_per_game):
            if positions_collected >= num_positions:
                break
                
            # Check if game is over
            if board.is_game_over():
                break
            
            # Encode current position
            input_data = encoder.encodePosition(board)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            calibration_data.append(input_tensor)
            positions_collected += 1
            
            # Use MCTS to find the best move
            with torch.no_grad():
                total_simulations = threads * rollouts_per_move
                root = MCTS.Root(board, model)
                root.parallelRolloutsTotal(board.copy(), model, total_simulations, threads)
                
                # Get the best move
                edge = root.maxNSelect()
                best_move = edge.getMove()
                
                # Clean up the tree to free memory
                root.cleanup()
            
            # Make the move
            board.push(best_move)
        
        # Progress update
        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games, collected {positions_collected} positions")
    
    print(f"Generated {len(calibration_data)} high-quality calibration positions from self-play")
    return calibration_data


def create_calibration_dataset(num_positions: int = 1000, input_planes: int = 16) -> List[torch.Tensor]:
    """
    Create a calibration dataset from various chess positions.
    
    Args:
        num_positions: Number of positions to generate (default 1000)
        input_planes: Number of input planes (16 for AlphaZero)
        
    Returns:
        List of input tensors for calibration
    """
    calibration_data = []
    
    print(f"Generating {num_positions} calibration positions...")
    
    # Generate positions from different game phases
    for i in range(num_positions):
        board = chess.Board()
        
        # Vary game phase: opening (0-10), middle (10-40), endgame (40+)
        if i < num_positions // 3:
            # Opening positions
            num_moves = np.random.randint(0, 10)
        elif i < 2 * num_positions // 3:
            # Middle game positions
            num_moves = np.random.randint(10, 40)
        else:
            # Endgame positions
            num_moves = np.random.randint(40, 80)
        
        # Play random moves to get diverse positions
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            board.push(move)
        
        # Encode the position
        input_data = encoder.encodePosition(board)
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        calibration_data.append(input_tensor)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_positions} positions")
    
    return calibration_data


def calibrate_model(model: nn.Module, calibration_data: List[torch.Tensor]) -> None:
    """
    Calibrate the model for static quantization using representative data.
    
    Args:
        model: Model prepared for quantization (with observers)
        calibration_data: List of calibration samples
    """
    model.eval()
    model.cpu()  # Quantization calibration must be on CPU
    
    print("Calibrating model for static quantization...")
    
    with torch.no_grad():
        for idx, input_tensor in enumerate(calibration_data):
            # Run forward pass for calibration
            _ = model(input_tensor.cpu())
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Calibrated on {idx + 1}/{len(calibration_data)} positions")


def apply_dynamic_quantization(model: AlphaZeroNet) -> nn.Module:
    """
    Apply dynamic INT8 quantization to AlphaZeroNet model for CPU inference.
    This quantizes weights to INT8 while keeping activations in FP32.
    
    Args:
        model: Original AlphaZeroNet model
        
    Returns:
        Dynamically quantized model
    """
    # Move model to CPU (required for quantization)
    model = model.cpu()
    model.eval()
    
    # Set quantization backend to qnnpack (works on all platforms)
    torch.backends.quantized.engine = 'qnnpack'
    
    # Apply dynamic quantization to Linear and Conv2d layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("Dynamic quantization complete!")
    print("Note: Weights are INT8, activations remain FP32")
    
    return quantized_model


def apply_static_quantization(model: AlphaZeroNet,
                            calibration_data: Optional[List[torch.Tensor]] = None,
                            backend: str = 'auto',
                            use_selfplay_calibration: bool = True) -> nn.Module:
    """
    Apply static INT8 quantization to AlphaZeroNet model for CPU inference.
    This provides the best performance for CNN architectures.
    
    Args:
        model: Original AlphaZeroNet model
        calibration_data: Calibration dataset (if None, will generate)
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM, 'auto' to detect)
        use_selfplay_calibration: Use self-play for calibration (slower but higher quality)
        
    Returns:
        Statically quantized model (INT8 weights and activations)
    """
    # Move model to CPU (required for quantization)
    model = model.cpu()
    model.eval()
    
    # Auto-detect backend if needed
    if backend == 'auto':
        import platform
        system = platform.system()
        machine = platform.machine()
        
        if system == 'Darwin':  # macOS
            backend = 'qnnpack'
        elif machine.startswith('arm') or machine == 'aarch64':
            backend = 'qnnpack'
        else:
            # x86 Linux/Windows
            backend = 'fbgemm'
    
    # Set quantization backend
    try:
        torch.backends.quantized.engine = backend
        print(f"Using quantization backend: {backend}")
    except RuntimeError as e:
        if 'fbgemm' in str(e).lower():
            print(f"FBGEMM not available, falling back to QNNPACK")
            backend = 'qnnpack'
            torch.backends.quantized.engine = backend
        else:
            raise
    
    # Prepare model for quantization
    quant_model = prepare_alphazero_for_quantization(model)
    quant_model.eval()
    
    # Set quantization config
    if backend == 'fbgemm':
        quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        quant_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Fuse modules for better performance (Conv+BN+ReLU fusion)
    quant_model = torch.quantization.fuse_modules(quant_model, 
        [['convBlock1.conv1', 'convBlock1.bn1', 'convBlock1.relu1']])
    
    # Fuse residual blocks (note: don't fuse conv2+bn2+relu2 because relu2 comes after the skip connection)
    for i, block in enumerate(quant_model.residualBlocks):
        torch.quantization.fuse_modules(block,
            [['conv1', 'bn1', 'relu1'],
             ['conv2', 'bn2']], inplace=True)
    
    # Fuse value head
    torch.quantization.fuse_modules(quant_model.valueHead,
        [['conv1', 'bn1', 'relu1']], inplace=True)
    
    # Fuse policy head
    torch.quantization.fuse_modules(quant_model.policyHead,
        [['conv1', 'bn1', 'relu1']], inplace=True)
    
    # Prepare model (insert observers)
    torch.quantization.prepare(quant_model, inplace=True)
    
    # Generate calibration data if not provided
    if calibration_data is None:
        if use_selfplay_calibration:
            # Use high-quality self-play calibration
            calibration_data = create_calibration_dataset_selfplay(
                model=model,
                num_positions=1000,
                rollouts_per_move=100,
                threads=10,
                input_planes=16
            )
        else:
            # Use random calibration (faster but lower quality)
            calibration_data = create_calibration_dataset(1000, input_planes=16)
    
    # Calibrate the model
    calibrate_model(quant_model, calibration_data)
    
    # Convert to quantized model
    print("Converting to INT8 quantized model...")
    quantized_model = torch.quantization.convert(quant_model)
    
    print("Static quantization complete!")
    return quantized_model


def save_quantized_model(quantized_model: nn.Module, original_path: str, 
                        suffix: str = "_quantized") -> str:
    """
    Save quantized AlphaZeroNet model with metadata.
    
    Args:
        quantized_model: The quantized model
        original_path: Path to original model
        suffix: Suffix to add to filename
        
    Returns:
        Path to saved quantized model
    """
    base, ext = os.path.splitext(original_path)
    quantized_path = f"{base}{suffix}{ext}"
    
    # Try to save as TorchScript (recommended for quantized models)
    try:
        # For static quantization, scripting is preferred
        torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    except:
        # Fallback to regular save
        torch.save(quantized_model.state_dict(), quantized_path)
    
    # Print size comparison
    if os.path.exists(original_path):
        original_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\nModel size comparison:")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        print(f"\nExpected speedup: 2-4x on CPU inference")
    
    return quantized_path


def load_quantized_model(model_path: str) -> nn.Module:
    """
    Load a quantized AlphaZeroNet model from file.
    
    Args:
        model_path: Path to quantized model file
        
    Returns:
        Loaded quantized model ready for inference
    """
    try:
        # Try to load as TorchScript first
        model = torch.jit.load(model_path, map_location='cpu')
    except:
        # If that fails, load as state dict
        print("Loading as state dict")
        raise RuntimeError(
            "Model saved as state dict. To load, recreate the quantized model "
            "architecture and load the state dict manually."
        )
    
    model.eval()
    
    print(f"Loaded quantized model from {model_path}")
    print("Model is ready for CPU inference")
    
    return model


def benchmark_quantized_model(original_model: AlphaZeroNet, quantized_model: nn.Module,
                             num_positions: int = 100, input_planes: int = 16) -> dict:
    """
    Benchmark the quantized model against the original.
    
    Args:
        original_model: Original FP32 AlphaZeroNet model
        quantized_model: Quantized INT8 model
        num_positions: Number of test positions
        input_planes: Number of input planes (16 for AlphaZero)
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    print(f"\nBenchmarking models on {num_positions} positions...")
    
    # Generate test positions
    test_positions = []
    for _ in range(num_positions):
        board = chess.Board()
        num_moves = np.random.randint(0, 60)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(np.random.choice(legal_moves))
        
        input_data = encoder.encodePosition(board)
        test_positions.append(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
    
    # Benchmark original model
    original_model.eval()
    original_model.cpu()
    
    # Create dummy policy mask (all moves legal for benchmarking)
    policy_mask = torch.ones((1, 72, 8, 8), dtype=torch.float32)
    
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = original_model(pos, policyMask=policy_mask)
    original_time = time.time() - start_time
    
    # Benchmark quantized model
    quantized_model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = quantized_model(pos, policyMask=policy_mask)
    quantized_time = time.time() - start_time
    
    # Calculate speedup
    speedup = original_time / quantized_time
    
    # Compare outputs on a sample
    with torch.no_grad():
        sample_input = test_positions[0]
        orig_value, orig_policy = original_model(sample_input, policyMask=policy_mask)
        quant_value, quant_policy = quantized_model(sample_input, policyMask=policy_mask)
        
        policy_diff = torch.abs(orig_policy - quant_policy).mean().item()
        value_diff = torch.abs(orig_value - quant_value).mean().item()
    
    results = {
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup': speedup,
        'positions_per_second_original': num_positions / original_time,
        'positions_per_second_quantized': num_positions / quantized_time,
        'policy_diff': policy_diff,
        'value_diff': value_diff
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Original model: {results['positions_per_second_original']:.1f} pos/sec")
    print(f"  Quantized model: {results['positions_per_second_quantized']:.1f} pos/sec")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Policy difference: {policy_diff:.6f}")
    print(f"  Value difference: {value_diff:.6f}")
    
    return results