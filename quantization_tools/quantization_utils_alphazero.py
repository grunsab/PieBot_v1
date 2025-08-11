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
                                       rollouts_per_move: int = 20,
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
    moves_per_game = 40  # Collect first 40 moves of each game
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
        if (game_idx + 1) % 1 == 0:
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
                num_positions=300,
                rollouts_per_move=20,
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


def check_accuracy_preservation(original_model: AlphaZeroNet, quantized_model: nn.Module,
                               calibration_data: Optional[List[torch.Tensor]] = None,
                               num_positions: int = 200, input_planes: int = 16,
                               tolerance: float = 0.01) -> dict:
    """
    Check how well the quantized model preserves accuracy compared to the original.
    
    Args:
        original_model: Original FP32 AlphaZeroNet model
        quantized_model: Quantized INT8 model
        calibration_data: If provided, use these positions for accuracy checking (from calibration)
        num_positions: Number of test positions to evaluate (if calibration_data not provided)
        input_planes: Number of input planes (16 for AlphaZero)
        tolerance: Maximum acceptable difference for considering outputs as matching
        
    Returns:
        Dictionary with detailed accuracy metrics
    """
    # Use calibration data if provided, otherwise generate test positions
    if calibration_data is not None:
        test_positions = calibration_data[:min(len(calibration_data), num_positions)]
        actual_num_positions = len(test_positions)
        print(f"\nChecking accuracy preservation on {actual_num_positions} calibration positions...")
        test_boards = None  # We'll skip detailed move checks when using calibration data
    else:
        print(f"\nChecking accuracy preservation on {num_positions} generated positions...")
        
        # Generate diverse test positions with boards for move analysis
        test_positions = []
        test_boards = []
        
        for i in range(num_positions):
            board = chess.Board()
            
            # Vary game phase for comprehensive testing
            if i < num_positions // 3:
                # Opening positions
                num_moves = np.random.randint(0, 15)
            elif i < 2 * num_positions // 3:
                # Middle game positions
                num_moves = np.random.randint(15, 40)
            else:
                # Endgame positions
                num_moves = np.random.randint(40, 80)
            
            for _ in range(num_moves):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                board.push(np.random.choice(legal_moves))
            
            test_boards.append(board.copy())
            input_data = encoder.encodePosition(board)
            test_positions.append(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
        
        actual_num_positions = num_positions
    
    # Prepare models
    original_model.eval()
    original_model.cpu()
    quantized_model.eval()
    
    # Collect metrics
    value_diffs = []
    policy_diffs = []
    policy_kl_divs = []
    top1_matches = 0
    top3_matches = 0
    top5_matches = 0
    
    print("Evaluating output differences...")
    
    with torch.no_grad():
        for idx, pos in enumerate(test_positions):
            # For detailed move analysis, we need the board state
            if test_boards is not None:
                board = test_boards[idx]
                # Create policy mask for legal moves
                legal_moves = list(board.legal_moves)
                policy_mask = torch.zeros((1, 72, 8, 8), dtype=torch.float32)
                for move in legal_moves:
                    move_idx = encoder.encodeMove(move, board.turn)
                    if move_idx is not None:
                        plane, row, col = move_idx
                        policy_mask[0, plane, row, col] = 1.0
            else:
                # Use a permissive mask when we don't have board state
                policy_mask = torch.ones((1, 72, 8, 8), dtype=torch.float32)
            
            # Get outputs from both models
            orig_value, orig_policy = original_model(pos, policyMask=policy_mask)
            quant_value, quant_policy = quantized_model(pos, policyMask=policy_mask)
            
            # Calculate value difference
            value_diff = torch.abs(orig_value - quant_value).item()
            value_diffs.append(value_diff)
            
            # Calculate policy differences
            policy_diff = torch.abs(orig_policy - quant_policy).mean().item()
            policy_diffs.append(policy_diff)
            
            # Calculate KL divergence for policy distributions
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            orig_policy_safe = orig_policy + epsilon
            quant_policy_safe = quant_policy + epsilon
            kl_div = (orig_policy_safe * torch.log(orig_policy_safe / quant_policy_safe)).sum().item()
            policy_kl_divs.append(kl_div)
            
            # Check top move predictions only if we have board state
            if test_boards is not None:
                orig_policy_flat = orig_policy.view(-1)
                quant_policy_flat = quant_policy.view(-1)
                
                # Get top-k indices
                orig_top5 = torch.topk(orig_policy_flat, min(5, len(legal_moves))).indices
                quant_top5 = torch.topk(quant_policy_flat, min(5, len(legal_moves))).indices
                
                if orig_top5[0] == quant_top5[0]:
                    top1_matches += 1
                if orig_top5[0] in quant_top5[:3]:
                    top3_matches += 1
                if orig_top5[0] in quant_top5[:5]:
                    top5_matches += 1
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"  Evaluated {idx + 1}/{actual_num_positions} positions")
    
    # Calculate statistics
    value_diffs = np.array(value_diffs)
    policy_diffs = np.array(policy_diffs)
    policy_kl_divs = np.array(policy_kl_divs)
    
    accuracy_results = {
        'value_mean_diff': np.mean(value_diffs),
        'value_max_diff': np.max(value_diffs),
        'value_std_diff': np.std(value_diffs),
        'value_within_tolerance': np.sum(value_diffs < tolerance) / len(value_diffs) * 100,
        
        'policy_mean_diff': np.mean(policy_diffs),
        'policy_max_diff': np.max(policy_diffs),
        'policy_std_diff': np.std(policy_diffs),
        'policy_within_tolerance': np.sum(policy_diffs < tolerance) / len(policy_diffs) * 100,
        
        'policy_mean_kl_div': np.mean(policy_kl_divs),
        'policy_max_kl_div': np.max(policy_kl_divs),
        
        'num_positions_tested': actual_num_positions,
        'tolerance': tolerance,
        'used_calibration_data': calibration_data is not None
    }
    
    # Add move prediction rates only if we have board states
    if test_boards is not None:
        accuracy_results['top1_match_rate'] = top1_matches / actual_num_positions * 100
        accuracy_results['top3_match_rate'] = top3_matches / actual_num_positions * 100
        accuracy_results['top5_match_rate'] = top5_matches / actual_num_positions * 100
    
    # Print detailed results
    print("\n" + "="*60)
    print("ACCURACY PRESERVATION ANALYSIS")
    print("="*60)
    
    if calibration_data is not None:
        print(f"Using {actual_num_positions} positions from calibration dataset")
    else:
        print(f"Using {actual_num_positions} randomly generated positions")
    
    print("\nValue Head Accuracy:")
    print(f"  Mean absolute difference: {accuracy_results['value_mean_diff']:.6f}")
    print(f"  Max absolute difference:  {accuracy_results['value_max_diff']:.6f}")
    print(f"  Std deviation:            {accuracy_results['value_std_diff']:.6f}")
    print(f"  Within tolerance ({tolerance}): {accuracy_results['value_within_tolerance']:.1f}%")
    
    print("\nPolicy Head Accuracy:")
    print(f"  Mean absolute difference: {accuracy_results['policy_mean_diff']:.6f}")
    print(f"  Max absolute difference:  {accuracy_results['policy_max_diff']:.6f}")
    print(f"  Std deviation:            {accuracy_results['policy_std_diff']:.6f}")
    print(f"  Within tolerance ({tolerance}): {accuracy_results['policy_within_tolerance']:.1f}%")
    print(f"  Mean KL divergence:       {accuracy_results['policy_mean_kl_div']:.6f}")
    
    if test_boards is not None:
        print("\nMove Prediction Consistency:")
        print(f"  Top-1 move match rate: {accuracy_results['top1_match_rate']:.1f}%")
        print(f"  Top-3 move match rate: {accuracy_results['top3_match_rate']:.1f}%")
        print(f"  Top-5 move match rate: {accuracy_results['top5_match_rate']:.1f}%")
    
    # Overall assessment
    print("\nOverall Assessment:")
    if test_boards is not None:
        # Full assessment with move prediction
        if (accuracy_results['value_within_tolerance'] > 95 and 
            accuracy_results['policy_within_tolerance'] > 95 and
            accuracy_results['top1_match_rate'] > 85):
            print("  ✓ EXCELLENT: Quantized model maintains high accuracy")
        elif (accuracy_results['value_within_tolerance'] > 90 and 
              accuracy_results['policy_within_tolerance'] > 90 and
              accuracy_results['top1_match_rate'] > 75):
            print("  ✓ GOOD: Quantized model maintains acceptable accuracy")
        elif (accuracy_results['value_within_tolerance'] > 80 and 
              accuracy_results['policy_within_tolerance'] > 80 and
              accuracy_results['top1_match_rate'] > 65):
            print("  ⚠ FAIR: Some accuracy loss, but model is still usable")
        else:
            print("  ✗ POOR: Significant accuracy loss detected")
    else:
        # Assessment based on value/policy differences only
        if (accuracy_results['value_within_tolerance'] > 95 and 
            accuracy_results['policy_within_tolerance'] > 95):
            print("  ✓ EXCELLENT: Quantized model maintains high accuracy")
        elif (accuracy_results['value_within_tolerance'] > 90 and 
              accuracy_results['policy_within_tolerance'] > 90):
            print("  ✓ GOOD: Quantized model maintains acceptable accuracy")
        elif (accuracy_results['value_within_tolerance'] > 80 and 
              accuracy_results['policy_within_tolerance'] > 80):
            print("  ⚠ FAIR: Some accuracy loss, but model is still usable")
        else:
            print("  ✗ POOR: Significant accuracy loss detected")
    
    print("="*60)
    
    return accuracy_results


def benchmark_quantized_model(original_model: AlphaZeroNet, quantized_model: nn.Module,
                             num_positions: int = 100, input_planes: int = 16) -> dict:
    """
    Benchmark the quantized model against the original for speed comparison.
    
    Args:
        original_model: Original FP32 AlphaZeroNet model
        quantized_model: Quantized INT8 model
        num_positions: Number of test positions
        input_planes: Number of input planes (16 for AlphaZero)
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    print(f"\nBenchmarking inference speed on {num_positions} positions...")
    
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
    quantized_model.eval()
    
    # Create dummy policy mask (all moves legal for benchmarking)
    policy_mask = torch.ones((1, 72, 8, 8), dtype=torch.float32)
    
    # Warm-up runs
    print("  Running warm-up iterations...")
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(test_positions[0], policyMask=policy_mask)
            _ = quantized_model(test_positions[0], policyMask=policy_mask)
    
    # Benchmark original model
    print("  Benchmarking original model...")
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = original_model(pos, policyMask=policy_mask)
    original_time = time.time() - start_time
    
    # Benchmark quantized model
    print("  Benchmarking quantized model...")
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = quantized_model(pos, policyMask=policy_mask)
    quantized_time = time.time() - start_time
    
    # Calculate speedup
    speedup = original_time / quantized_time
    
    results = {
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup': speedup,
        'positions_per_second_original': num_positions / original_time,
        'positions_per_second_quantized': num_positions / quantized_time,
    }
    
    print(f"\nSpeed Benchmark Results:")
    print(f"  Original model:  {results['positions_per_second_original']:.1f} pos/sec")
    print(f"  Quantized model: {results['positions_per_second_quantized']:.1f} pos/sec")
    print(f"  Speedup:         {speedup:.2f}x")
    
    return results