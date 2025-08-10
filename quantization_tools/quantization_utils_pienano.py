"""
Quantization utilities for PieNano neural network models.

This module provides functions to quantize PieNano models for improved CPU inference performance,
reducing model size and increasing throughput while maintaining accuracy.
Optimized specifically for static quantization on CPU for deployment on low-end VPS.
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
from PieNanoNetwork import PieNano
from device_utils import get_optimal_device


class QuantizablePieNano(nn.Module):
    """
    Quantization-ready version of PieNano with quant/dequant stubs for static quantization.
    """
    
    def __init__(self, base_model: PieNano):
        super().__init__()
        self.quant = QuantStub()
        self.dequant_value = DeQuantStub()
        self.dequant_policy = DeQuantStub()
        
        # Copy the base model components
        self.conv_block = base_model.conv_block
        self.residual_tower = base_model.residual_tower
        self.value_head = base_model.value_head
        self.policy_head = base_model.policy_head
        self.policy_weight = base_model.policy_weight
        
    def forward(self, x, value_targets=None, policy_targets=None):
        """
        Forward pass with quantization stubs.
        """
        # Quantize input
        x = self.quant(x)
        
        # Forward through the model (matches PieNano forward)
        x = self.conv_block(x)
        x = self.residual_tower(x)
        
        # Compute heads
        value_logits = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # Dequantize outputs
        value_logits = self.dequant_value(value_logits)
        policy_logits = self.dequant_policy(policy_logits)
        
        # Return based on mode (training vs inference)
        if value_targets is not None and policy_targets is not None:
            # Training mode - compute losses (copy from PieNano)
            # This shouldn't be used after quantization, but included for completeness
            return self._compute_loss(value_logits, policy_logits, value_targets, policy_targets)
        else:
            # Inference mode
            return policy_logits, value_logits
    
    def _compute_loss(self, value_logits, policy_logits, value_targets, policy_targets):
        """Helper to compute losses (copied from PieNano)"""
        import torch.nn.functional as F
        
        # Value loss (WDL uses CrossEntropy)
        if value_targets.dim() == 1 or (value_targets.dim() == 2 and value_targets.size(1) == 1):
            # Convert scalar values to WDL
            wdl_targets = torch.zeros(value_targets.size(0), 3, device=value_targets.device)
            value_targets = value_targets.view(-1)
            
            draw_prob = torch.exp(-4 * value_targets**2)
            win_prob = torch.clamp((value_targets + 1) / 2 * (1 - draw_prob), 0, 1)
            loss_prob = torch.clamp((1 - value_targets) / 2 * (1 - draw_prob), 0, 1)
            
            wdl_targets[:, 0] = win_prob
            wdl_targets[:, 1] = draw_prob
            wdl_targets[:, 2] = loss_prob
            
            wdl_targets = wdl_targets / wdl_targets.sum(dim=1, keepdim=True)
            value_targets = wdl_targets
        
        value_loss = F.cross_entropy(value_logits, value_targets.argmax(dim=1))
        
        # Policy loss
        if policy_targets.dim() == 1:
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
        else:
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum() / policy_targets.size(0)
        
        total_loss = value_loss + self.policy_weight * policy_loss
        
        return total_loss, value_loss, policy_loss


def prepare_pienano_for_quantization(model: PieNano) -> QuantizablePieNano:
    """
    Prepare a PieNano model for static quantization.
    
    Args:
        model: Original PieNano model
        
    Returns:
        QuantizablePieNano ready for quantization
    """
    # Create quantizable model
    quant_model = QuantizablePieNano(model)
    
    # Copy all parameters
    quant_model.load_state_dict(model.state_dict(), strict=False)
    
    return quant_model


def create_calibration_dataset(num_positions: int = 500, enhanced_encoder: bool = False) -> List[torch.Tensor]:
    """
    Create a calibration dataset from various chess positions.
    Optimized for PieNano with smaller dataset for faster calibration.
    
    Args:
        num_positions: Number of positions to generate (default 500 for faster calibration)
        enhanced_encoder: Whether to use 112-plane encoder
        
    Returns:
        List of input tensors for calibration
    """
    calibration_data = []
    
    print(f"Generating {num_positions} calibration positions...")
    
    # Generate positions from different game phases for diversity
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
        if enhanced_encoder:
            import encoder_enhanced
            input_planes = encoder_enhanced.encode_enhanced_position(board)
        else:
            input_planes = encoder.encodePosition(board)
        
        input_tensor = torch.tensor(input_planes, dtype=torch.float32).unsqueeze(0)
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
            # Run forward pass for calibration (inference mode)
            _ = model(input_tensor.cpu())
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Calibrated on {idx + 1}/{len(calibration_data)} positions")


def apply_dynamic_quantization(model: PieNano) -> nn.Module:
    """
    Apply dynamic INT8 quantization to PieNano model for CPU inference.
    This quantizes weights to INT8 while keeping activations in FP32.
    Simpler and more compatible than static quantization.
    
    Args:
        model: Original PieNano model
        
    Returns:
        Dynamically quantized model
    """
    # Move model to CPU (required for quantization)
    model = model.cpu()
    model.eval()
    
    # Apply dynamic quantization to Linear and Conv2d layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("Dynamic quantization complete!")
    print("Note: Weights are INT8, activations remain FP32")
    
    return quantized_model


def apply_static_quantization(model: PieNano,
                            calibration_data: Optional[List[torch.Tensor]] = None,
                            backend: str = 'auto') -> nn.Module:
    """
    Apply static INT8 quantization to PieNano model for CPU inference.
    This is the recommended approach for deployment on VPS.
    
    Args:
        model: Original PieNano model
        calibration_data: Calibration dataset (if None, will generate)
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM, 'auto' to detect)
        
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
            # macOS doesn't support fbgemm, use qnnpack
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
        # Fallback to qnnpack if fbgemm isn't available
        if 'fbgemm' in str(e).lower():
            print(f"FBGEMM not available, falling back to QNNPACK")
            backend = 'qnnpack'
            torch.backends.quantized.engine = backend
        else:
            raise
    
    # Prepare model for quantization
    quant_model = prepare_pienano_for_quantization(model)
    quant_model.eval()
    
    # Set quantization config for static quantization
    # Use fbgemm for x86 CPUs (VPS), qnnpack for ARM
    if backend == 'fbgemm':
        quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        quant_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Skip module fusion for now due to the complex architecture
    # Module fusion would require careful handling of depthwise separable convs
    
    # Prepare model (insert observers)
    torch.quantization.prepare(quant_model, inplace=True)
    
    # Generate calibration data if not provided
    if calibration_data is None:
        # Detect encoder type from model
        num_input_planes = quant_model.conv_block[0].in_channels
        enhanced = (num_input_planes == 112)
        calibration_data = create_calibration_dataset(500, enhanced_encoder=enhanced)
    
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
    Save quantized PieNano model with metadata.
    
    Args:
        quantized_model: The quantized model
        original_path: Path to original model
        suffix: Suffix to add to filename
        
    Returns:
        Path to saved quantized model
    """
    base, ext = os.path.splitext(original_path)
    quantized_path = f"{base}{suffix}{ext}"
    
    # Try to save as TorchScript (for static quantization)
    try:
        torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    except:
        # Fallback to regular save for dynamic quantization
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
    Load a quantized PieNano model from file.
    
    Args:
        model_path: Path to quantized model file
        
    Returns:
        Loaded quantized model ready for inference
    """
    # Load as TorchScript
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    print(f"Loaded quantized model from {model_path}")
    print("Model is ready for CPU inference")
    
    return model


def benchmark_quantized_model(original_model: PieNano, quantized_model: nn.Module,
                             num_positions: int = 100) -> dict:
    """
    Benchmark the quantized model against the original.
    
    Args:
        original_model: Original FP32 PieNano model
        quantized_model: Quantized INT8 model
        num_positions: Number of test positions
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    print(f"\nBenchmarking models on {num_positions} positions...")
    
    # Detect encoder type
    num_input_planes = original_model.conv_block[0].in_channels
    enhanced = (num_input_planes == 112)
    
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
        
        if enhanced:
            import encoder_enhanced
            input_planes = encoder_enhanced.encode_enhanced_position(board)
        else:
            input_planes = encoder.encodePosition(board)
        
        test_positions.append(torch.tensor(input_planes, dtype=torch.float32).unsqueeze(0))
    
    # Benchmark original model
    original_model.eval()
    original_model.cpu()
    
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = original_model(pos)
    original_time = time.time() - start_time
    
    # Benchmark quantized model
    quantized_model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for pos in test_positions:
            _ = quantized_model(pos)
    quantized_time = time.time() - start_time
    
    # Calculate speedup
    speedup = original_time / quantized_time
    
    # Compare outputs on a sample
    with torch.no_grad():
        sample_input = test_positions[0]
        orig_policy, orig_value = original_model(sample_input)
        quant_policy, quant_value = quantized_model(sample_input)
        
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