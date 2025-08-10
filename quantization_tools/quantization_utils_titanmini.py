"""
Quantization utilities for TitanMini transformer neural network models.

This module provides functions to quantize TitanMini models for improved CPU inference performance,
reducing model size and increasing throughput while maintaining accuracy.
Optimized specifically for transformer architectures on CPU for deployment on low-end VPS.
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
import encoder_enhanced
from TitanMiniNetwork import TitanMini
from device_utils import get_optimal_device


class QuantizableTitanMini(nn.Module):
    """
    Quantization-ready version of TitanMini with quant/dequant stubs for static quantization.
    Handles the transformer architecture's special requirements.
    """
    
    def __init__(self, base_model: TitanMini):
        super().__init__()
        self.quant = QuantStub()
        self.dequant_value = DeQuantStub()
        self.dequant_policy = DeQuantStub()
        
        # Copy the base model components
        self.input_projection = base_model.input_projection
        self.pos_encoder = base_model.pos_encoder
        self.relative_pos_bias = base_model.relative_pos_bias
        self.transformer_layers = base_model.transformer_layers
        self.layer_norm = base_model.layer_norm
        
        # CLS token if used
        if hasattr(base_model, 'cls_token'):
            self.cls_token = base_model.cls_token
            self.use_cls_token = base_model.use_cls_token
        else:
            self.use_cls_token = False
        
        # Output heads
        self.value_head = base_model.value_head
        self.policy_head = base_model.policy_head
        
        # Copy other attributes
        self.d_model = base_model.d_model
        self.n_heads = base_model.n_heads
        
    def forward(self, x):
        """
        Forward pass with quantization stubs.
        """
        # Quantize input
        x = self.quant(x)
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Reshape input: (B, C, 8, 8) -> (B, 64, C)
        x = x.view(batch_size, x.shape[1], -1).transpose(1, 2)
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, 1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Get sequence length and relative position bias
        seq_len = x.shape[1]
        rel_pos_bias = self.relative_pos_bias(seq_len)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, rel_pos_bias=rel_pos_bias)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Extract features for heads
        if self.use_cls_token:
            # Use CLS token for value head
            cls_features = x[:, 0, :]
            board_features = x[:, 1:, :]
        else:
            # Use global average pooling for value head
            cls_features = x.mean(dim=1)
            board_features = x
        
        # Compute heads
        value_logits = self.value_head(cls_features)
        policy_logits = self.policy_head(board_features)
        
        # Dequantize outputs
        value_logits = self.dequant_value(value_logits)
        policy_logits = self.dequant_policy(policy_logits)
        
        return policy_logits, value_logits


def prepare_titanmini_for_quantization(model: TitanMini) -> QuantizableTitanMini:
    """
    Prepare a TitanMini model for static quantization.
    
    Args:
        model: Original TitanMini model
        
    Returns:
        QuantizableTitanMini ready for quantization
    """
    # Create quantizable model
    quant_model = QuantizableTitanMini(model)
    
    # Copy all parameters
    quant_model.load_state_dict(model.state_dict(), strict=False)
    
    return quant_model


def create_calibration_dataset(num_positions: int = 500, input_planes: int = 112) -> List[torch.Tensor]:
    """
    Create a calibration dataset from various chess positions.
    Optimized for TitanMini with enhanced encoder (112 planes).
    
    Args:
        num_positions: Number of positions to generate (default 500 for faster calibration)
        input_planes: Number of input planes (112 for TitanMini with enhanced encoder)
        
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
        
        # Encode the position (TitanMini uses enhanced encoder)
        input_data = encoder_enhanced.encode_enhanced_position(board)
        
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
            # Run forward pass for calibration (inference mode)
            _ = model(input_tensor.cpu())
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Calibrated on {idx + 1}/{len(calibration_data)} positions")


def apply_dynamic_quantization(model: TitanMini) -> nn.Module:
    """
    Apply dynamic INT8 quantization to TitanMini model for CPU inference.
    This quantizes weights to INT8 while keeping activations in FP32.
    Recommended for transformer architectures as they are more sensitive to quantization.
    
    Args:
        model: Original TitanMini model
        
    Returns:
        Dynamically quantized model
    """
    # Move model to CPU (required for quantization)
    model = model.cpu()
    model.eval()
    
    # Apply dynamic quantization to Linear layers (main components in transformers)
    # Note: Conv2d layers are rare in TitanMini, mainly Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},  # Focus on Linear layers for transformers
        dtype=torch.qint8
    )
    
    print("Dynamic quantization complete!")
    print("Note: Weights are INT8, activations remain FP32")
    print("This is optimal for transformer architectures like TitanMini")
    
    return quantized_model


def apply_static_quantization(model: TitanMini,
                            calibration_data: Optional[List[torch.Tensor]] = None,
                            backend: str = 'auto') -> nn.Module:
    """
    Apply static INT8 quantization to TitanMini model for CPU inference.
    Note: Transformers may see reduced benefits from static quantization.
    
    Args:
        model: Original TitanMini model
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
    quant_model = prepare_titanmini_for_quantization(model)
    quant_model.eval()
    
    # Set quantization config for static quantization
    if backend == 'fbgemm':
        quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        quant_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Note: Module fusion is complex for transformers and may not provide benefits
    # Skipping fusion for TitanMini's complex architecture
    
    # Prepare model (insert observers)
    torch.quantization.prepare(quant_model, inplace=True)
    
    # Generate calibration data if not provided
    if calibration_data is None:
        calibration_data = create_calibration_dataset(500, input_planes=112)
    
    # Calibrate the model
    calibrate_model(quant_model, calibration_data)
    
    # Convert to quantized model
    print("Converting to INT8 quantized model...")
    print("Warning: Transformers may have limited quantization benefits")
    quantized_model = torch.quantization.convert(quant_model)
    
    print("Static quantization complete!")
    return quantized_model


def save_quantized_model(quantized_model: nn.Module, original_path: str, 
                        suffix: str = "_quantized") -> str:
    """
    Save quantized TitanMini model with metadata.
    
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
        # For transformers, scripting can be challenging
        torch.jit.save(torch.jit.script(quantized_model), quantized_path)
    except:
        # Fallback to regular save for dynamic quantization
        # This is common for transformer models
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
        print(f"\nExpected speedup: 1.5-3x on CPU inference")
        print(f"Note: Transformers typically see smaller speedups than CNNs")
    
    return quantized_path


def load_quantized_model(model_path: str) -> nn.Module:
    """
    Load a quantized TitanMini model from file.
    
    Args:
        model_path: Path to quantized model file
        
    Returns:
        Loaded quantized model ready for inference
    """
    try:
        # Try to load as TorchScript first
        model = torch.jit.load(model_path, map_location='cpu')
    except:
        # If that fails, load as state dict (common for transformers)
        print("Loading as state dict (transformer model)")
        # This would require reconstructing the model architecture
        # For now, raise an informative error
        raise RuntimeError(
            "Model saved as state dict. To load, recreate the quantized model "
            "architecture and load the state dict manually."
        )
    
    model.eval()
    
    print(f"Loaded quantized model from {model_path}")
    print("Model is ready for CPU inference")
    
    return model


def benchmark_quantized_model(original_model: TitanMini, quantized_model: nn.Module,
                             num_positions: int = 100, input_planes: int = 112) -> dict:
    """
    Benchmark the quantized model against the original.
    
    Args:
        original_model: Original FP32 TitanMini model
        quantized_model: Quantized INT8 model
        num_positions: Number of test positions
        input_planes: Number of input planes (112 for TitanMini)
        
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
        
        # TitanMini uses enhanced encoder
        input_data = encoder_enhanced.encode_enhanced_position(board)
        test_positions.append(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0))
    
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