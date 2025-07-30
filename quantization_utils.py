"""
Quantization utilities for AlphaZero neural network models.

This module provides functions to quantize AlphaZero models for improved inference performance,
reducing model size and increasing throughput while maintaining accuracy.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import os
import chess
import numpy as np
from typing import List, Tuple, Optional
import encoder
import AlphaZeroNetwork
from device_utils import get_optimal_device, move_to_device


class QuantizableAlphaZeroNet(nn.Module):
    """
    Quantization-ready version of AlphaZeroNet with quant/dequant stubs.
    """
    
    def __init__(self, num_blocks, num_filters, policy_weight=1.0):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Create the base model
        self.base_model = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_filters, policy_weight)
        
    def forward(self, x, valueTarget=None, policyTarget=None, policyMask=None):
        # Quantize input
        x = self.quant(x)
        
        # Forward through the model components
        x = self.base_model.convBlock1(x)
        
        for block in self.base_model.residualBlocks:
            x = block(x)
        
        # Split for value and policy heads
        value = self.base_model.valueHead(x)
        policy = self.base_model.policyHead(x)
        
        # Dequantize outputs
        value = self.dequant(value)
        policy = self.dequant(policy)
        
        # Handle training vs inference
        if self.training:
            return self.base_model(x, valueTarget, policyTarget, policyMask)
        else:
            # Apply softmax for policy during inference
            if policyMask is not None:
                policyMask = policyMask.view(policyMask.shape[0], -1)
                policy_exp = torch.exp(policy)
                policy_exp *= policyMask.type(torch.float32)
                policy_exp_sum = torch.sum(policy_exp, dim=1, keepdim=True)
                policy_softmax = policy_exp / policy_exp_sum
                return value, policy_softmax
            return value, policy


def prepare_model_for_quantization(model: AlphaZeroNetwork.AlphaZeroNet) -> QuantizableAlphaZeroNet:
    """
    Prepare an AlphaZero model for quantization by wrapping it with quantization stubs.
    
    Args:
        model: Original AlphaZeroNet model
        
    Returns:
        QuantizableAlphaZeroNet ready for quantization
    """
    # Extract parameters
    num_blocks = len(model.residualBlocks)
    num_filters = model.convBlock1.conv1.out_channels
    policy_weight = model.policy_weight
    
    # Create quantizable model
    quant_model = QuantizableAlphaZeroNet(num_blocks, num_filters, policy_weight)
    
    # Copy weights
    quant_model.base_model.load_state_dict(model.state_dict())
    
    return quant_model


class DynamicQuantizedAlphaZeroNet(nn.Module):
    """
    Wrapper for dynamically quantized AlphaZero model that handles inference only.
    """
    def __init__(self, model: AlphaZeroNetwork.AlphaZeroNet):
        super().__init__()
        # Check if quantization is supported
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            model, 
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
    def forward(self, x, policyMask=None):
        """
        Forward pass for inference only.
        """
        # Set model to eval mode to ensure consistent behavior
        self.quantized_model.eval()
        
        # Call the quantized model
        with torch.no_grad():
            value, policy = self.quantized_model(x, policyMask=policyMask)
        
        return value, policy


def apply_dynamic_quantization(model: AlphaZeroNetwork.AlphaZeroNet) -> nn.Module:
    """
    Apply dynamic quantization to the model (easiest, no calibration needed).
    Quantizes weights to INT8, keeps activations in FP32.
    
    Args:
        model: Original AlphaZeroNet model
        
    Returns:
        Dynamically quantized model wrapped for inference
    """
    # Set model to eval mode
    model.eval()
    
    # Move model to CPU for quantization (quantization only works on CPU)
    device = next(model.parameters()).device
    model_cpu = model.cpu()
    
    # Create wrapper that handles dynamic quantization
    quantized_wrapper = DynamicQuantizedAlphaZeroNet(model_cpu)
    
    # Note: Quantized models must run on CPU, even on CUDA systems
    print("Note: Quantized model will run on CPU for inference")
    
    return quantized_wrapper


def create_calibration_dataset(num_positions: int = 1000) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a calibration dataset from various chess positions.
    
    Args:
        num_positions: Number of positions to generate
        
    Returns:
        List of (input_tensor, mask_tensor) tuples
    """
    calibration_data = []
    
    # Generate positions from different game phases
    for i in range(num_positions):
        board = chess.Board()
        
        # Play random moves to get diverse positions
        num_moves = np.random.randint(0, 60)  # 0-60 moves
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            board.push(move)
        
        # Encode the position
        input_planes = encoder.encodePosition(board)
        input_tensor = torch.tensor(input_planes, dtype=torch.float32).unsqueeze(0)
        
        # Create legal move mask
        mask = encoder.getLegalMoveMask(board)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        calibration_data.append((input_tensor, mask_tensor))
    
    return calibration_data


def calibrate_model(model: nn.Module, calibration_data: List[Tuple[torch.Tensor, torch.Tensor]],
                   device: torch.device) -> None:
    """
    Calibrate the model for static quantization.
    
    Args:
        model: Model prepared for quantization
        calibration_data: List of calibration samples
        device: Device to run calibration on
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for idx, (input_tensor, mask_tensor) in enumerate(calibration_data):
            input_tensor = input_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            # Run forward pass for calibration
            # Only pass required arguments for inference
            _ = model(input_tensor, policyMask=mask_tensor)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Calibrated on {idx + 1}/{len(calibration_data)} positions")


def apply_static_quantization(model: AlphaZeroNetwork.AlphaZeroNet,
                            calibration_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                            backend: str = 'fbgemm') -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Args:
        model: Original AlphaZeroNet model
        calibration_data: Calibration dataset (if None, will generate)
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM/mobile)
        
    Returns:
        Statically quantized model
    """
    # Move model to CPU for quantization
    device = next(model.parameters()).device
    model = model.cpu()
    
    # Set quantization backend - use fbgemm for Windows/Linux x86
    import platform
    if platform.system() == "Windows":
        backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    # Prepare model for quantization
    quant_model = prepare_model_for_quantization(model)
    
    # Set quantization config
    # Use per-tensor quantization to avoid per_channel_affine error
    if backend == 'fbgemm':
        # For x86 CPUs, use per-tensor quantization
        quant_model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_tensor_weight_observer
        )
    else:
        # For other backends (ARM/mobile)
        quant_model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare model (insert observers)
    torch.quantization.prepare(quant_model, inplace=True)
    
    # Generate calibration data if not provided
    if calibration_data is None:
        print("Generating calibration dataset...")
        calibration_data = create_calibration_dataset(1000)
    
    # Calibrate the model on CPU (required for quantization)
    print(f"Calibrating model on CPU...")
    calibrate_model(quant_model, calibration_data, torch.device('cpu'))
    
    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = torch.quantization.convert(quant_model)
    
    return quantized_model


class FP16OptimizedAlphaZeroNet(nn.Module):
    """
    Wrapper for FP16 (half precision) optimized model with automatic mixed precision.
    """
    def __init__(self, model: AlphaZeroNetwork.AlphaZeroNet, device: torch.device):
        super().__init__()
        self.model = model.half().to(device)
        self.device = device
        
    def forward(self, x, policyMask=None):
        """
        Forward pass with FP16 optimization.
        """
        # Ensure inputs are FP16
        x = x.half()
        if policyMask is not None:
            policyMask = policyMask.half()
        
        # Run inference
        with torch.cuda.amp.autocast(enabled=True):
            return self.model(x, policyMask=policyMask)


def apply_fp16_optimization(model: AlphaZeroNetwork.AlphaZeroNet, device: torch.device) -> nn.Module:
    """
    Apply FP16 (half precision) optimization for GPU/MPS inference.
    This is an alternative to quantization that works on GPU.
    
    Args:
        model: Original AlphaZeroNet model
        device: Target device (cuda or mps)
        
    Returns:
        FP16 optimized model
    """
    if device.type not in ['cuda', 'mps']:
        print("FP16 optimization is only beneficial on GPU/MPS. Returning original model.")
        return model
    
    # Set model to eval mode
    model.eval()
    
    # Create FP16 wrapper
    fp16_model = FP16OptimizedAlphaZeroNet(model, device)
    
    print(f"Model converted to FP16 for {device.type.upper()} inference")
    print("Note: This provides ~2x memory reduction and faster inference on modern GPUs")
    
    return fp16_model


def save_quantized_model(quantized_model: nn.Module, original_path: str, suffix: str = "_quantized") -> str:
    """
    Save quantized model with appropriate naming.
    
    Args:
        quantized_model: The quantized model
        original_path: Path to original model
        suffix: Suffix to add to filename
        
    Returns:
        Path to saved quantized model
    """
    base, ext = os.path.splitext(original_path)
    quantized_path = f"{base}{suffix}{ext}"
    
    # Handle different model types
    if isinstance(quantized_model, DynamicQuantizedAlphaZeroNet):
        # Save the state dict for dynamic quantized
        torch.save({
            'model_state_dict': quantized_model.quantized_model.state_dict(),
            'model_type': 'dynamic_quantized'
        }, quantized_path)
    elif isinstance(quantized_model, FP16OptimizedAlphaZeroNet):
        # Save FP16 model with metadata
        torch.save({
            'model_state_dict': quantized_model.model.state_dict(),
            'model_type': 'fp16',
            'device_type': quantized_model.device.type
        }, quantized_path)
    elif hasattr(quantized_model, 'qconfig'):
        # For static quantized models, use torch.jit.save
        scripted = torch.jit.script(quantized_model)
        torch.jit.save(scripted, quantized_path)
    else:
        # For regular models, save normally
        torch.save(quantized_model.state_dict(), quantized_path)
    
    # Print size comparison
    original_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\nModel size comparison:")
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    
    return quantized_path


def load_quantized_model(model_path: str, device: torch.device, 
                        num_blocks: int = None, num_filters: int = None) -> nn.Module:
    """
    Load a quantized model from file.
    
    Args:
        model_path: Path to quantized model file
        device: Device to load model on
        num_blocks: Number of residual blocks (for dynamic quantized models)
        num_filters: Number of filters (for dynamic quantized models)
        
    Returns:
        Loaded quantized model
    """
    try:
        # Try loading as TorchScript (static quantized)
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    except:
        # Try loading as state dict (dynamic quantized)
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get('model_type') == 'dynamic_quantized':
            # Need to reconstruct the model
            if num_blocks is None or num_filters is None:
                raise ValueError("For dynamic quantized models, num_blocks and num_filters must be provided")
            
            # Create base model
            base_model = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_filters)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Apply dynamic quantization
            quantized_model = apply_dynamic_quantization(base_model)
            return quantized_model
        else:
            raise ValueError(f"Unknown model format in {model_path}")


def compare_model_outputs(original_model: nn.Module, quantized_model: nn.Module,
                         test_positions: List[Tuple[torch.Tensor, torch.Tensor]],
                         device: torch.device) -> dict:
    """
    Compare outputs between original and quantized models.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized model
        test_positions: List of test positions
        device: Device to run comparison on
        
    Returns:
        Dictionary with comparison metrics
    """
    original_model.eval()
    quantized_model.eval()
    original_model.to(device)
    
    value_diffs = []
    policy_diffs = []
    
    with torch.no_grad():
        for input_tensor, mask_tensor in test_positions:
            input_tensor = input_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            # Get outputs from both models
            orig_value, orig_policy = original_model(input_tensor, policyMask=mask_tensor)
            
            # For quantized model, need to handle CPU execution
            if hasattr(quantized_model, 'forward'):
                input_cpu = input_tensor.cpu()
                mask_cpu = mask_tensor.cpu()
                quant_value, quant_policy = quantized_model(input_cpu, policyMask=mask_cpu)
                quant_value = quant_value.to(device)
                quant_policy = quant_policy.to(device)
            else:
                quant_value, quant_policy = quantized_model(input_tensor, policyMask=mask_tensor)
            
            # Calculate differences
            value_diff = torch.abs(orig_value - quant_value).mean().item()
            policy_diff = torch.abs(orig_policy - quant_policy).mean().item()
            
            value_diffs.append(value_diff)
            policy_diffs.append(policy_diff)
    
    return {
        'avg_value_diff': np.mean(value_diffs),
        'max_value_diff': np.max(value_diffs),
        'avg_policy_diff': np.mean(policy_diffs),
        'max_policy_diff': np.max(policy_diffs),
        'value_rmse': np.sqrt(np.mean(np.square(value_diffs))),
        'policy_rmse': np.sqrt(np.mean(np.square(policy_diffs)))
    }