#!/usr/bin/env python3
"""
Simple script to quantize an AlphaZero model for improved performance.
"""

import argparse
import torch
import os
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
from quantization_utils import (
    apply_dynamic_quantization,
    apply_static_quantization,
    apply_fp16_optimization,
    save_quantized_model,
    create_calibration_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description='Quantize AlphaZero model for improved inference performance'
    )
    parser.add_argument('--model', help='Path to model file', required=True)
    parser.add_argument('--type', choices=['dynamic', 'static', 'fp16'], default='dynamic',
                       help='Type of optimization: dynamic/static quantization (CPU only) or fp16 (GPU/MPS)')
    parser.add_argument('--calibration-size', type=int, default=1000,
                       help='Number of positions for calibration (static only)')
    parser.add_argument('--output', help='Output path for quantized model (optional)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    device, device_str = get_optimal_device()
    print(f"Device: {device_str}")
    
    # Check platform compatibility
    import platform
    if platform.system() == "Windows":
        print("\nRunning on Windows - quantization is fully supported.")
        print("Note: Quantized models run on CPU for best compatibility.\n")
    
    # Determine architecture
    if '20x256' in args.model:
        num_blocks, num_filters = 20, 256
    elif '10x128' in args.model:
        num_blocks, num_filters = 10, 128
    else:
        print("Using default architecture: 20x256")
        num_blocks, num_filters = 20, 256
    
    model = AlphaZeroNetwork.AlphaZeroNet(num_blocks, num_filters)
    weights = torch.load(args.model, map_location=device)
    model.load_state_dict(weights)
    model = optimize_for_device(model, device)
    model.eval()
    
    # Apply optimization based on type
    if args.type == 'fp16':
        if device.type in ['cuda', 'mps']:
            print(f"\nApplying FP16 optimization for {device.type.upper()}...")
            quantized_model = apply_fp16_optimization(model, device)
            suffix = "_fp16"
        else:
            print(f"\nWARNING: FP16 optimization requires GPU/MPS. Current device: {device.type}")
            print("Falling back to dynamic quantization...")
            quantized_model = apply_dynamic_quantization(model)
            suffix = "_dynamic_quantized"
    elif args.type == 'dynamic':
        print("\nApplying dynamic quantization (CPU only)...")
        quantized_model = apply_dynamic_quantization(model)
        suffix = "_dynamic_quantized"
    else:  # static
        print(f"\nApplying static quantization with {args.calibration_size} calibration samples (CPU only)...")
        print("Generating calibration data...")
        calibration_data = create_calibration_dataset(args.calibration_size)
        
        # Use fbgemm backend for Windows/x86
        backend = 'fbgemm'
        quantized_model = apply_static_quantization(model, calibration_data, backend=backend)
        suffix = "_static_quantized"
    
    # Save quantized model
    if args.output:
        output_path = args.output
        # Save using standard torch.save for dynamic quantization
        if args.type == 'dynamic':
            torch.save(quantized_model.state_dict(), output_path)
            print(f"\nQuantized model saved to: {output_path}")
            
            # Print size comparison
            original_size = os.path.getsize(args.model) / (1024 * 1024)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100
            
            print(f"Original size: {original_size:.2f} MB")
            print(f"Quantized size: {quantized_size:.2f} MB")
            print(f"Size reduction: {reduction:.1f}%")
        else:
            # Static quantization uses TorchScript
            output_path = save_quantized_model(quantized_model, args.model, suffix)
    else:
        output_path = save_quantized_model(quantized_model, args.model, suffix)
    
    print(f"\nQuantization complete!")
    print(f"To use the quantized model, load it with:")
    if args.type == 'dynamic':
        print(f"  model = AlphaZeroNetwork.AlphaZeroNet({num_blocks}, {num_filters})")
        print(f"  model = apply_dynamic_quantization(model)")
        print(f"  model.load_state_dict(torch.load('{output_path}'))")
    else:
        print(f"  model = torch.jit.load('{output_path}')")
    
    print(f"\nExpected performance improvement: 2-4x faster inference")


if __name__ == '__main__':
    main()