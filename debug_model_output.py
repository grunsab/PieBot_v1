#!/usr/bin/env python3
"""Debug script to check model output format."""

import torch
import chess
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoder
from playchess import load_model_multi_gpu

# Load model
model_file = "weights/AlphaZeroNet_20x256.pt"
print(f"Loading {model_file}...")

# First, let's check what's in the model file
checkpoint = torch.load(model_file, map_location='cpu')
print("\nModel file contents:")
if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print(f"  {key}")
    if 'model_type' in checkpoint:
        print(f"  Model type: {checkpoint['model_type']}")
else:
    print("  Model is not a dictionary, it's a direct state dict")

# Load model properly
models, devices = load_model_multi_gpu(model_file, None)
model = models[0]
device = devices[0]

print(f"\nModel class: {type(model)}")
print(f"Model device: {device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Test with a simple position
board = chess.Board()
position, mask = encoder.encodePositionForInference(board)

# Convert to tensors
pos_tensor = torch.from_numpy(position).unsqueeze(0).to(device, dtype=next(model.parameters()).dtype)
mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device, dtype=next(model.parameters()).dtype)
mask_flat = mask_tensor.view(1, -1)

print(f"\nInput shapes:")
print(f"  Position: {pos_tensor.shape}")
print(f"  Mask: {mask_flat.shape}")

# Get model output
with torch.no_grad():
    model.eval()
    outputs = model(pos_tensor, policyMask=mask_flat)
    print(f"\nModel outputs: {len(outputs)} tensors")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape {out.shape}, dtype {out.dtype}")
    
    value, policy = outputs[0], outputs[1]
    
    print(f"\nPolicy tensor details:")
    print(f"  Shape: {policy.shape}")
    print(f"  Size: {policy.numel()}")
    print(f"  Min: {policy.min().item():.4f}")
    print(f"  Max: {policy.max().item():.4f}")
    print(f"  Sum: {policy.sum().item():.4f}")
    
    # Check if it's already masked/compressed
    if policy.shape[-1] < 100:
        print(f"\nWARNING: Policy output seems compressed! Expected 4608, got {policy.shape[-1]}")
        print("This might be a special model that outputs only legal move probabilities.")
    
    # Check model architecture
    print(f"\nChecking model layers...")
    for name, module in model.named_modules():
        if 'fc1' in name or 'policy' in name.lower():
            print(f"  {name}: {module}")