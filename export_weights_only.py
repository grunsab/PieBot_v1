#!/usr/bin/env python3
"""
Export weights-only version of a checkpoint for deployment.
This creates a much smaller file containing only the model weights.
"""

import torch
import argparse

def export_weights_only(checkpoint_path, output_path):
    """Export just the model weights from a full checkpoint."""
    
    # Load the full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract just the model weights
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        # Assume it's already just weights
        model_weights = checkpoint
    
    # Save just the weights
    torch.save(model_weights, output_path)
    
    # Report file sizes
    import os
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original checkpoint: {original_size:.1f} MB")
    print(f"Weights-only file: {new_size:.1f} MB")
    print(f"Reduction: {(1 - new_size/original_size) * 100:.1f}%")
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Export weights-only version of checkpoint')
    parser.add_argument('checkpoint', help='Path to full checkpoint file')
    parser.add_argument('-o', '--output', help='Output path (default: adds _weights_only suffix)')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.checkpoint.replace('.pt', '_weights_only.pt')
    
    export_weights_only(args.checkpoint, args.output)

if __name__ == "__main__":
    main()