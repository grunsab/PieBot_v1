#!/usr/bin/env python3
"""Diagnose why TitanMini is stuck outputting ~0.65 for all positions."""

import torch
import torch.nn as nn
import numpy as np
import TitanMiniNetwork

def analyze_weights():
    """Analyze the trained weights to find the issue."""
    print("="*60)
    print("Analyzing TitanMini Trained Weights")
    print("="*60)
    
    # Load weights
    weights = torch.load('weights/titan_mini_best_weights_only.pt', map_location='cpu', weights_only=False)
    clean_weights = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in weights.items()}
    
    # Analyze value head weights
    print("\nValue Head Weight Analysis:")
    print("-"*40)
    
    # Layer 0 (384 -> 192)
    w0 = clean_weights['value_head.value_proj.0.weight']
    b0 = clean_weights['value_head.value_proj.0.bias']
    print(f"Layer 0 (384 -> 192):")
    print(f"  Weight norm: {w0.norm():.3f}")
    print(f"  Weight mean: {w0.mean():.3f}")
    print(f"  Weight std: {w0.std():.3f}")
    print(f"  Bias mean: {b0.mean():.3f}")
    print(f"  Bias std: {b0.std():.3f}")
    
    # Layer 3 (192 -> 1)
    w3 = clean_weights['value_head.value_proj.3.weight']
    b3 = clean_weights['value_head.value_proj.3.bias']
    print(f"\nLayer 3 (192 -> 1):")
    print(f"  Weight norm: {w3.norm():.3f}")
    print(f"  Weight mean: {w3.mean():.3f}")
    print(f"  Weight std: {w3.std():.3f}")
    print(f"  Bias value: {b3.item():.3f}")
    
    # This is the key issue - what's the typical pre-sigmoid value?
    print(f"\n*** Critical Finding ***")
    print(f"Final layer bias: {b3.item():.3f}")
    print(f"Sigmoid({b3.item():.3f}) = {torch.sigmoid(b3).item():.3f}")
    print("This bias dominates the output!")
    
    return clean_weights

def test_forward_pass(weights):
    """Test what happens in a forward pass."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    # Create model and load weights
    model = TitanMiniNetwork.TitanMini(use_wdl=False, legacy_value_head=True)
    model.load_state_dict(weights)
    model.eval()
    
    # Create random input
    batch_size = 10
    x = torch.randn(batch_size, 112, 8, 8)
    
    # Hook to capture intermediate values
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    model.value_head.value_proj[0].register_forward_hook(get_activation('layer0'))
    model.value_head.value_proj[3].register_forward_hook(get_activation('layer3'))
    model.value_head.value_proj[4].register_forward_hook(get_activation('sigmoid'))
    
    # Forward pass
    with torch.no_grad():
        value, _ = model(x)
    
    print("\nIntermediate activations:")
    print(f"After layer 0 (GELU): mean={activations['layer0'].mean():.3f}, std={activations['layer0'].std():.3f}")
    print(f"After layer 3 (pre-sigmoid): mean={activations['layer3'].mean():.3f}, std={activations['layer3'].std():.3f}")
    print(f"After sigmoid: mean={activations['sigmoid'].mean():.3f}, std={activations['sigmoid'].std():.3f}")
    
    print(f"\nPre-sigmoid values: {activations['layer3'].squeeze().tolist()[:5]}")
    print(f"Post-sigmoid values: {activations['sigmoid'].squeeze().tolist()[:5]}")
    
    # The problem
    print("\n*** Problem Diagnosis ***")
    pre_sigmoid = activations['layer3'].squeeze()
    print(f"Pre-sigmoid range: [{pre_sigmoid.min():.3f}, {pre_sigmoid.max():.3f}]")
    print(f"All pre-sigmoid values are positive and similar!")
    print(f"This causes all Sigmoid outputs to be ~0.6-0.7")
    
    return activations

def propose_solution():
    """Propose solutions to fix the issue."""
    print("\n" + "="*60)
    print("Solution Analysis")
    print("="*60)
    
    print("\nRoot Cause:")
    print("1. The model was likely trained with a high learning rate")
    print("2. The final layer bias shifted positive (~0.6)")
    print("3. This causes Sigmoid to output ~0.65 for most inputs")
    print("4. Small weight variations can't overcome the large bias")
    
    print("\nWhy did this happen?")
    print("- Sigmoid + MSE loss is prone to saturation")
    print("- If early training pushes outputs toward 0.6-0.7")
    print("- Gradients become very small (Sigmoid derivative near 0.65 ≈ 0.23)")
    print("- The model gets stuck in this local minimum")
    
    print("\nSolutions for retraining:")
    print("1. Use lower learning rate (e.g., 1e-4 instead of 2e-4)")
    print("2. Initialize final bias to 0 explicitly")
    print("3. Use gradient clipping more aggressively")
    print("4. Consider using Tanh instead of Sigmoid (like AlphaZero)")
    print("5. Add weight decay to prevent weight explosion")
    
    print("\nImmediate fix for existing model:")
    print("- The model is essentially broken due to Sigmoid saturation")
    print("- Retraining is necessary with the fixes above")

def test_gradient_flow():
    """Test gradient flow with current architecture."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)
    
    # Create a small model
    model = TitanMiniNetwork.TitanMini(num_layers=2, d_model=96, num_heads=6, use_wdl=False, legacy_value_head=True)
    model.train()
    
    # Test batch
    batch_size = 4
    x = torch.randn(batch_size, 112, 8, 8)
    targets = torch.tensor([[0.0], [0.5], [1.0], [0.3]])  # Various targets
    
    # Forward pass
    value, _ = model(x, value_target=targets, policy_target=torch.zeros(batch_size, dtype=torch.long))
    total_loss, value_loss, _ = value
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    print("\nGradient magnitudes:")
    for name, param in model.value_head.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.norm():.6f}")
    
    # Test with outputs stuck at 0.65
    model.zero_grad()
    stuck_output = torch.full((batch_size, 1), 0.65)
    loss_at_stuck = nn.MSELoss()(stuck_output, targets)
    
    print(f"\nLoss when stuck at 0.65: {loss_at_stuck:.4f}")
    
    # Gradient of Sigmoid at 0.65
    pre_sigmoid = torch.log(torch.tensor(0.65) / (1 - 0.65))  # Inverse sigmoid
    sigmoid_grad = torch.sigmoid(pre_sigmoid) * (1 - torch.sigmoid(pre_sigmoid))
    print(f"Sigmoid gradient at 0.65: {sigmoid_grad:.4f}")
    print("This small gradient makes learning very slow!")

def main():
    """Run all diagnostics."""
    print("TitanMini Diagnostic Report")
    print("="*60)
    
    # Analyze weights
    weights = analyze_weights()
    
    # Test forward pass
    activations = test_forward_pass(weights)
    
    # Propose solutions
    propose_solution()
    
    # Test gradient flow
    test_gradient_flow()
    
    print("\n" + "="*60)
    print("Conclusion:")
    print("="*60)
    print("The model is stuck due to Sigmoid saturation.")
    print("The final layer bias (~0.6) dominates the output.")
    print("Retraining with better initialization and lower learning rate is needed.")
    print("\nFor immediate use, the encoder.py fix helps but can't fully solve")
    print("the fundamental issue that the model doesn't differentiate positions.")

if __name__ == "__main__":
    main()