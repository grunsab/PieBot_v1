import torch
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model

def check_model_quantization(model_path):
    """Check if a model is truly quantized."""
    
    # Load model
    model, device, is_quantized = load_model(model_path)
    
    print(f"\nModel: {os.path.basename(model_path)}")
    print(f"Is Quantized Flag: {is_quantized}")
    print(f"Device: {device}")
    
    # Check for quantized layers
    quantized_layers = 0
    regular_layers = 0
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'quantized' in module_type.lower() or 'qint' in str(module).lower():
            quantized_layers += 1
            if quantized_layers <= 3:  # Show first few quantized layers
                print(f"  Quantized layer: {name} ({module_type})")
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            regular_layers += 1
    
    print(f"\nQuantized layers: {quantized_layers}")
    print(f"Regular Conv2d/Linear layers: {regular_layers}")
    
    # Check memory usage
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size in memory: {total_size / (1024*1024):.2f} MB")
    
    return is_quantized

if __name__ == "__main__":
    # Test quantized model
    quantized_path = "weights/AlphaZeroNet_20x256_quantized_static.pt"
    if os.path.exists(quantized_path):
        check_model_quantization(quantized_path)
