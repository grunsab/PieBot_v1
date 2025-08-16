import torch
import sys
import os
import time
import chess

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model
import encoder

def test_model_inference(model_path, num_tests=10):
    """Test inference speed of a model."""
    
    # Load model
    model, device, is_quantized = load_model(model_path)
    
    print(f"\nModel: {model_path}")
    print(f"Device: {device}")
    print(f"Is Quantized: {is_quantized}")
    
    # Create test input
    board = chess.Board()
    input_data = encoder.encodePosition(board)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    # Time inference
    times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Inference time (ms): avg={avg_time:.2f}, min={min_time:.2f}, max={max_time:.2f}")
    
    # Check if output is quantized tensor
    if hasattr(output, '__class__'):
        print(f"Output type: {output.__class__.__name__}")
        if hasattr(output, 'dtype'):
            print(f"Output dtype: {output.dtype}")
    
    return avg_time

if __name__ == "__main__":
    # Test quantized model
    quantized_path = "weights/AlphaZeroNet_20x256_quantized_static.pt"
    if os.path.exists(quantized_path):
        test_model_inference(quantized_path)
    
    # Compare with original if available
    original_path = "weights/AlphaZeroNet_20x256.pt"
    if os.path.exists(original_path):
        print("\n" + "="*50)
        test_model_inference(original_path)
