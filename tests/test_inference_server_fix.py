"""Test script to verify the inference server fix for empty values/policies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chess
import AlphaZeroNetwork
import multiprocessing as mp
from inference_server import InferenceServer, InferenceRequest, InferenceResult
import time

def test_inference_server():
    """Test the inference server with a simple batch."""
    
    # Load model
    model_file = "weights/AlphaZeroNet_20x256.pt"
    model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    
    # Check if weights file exists, if not use a smaller model for testing
    try:
        weights = torch.load(model_file, map_location='cpu')
        model.load_state_dict(weights)
        print(f"Loaded model from {model_file}")
    except FileNotFoundError:
        print(f"Model file {model_file} not found, using random weights for testing")
        # Model will use random weights for testing
    
    model.eval()
    
    # Create device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create inference server
    server = InferenceServer(model, device, batch_size=4, timeout_ms=100)
    
    # Create test requests
    boards = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After e4
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),  # After e4 e5
    ]
    
    print("\nTesting batch processing with {} unique positions".format(len(boards)))
    
    # Create request tuples
    request_tuples = []
    result_queues = []
    
    for i, board in enumerate(boards):
        req = InferenceRequest(request_id=i, board_fen=board.fen())
        result_queue = mp.Queue()
        request_tuples.append((req, result_queue))
        result_queues.append(result_queue)
    
    # Process batch
    print("\nProcessing batch...")
    server.process_batch(request_tuples)
    
    # Check results
    print("\nChecking results...")
    for i, result_queue in enumerate(result_queues):
        try:
            result = result_queue.get(timeout=1.0)
            print(f"\nRequest {i}:")
            print(f"  Board: {boards[i].fen()}")
            print(f"  Value: {result.value}")
            print(f"  Value type: {type(result.value)}")
            print(f"  Move probabilities shape: {result.move_probabilities.shape}")
            print(f"  Non-zero probabilities: {(result.move_probabilities > 0).sum()}")
            print(f"  Sum of probabilities: {result.move_probabilities.sum():.4f}")
            
            # Check if values are empty or invalid
            if result.value is None or (isinstance(result.value, float) and abs(result.value) < 1e-10 and result.move_probabilities.sum() < 1e-10):
                print("  WARNING: Value and/or policies appear to be empty!")
            else:
                print("  SUCCESS: Value and policies are populated!")
                
        except Exception as e:
            print(f"Error getting result for request {i}: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_inference_server()