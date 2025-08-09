"""Test script to verify value outputs from the neural network."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chess
import AlphaZeroNetwork
import encoder

# Load the model
model_file = "weights/AlphaZeroNet_20x256.pt"
model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
weights = torch.load(model_file, map_location='cpu')
model.load_state_dict(weights)
model.eval()

# Disable gradients for inference
with torch.no_grad():
    for param in model.parameters():
        param.requires_grad = False

# Test on starting position
board = chess.Board()
print("Testing on starting position:")
print(board)
print()

# Get neural network output
value, move_probs = encoder.callNeuralNetwork(board, model)
print(f"Value output: {value}")
print(f"Value range should be between -1 and 1, actual: {value}")
print()

# Test on a few moves into the game
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
print("Testing after e4 e5 Nf3:")
print(board)
print()

value, move_probs = encoder.callNeuralNetwork(board, model)
print(f"Value output: {value}")
print(f"Value range should be between -1 and 1, actual: {value}")
print()

# Test on a winning position for white
board = chess.Board("4k3/8/4K3/8/8/8/8/4R3 w - - 0 1")
print("Testing on a winning position for white (endgame):")
print(board)
print()

value, move_probs = encoder.callNeuralNetwork(board, model)
print(f"Value output: {value}")
print(f"Should be positive (white winning): {value}")