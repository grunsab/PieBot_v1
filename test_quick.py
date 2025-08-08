import chess
from MCTS_multiprocess import MCTS
import torch
import AlphaZeroNetwork
import time

# Load model
model_file = "weights/AlphaZeroNet_20x256.pt"
weights = torch.load(model_file, map_location='cpu')
alphaZeroNet = AlphaZeroNetwork.AlphaZeroNet(20, 256)
alphaZeroNet.load_state_dict(weights)
alphaZeroNet.eval()

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
alphaZeroNet = alphaZeroNet.to(device)

# Create board
board = chess.Board()

# Create MCTS engine
print("Creating MCTS engine...")
mcts_engine = MCTS(model=alphaZeroNet)

# Search for best move
print("Searching for best move...")
start_time = time.time()
best_move = mcts_engine.search(board, num_simulations=10)
elapsed = time.time() - start_time

print(f"Best move: {best_move}")
print(f"Time taken: {elapsed:.2f} seconds")
print(f"NPS: {10/elapsed:.2f}")