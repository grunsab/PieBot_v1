#!/usr/bin/env python3
"""
UCI Protocol Implementation for AlphaZero Chess Engine with CUDA Optimization

This module implements the Universal Chess Interface (UCI) protocol for the AlphaZero
chess engine with CUDA optimizations, enabling it to communicate with chess GUI 
applications and online chess platforms like Lichess.

Time management: Dynamically adjusts the number of MCTS rollouts based on available
time. The CUDA version can handle significantly more rollouts per second.
"""

import sys
import os
import chess
import torch
import argparse
import time
import threading
from queue import Queue
import AlphaZeroNetwork
from device_utils import get_optimal_device, optimize_for_device
from quantization_utils import load_quantized_model

# Import the appropriate MCTS implementation
try:
    import MCTS_cuda_optimized as MCTS
    USING_CUDA_MCTS = True
    print("info string Using CUDA-optimized MCTS implementation", flush=True)
except ImportError:
    try:
        import MCTS_advanced_optimizations as MCTS
        USING_CUDA_MCTS = False
        print("info string Using advanced MCTS implementation (CUDA not available)", flush=True)
    except ImportError:
        import MCTS
        USING_CUDA_MCTS = False
        print("info string Using original MCTS implementation", flush=True)


class TimeManager:
    """Manages time allocation for moves based on game time constraints."""
    
    def __init__(self, base_rollouts=1500, base_time=1.0, threads=12):
        """
        Initialize time manager.
        
        Args:
            base_rollouts: Number of rollouts that take base_time seconds
            base_time: Time in seconds for base_rollouts
            threads: Number of threads available
        """
        self.base_rollouts = base_rollouts
        self.base_time = base_time
        self.threads = threads
        
        # Adjust base values for CUDA implementation (much faster)
        if USING_CUDA_MCTS:
            # CUDA version is typically 3-5x faster
            self.rollouts_per_second = base_rollouts * 4 / base_time
        else:
            self.rollouts_per_second = base_rollouts / base_time
        
        # Track actual performance
        self.measured_rollouts_per_second = None
        self.measurement_count = 0
        
    def update_performance(self, rollouts, elapsed_time):
        """Update measured performance based on actual timing."""
        if elapsed_time > 0.1:  # Only update for meaningful measurements
            new_rps = rollouts / elapsed_time
            if self.measured_rollouts_per_second is None:
                self.measured_rollouts_per_second = new_rps
            else:
                # Exponential moving average
                alpha = 0.3
                self.measured_rollouts_per_second = alpha * new_rps + (1 - alpha) * self.measured_rollouts_per_second
            self.measurement_count += 1
            
    def get_rollouts_per_second(self):
        """Get the best estimate of rollouts per second."""
        if self.measured_rollouts_per_second and self.measurement_count >= 3:
            return self.measured_rollouts_per_second
        return self.rollouts_per_second
        
    def calculate_rollouts(self, wtime, btime, winc, binc, movestogo, turn):
        """
        Calculate optimal number of rollouts based on time constraints.
        
        Args:
            wtime: White time in milliseconds
            btime: Black time in milliseconds
            winc: White increment in milliseconds
            binc: Black increment in milliseconds
            movestogo: Moves to go until time control (0 if sudden death)
            turn: True if white to move, False if black
            
        Returns:
            Number of rollouts to perform
        """
        # Get time for current player
        time_left = wtime if turn else btime
        increment = winc if turn else binc
        
        if time_left is None:
            # No time limit, use default (more for CUDA)
            return self.base_rollouts * (4 if USING_CUDA_MCTS else 1)
            
        # Convert to seconds
        time_left_sec = time_left / 1000.0
        increment_sec = increment / 1000.0 if increment else 0
        
        # Calculate time to allocate for this move
        if movestogo and movestogo > 0:
            # Time control with moves to go
            time_per_move = time_left_sec / movestogo + increment_sec * 0.8
        else:
            # Sudden death or unknown moves to go
            # Estimate 40 more moves in the game
            moves_remaining = 40
            time_per_move = time_left_sec / moves_remaining + increment_sec * 0.8
            
        # Safety margins
        time_per_move = min(time_per_move, time_left_sec * 0.1)  # Never use more than 10% of remaining time
        time_per_move = max(time_per_move, 0.1)  # Always think for at least 0.1 seconds
        
        # Calculate rollouts based on time available
        rps = self.get_rollouts_per_second()
        rollouts = int(time_per_move * rps)
        
        # Adjust for parallel rollouts
        rollouts = max(rollouts // self.threads, 1) * self.threads
        
        # Set reasonable bounds (higher for CUDA)
        if USING_CUDA_MCTS:
            min_rollouts = 100
            max_rollouts = 50000
        else:
            min_rollouts = 10
            max_rollouts = 10000
            
        rollouts = max(min_rollouts, min(rollouts, max_rollouts))
        
        return rollouts


class UCIEngine:
    """UCI protocol implementation for AlphaZero chess engine."""
    
    def __init__(self, model_path=None, threads=12, verbose=False):
        """
        Initialize the UCI engine.
        
        Args:
            model_path: Path to the neural network model file
            threads: Number of threads to use for MCTS
            verbose: Whether to print debug information
        """
        self.model_path = model_path
        self.threads = threads
        self.verbose = verbose
        self.board = chess.Board()
        self.model = None
        self.device = None
        self.time_manager = TimeManager(threads=threads)
        self.move_count = 0
        
        # Performance tracking
        self.total_nodes = 0
        self.total_time = 0
        
    def load_model(self):
        """Load the neural network model."""
        if not self.model_path:
            print("info string No model path specified", flush=True)
            return False
            
        try:
            # Load model with device optimization
            self.device, device_str = get_optimal_device()
            print(f"info string Loading model on {device_str}", flush=True)
            
            # Check if CUDA extensions are available
            if USING_CUDA_MCTS:
                if hasattr(MCTS, 'CPP_AVAILABLE') and MCTS.CPP_AVAILABLE:
                    print("info string C++ extensions loaded", flush=True)
                if hasattr(MCTS, 'CUDA_AVAILABLE') and MCTS.CUDA_AVAILABLE:
                    print("info string CUDA extensions loaded", flush=True)
                    print(f"info string Batch size: {MCTS.BATCH_SIZE}", flush=True)
            
            # Handle different model formats
            weights = torch.load(self.model_path, map_location='cpu')
            
            # Check if it's a quantized model
            is_static_quantized = (isinstance(weights, dict) and 
                                 'quant.scale' in weights and 
                                 any(k.startswith('base_model.') for k in weights.keys()))
            
            if is_static_quantized or (isinstance(weights, dict) and weights.get('model_type') == 'static_quantized'):
                # Static quantized model
                try:
                    self.model = torch.jit.load(self.model_path, map_location=self.device)
                    self.model.eval()
                    print("info string Loaded static quantized model", flush=True)
                except:
                    # Fall back to quantization utils
                    self.model = load_quantized_model(self.model_path, self.device, 20, 256)
                    print("info string Loaded quantized model", flush=True)
            else:
                # Regular model
                self.model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
                
                if isinstance(weights, dict) and 'model_state_dict' in weights:
                    # FP16 model format
                    self.model.load_state_dict(weights['model_state_dict'])
                    if weights.get('model_type') == 'fp16':
                        self.model = self.model.half()
                        print("info string Loaded FP16 model", flush=True)
                else:
                    # Regular model format
                    self.model.load_state_dict(weights)
                
                self.model = optimize_for_device(self.model, self.device)
                self.model.eval()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
                
            print("info string Model loaded successfully", flush=True)
            return True
            
        except Exception as e:
            print(f"info string Error loading model: {e}", flush=True)
            return False
    
    def search(self, wtime=None, btime=None, winc=None, binc=None, movestogo=None):
        """
        Search for the best move using MCTS.
        
        Args:
            wtime: White time in milliseconds
            btime: Black time in milliseconds
            winc: White increment in milliseconds
            binc: Black increment in milliseconds
            movestogo: Moves to go until time control
            
        Returns:
            Best move in UCI format
        """
        if not self.model:
            return None
            
        # Calculate number of rollouts based on time
        rollouts = self.time_manager.calculate_rollouts(
            wtime, btime, winc, binc, movestogo, self.board.turn
        )
        
        if self.verbose:
            print(f"info string Calculated rollouts: {rollouts}", flush=True)
            time_allocated = rollouts / self.time_manager.get_rollouts_per_second()
            print(f"info string Time allocated: {time_allocated:.2f}s", flush=True)
            
        start_time = time.perf_counter()
        
        # Run MCTS
        with torch.no_grad():
            root = MCTS.Root(self.board, self.model)
            
            # Progress reporting
            report_interval = max(rollouts // 10, 10)
            
            for i in range(rollouts):
                root.parallelRollouts(self.board.copy(), self.model, self.threads)
                
                # Report progress
                if i % report_interval == 0 and i > 0:
                    elapsed = time.perf_counter() - start_time
                    nps = root.getN() / elapsed
                    pv = self.get_pv(root)
                    print(f"info depth {i} nodes {int(root.getN())} nps {int(nps)} pv {pv}", flush=True)
                    
        # Get final statistics
        elapsed_time = time.perf_counter() - start_time
        total_nodes = root.getN()
        nps = total_nodes / elapsed_time
        
        # Update time manager with actual performance
        self.time_manager.update_performance(total_nodes, elapsed_time)
        
        # Update global statistics
        self.total_nodes += total_nodes
        self.total_time += elapsed_time
        self.move_count += 1
        
        # Get best move
        edge = root.maxNSelect()
        if not edge:
            return None
            
        best_move = edge.getMove()
        
        # Final info
        pv = self.get_pv(root)
        avg_nps = self.total_nodes / self.total_time if self.total_time > 0 else nps
        
        print(f"info depth {rollouts} nodes {int(total_nodes)} time {int(elapsed_time * 1000)} nps {int(nps)} pv {pv}", flush=True)
        
        if self.verbose:
            print(f"info string Move {self.move_count}: {best_move}, NPS: {nps:.0f}, Avg NPS: {avg_nps:.0f}", flush=True)
            print(root.getStatisticsString(), flush=True)
            
            # Cache statistics if available
            if USING_CUDA_MCTS and hasattr(MCTS, 'position_cache'):
                print(f"info string Cache hits: Positions={len(MCTS.position_cache)}, Moves={len(MCTS.legal_move_cache)}", flush=True)
        
        # Cleanup
        if hasattr(root, 'cleanup'):
            root.cleanup()
            
        # Clear caches periodically
        if USING_CUDA_MCTS and self.move_count % 50 == 0:
            if hasattr(MCTS, 'clear_caches'):
                MCTS.clear_caches()
                print("info string Cleared caches", flush=True)
            if hasattr(MCTS, 'clear_batch_queue'):
                MCTS.clear_batch_queue()
        
        return best_move.uci()
    
    def get_pv(self, root):
        """Get principal variation from root."""
        pv = []
        node = root
        board = self.board.copy()
        
        for _ in range(10):  # Limit PV length
            if not hasattr(node, 'maxNSelect'):
                break
                
            edge = node.maxNSelect()
            if not edge:
                break
                
            move = edge.getMove()
            pv.append(move.uci())
            board.push(move)
            
            # Get child node if available
            if hasattr(edge, 'child') and edge.child:
                node = edge.child
            else:
                break
                
        return ' '.join(pv)
    
    def run(self):
        """Main UCI protocol loop."""
        print("AlphaZero Chess Engine (CUDA-Optimized)", flush=True)
        print(f"Threads: {self.threads}", flush=True)
        if USING_CUDA_MCTS:
            print("CUDA optimizations enabled", flush=True)
        print("Type 'uci' to start", flush=True)
        
        while True:
            try:
                command = input().strip()
                
                if command == "uci":
                    print("id name AlphaZero CUDA", flush=True)
                    print("id author PyTorch AlphaZero (CUDA-optimized)", flush=True)
                    print("option name Threads type spin default 12 min 1 max 128", flush=True)
                    print("option name ModelPath type string default weights/AlphaZeroNet_20x256.pt", flush=True)
                    print("option name Verbose type check default false", flush=True)
                    print("uciok", flush=True)
                    
                elif command == "isready":
                    if not self.model and self.model_path:
                        self.load_model()
                    print("readyok", flush=True)
                    
                elif command.startswith("setoption"):
                    parts = command.split()
                    if "Threads" in parts:
                        idx = parts.index("value")
                        self.threads = int(parts[idx + 1])
                        self.time_manager.threads = self.threads
                        print(f"info string Threads set to {self.threads}", flush=True)
                    elif "ModelPath" in parts:
                        idx = parts.index("value")
                        self.model_path = " ".join(parts[idx + 1:])
                        print(f"info string Model path set to {self.model_path}", flush=True)
                    elif "Verbose" in parts:
                        idx = parts.index("value")
                        self.verbose = parts[idx + 1].lower() == "true"
                        print(f"info string Verbose set to {self.verbose}", flush=True)
                        
                elif command == "ucinewgame":
                    self.board = chess.Board()
                    self.move_count = 0
                    self.total_nodes = 0
                    self.total_time = 0
                    # Clear caches for new game
                    if USING_CUDA_MCTS:
                        if hasattr(MCTS, 'clear_caches'):
                            MCTS.clear_caches()
                        if hasattr(MCTS, 'clear_batch_queue'):
                            MCTS.clear_batch_queue()
                    print("info string New game started", flush=True)
                    
                elif command.startswith("position"):
                    parts = command.split()
                    if "startpos" in parts:
                        self.board = chess.Board()
                        moves_idx = parts.index("moves") if "moves" in parts else len(parts)
                    else:
                        fen_idx = parts.index("fen")
                        moves_idx = parts.index("moves") if "moves" in parts else len(parts)
                        fen = " ".join(parts[fen_idx + 1:moves_idx])
                        self.board = chess.Board(fen)
                        
                    if "moves" in parts:
                        moves_idx = parts.index("moves")
                        for move_uci in parts[moves_idx + 1:]:
                            self.board.push_uci(move_uci)
                            
                elif command.startswith("go"):
                    parts = command.split()
                    wtime = None
                    btime = None
                    winc = None
                    binc = None
                    movestogo = None
                    
                    # Parse time controls
                    for i, part in enumerate(parts):
                        if part == "wtime" and i + 1 < len(parts):
                            wtime = int(parts[i + 1])
                        elif part == "btime" and i + 1 < len(parts):
                            btime = int(parts[i + 1])
                        elif part == "winc" and i + 1 < len(parts):
                            winc = int(parts[i + 1])
                        elif part == "binc" and i + 1 < len(parts):
                            binc = int(parts[i + 1])
                        elif part == "movestogo" and i + 1 < len(parts):
                            movestogo = int(parts[i + 1])
                            
                    # Search for best move
                    best_move = self.search(wtime, btime, winc, binc, movestogo)
                    if best_move:
                        print(f"bestmove {best_move}", flush=True)
                    else:
                        # Random legal move as fallback
                        legal_moves = list(self.board.legal_moves)
                        if legal_moves:
                            print(f"bestmove {legal_moves[0].uci()}", flush=True)
                            
                elif command == "quit":
                    # Cleanup before exit
                    if USING_CUDA_MCTS:
                        if hasattr(MCTS, 'clear_caches'):
                            MCTS.clear_caches()
                        if hasattr(MCTS, 'clear_batch_queue'):
                            MCTS.clear_batch_queue()
                    break
                    
            except Exception as e:
                print(f"info string Error: {e}", flush=True)
                if self.verbose:
                    import traceback
                    traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='UCI Chess Engine with CUDA Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a CUDA-optimized version of the AlphaZero UCI engine.

Performance improvements:
- C++ extensions for hot paths
- CUDA kernels for tree operations
- Aggressive neural network batching
- GPU-accelerated parallel search

For best performance:
- Use high thread counts (16-32)
- Ensure CUDA extensions are built
- Use on Windows/Linux with NVIDIA GPU

Build extensions:
  python setup_extensions.py build_ext --inplace

Example usage with cutechess:
  cutechess-cli -engine cmd="python UCI_engine_cuda.py --model weights/model.pt --threads 32" \
                -engine cmd=stockfish -each tc=60+1 -games 100
        """
    )
    parser.add_argument('--model', default='weights/AlphaZeroNet_20x256.pt',
                       help='Path to neural network model file')
    parser.add_argument('--threads', type=int, default=12,
                       help='Number of threads for MCTS (default: 12)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    engine = UCIEngine(
        model_path=args.model,
        threads=args.threads,
        verbose=args.verbose
    )
    
    try:
        engine.run()
    except KeyboardInterrupt:
        print("info string Interrupted by user", flush=True)
    except Exception as e:
        print(f"info string Fatal error: {e}", flush=True)
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()