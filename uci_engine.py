#!/usr/bin/env python3
"""
UCI Protocol Implementation for AlphaZero Chess Engine

This module implements the Universal Chess Interface (UCI) protocol for the AlphaZero
chess engine, enabling it to communicate with chess GUI applications and online
chess platforms like Lichess.

Time management: Dynamically adjusts the number of MCTS rollouts based on available
time, using the baseline that 800 rollouts take approximately 1 second on 1000 threads.
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
import MCTS_profiling_speedups_v2 as MCTS
from device_utils import get_optimal_device, optimize_for_device
from quantization_utils import load_quantized_model

import sys

class TimeManager:
    """Manages time allocation for moves based on game time constraints."""
    
    def __init__(self, base_rollouts=10000, base_time=1.0, threads=2000):
        """
        Initialize time manager.
        
        Args:
            base_rollouts: Number of rollouts that take base_time seconds
            base_time: Time in seconds for base_rollouts
            threads: Number of threads available
        """
        device, device_str = get_optimal_device()
        self.base_rollouts = base_rollouts
        self.base_time = base_time
        self.threads = threads
        # Adjust for the fact that parallelRollouts does 'threads' rollouts per call
        # So actual time per rollout is different
        # My Macbook Mini M4 Runs around 800 rollouts per second, and my RTX 4080 does about 2800-3500.
        if device.type == "mps":
            self.rollouts_per_second = 800
        else:
            self.rollouts_per_second = 2700  # Average for RTX 4080
        
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
            # No time limit, use default
            return self.base_rollouts
            
        # Convert to seconds
        time_left_sec = time_left / 1000.0
        increment_sec = increment / 1000.0 if increment else 0
        
        # Calculate time to allocate for this move
        if movestogo and movestogo > 0:
            # Time control with moves to go
            time_per_move = time_left_sec / movestogo + increment_sec * 0.8
        else:
            # Sudden death or unknown moves to go
            # Use a fraction of remaining time, scaling down as time decreases
            if time_left_sec > 60:
                time_fraction = 0.04  # Use 4% when we have plenty of time
            elif time_left_sec > 10:
                time_fraction = 0.06  # Use 6% when time is getting lower
            else:
                time_fraction = 0.10  # Use 10% when very low on time
                
            time_per_move = time_left_sec * time_fraction + increment_sec * 0.8
        
        # Add safety buffer for overhead (communication, model loading, etc)
        # Reserve at least 100ms for overhead
        time_per_move = time_per_move * 0.9 - 0.1
        
        # Ensure minimum thinking time
        time_per_move = max(0.05, time_per_move)
        
        # Ensure we don't use more than 40% of remaining time
        time_per_move = min(time_per_move, time_left_sec * 0.4)
        
        # Calculate rollouts based on available time
        rps = self.get_rollouts_per_second()
        rollouts = int(rps * time_per_move)
        
        # Ensure minimum rollouts for quality
        rollouts = max(100, rollouts)
        
        # Cap maximum rollouts to prevent excessive thinking
        rollouts = min(100000, rollouts)
        
        return rollouts


class UCIEngine:
    """UCI Protocol handler for AlphaZero chess engine."""
    
    def __init__(self, model_path=None, threads=128, verbose=False):
        """
        Initialize UCI engine.
        
        Args:
            model_path: Path to the neural network model file
            threads: Number of threads to use for MCTS
            verbose: Whether to output debug information
        """
        self.model_path = model_path
        self.device, self.device_str = get_optimal_device() 
        if self.device.type == "mps":
            self.threads = threads * 8
        else:
            self.threads = threads * 32

        self.verbose = verbose
        self.board = chess.Board()
        self.model = None
        self.mcts_engine = None
        self.device = None
        self.time_manager = TimeManager(threads=self.threads)
        self.search_thread = None
        self.stop_search = threading.Event()
        self.best_move = None
        self.move_overhead = 30  # Default move overhead in milliseconds
        
    def load_model(self):
        """Load the neural network model."""
        try:
            if not self.model_path:
                # Use default model if none specified
                self.model_path = "AlphaZeroNet_20x256_distributed.pt"
            
            # Try to find the model file
            if not os.path.isabs(self.model_path):
                # Try relative to current directory first
                if os.path.exists(self.model_path):
                    full_path = self.model_path
                else:
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    full_path = os.path.join(script_dir, self.model_path)
                    if not os.path.exists(full_path):
                        print(f"info string ERROR: Model file not found: {self.model_path}")
                        print(f"info string Tried: {os.path.abspath(self.model_path)}")
                        print(f"info string Tried: {full_path}")
                        sys.stdout.flush()
                        return False
            else:
                full_path = self.model_path
                if not os.path.exists(full_path):
                    print(f"info string ERROR: Model file not found: {full_path}")
                    sys.stdout.flush()
                    return False
                
            self.device, device_str = get_optimal_device()
            if self.verbose:
                print(f"info string Loading model from: {full_path}")
                print(f"info string Loading model on device: {device_str}")
                sys.stdout.flush()
                
            self.model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
            weights = torch.load(full_path, map_location=self.device)
            
            # Handle different model formats
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                # FP16 model format
                self.model.load_state_dict(weights['model_state_dict'])
                if weights.get('model_type') == 'fp16':
                    self.model = self.model.half()
                    if self.verbose:
                        print(f"info string Loaded FP16 model on {device_str}")
                        sys.stdout.flush()
            else:
                # Regular model format
                self.model.load_state_dict(weights)
            
            self.model = optimize_for_device(self.model, self.device)
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                            
            if self.verbose:
                print(f"info string Model loaded successfully")
                sys.stdout.flush()
            return True
            
        except Exception as e:
            print(f"info string ERROR loading model: {str(e)}")
            sys.stdout.flush()
            return False
            
    def uci(self):
        """Handle 'uci' command."""
        print("id name AlphaZero UCI Engine")
        print("id author Rishi Sachdev")
        print("option name Threads type spin default 8 min 1 max 128")
        print("option name Model type string default AlphaZeroNet_20x256_distributed.pt")
        print("option name Verbose type check default false")
        print("option name Move Overhead type spin default 30 min 0 max 5000")
        print("uciok")
        sys.stdout.flush()
        
    def isready(self):
        """Handle 'isready' command."""
        if self.model is None:
            if not self.load_model():
                # Model loading failed, but we still need to respond
                pass
        print("readyok")
        sys.stdout.flush()
        
    def position(self, args):
        """
        Handle 'position' command.
        
        Args:
            args: List of position arguments
        """
        if len(args) < 1:
            return
            
        if args[0] == "startpos":
            self.board = chess.Board()
            moves_start = 1
        elif args[0] == "fen":
            if len(args) < 7:
                return
            fen = " ".join(args[1:7])
            self.board = chess.Board(fen)
            moves_start = 7
        else:
            return
            
        # Apply moves if provided
        if len(args) > moves_start and args[moves_start] == "moves":
            for move_str in args[moves_start + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except:
                    if self.verbose:
                        print(f"info string Invalid move: {move_str}")
                        
    def search_position(self, rollouts):
        """
        Search current position using MCTS.
        
        Args:
            rollouts: Number of rollouts to perform
        """
        self.stop_search.clear()
        self.best_move = None
        
        try:
            # Check if we have a model
            if self.model is None:
                print(f"info string ERROR: No model loaded")
                sys.stdout.flush()
                return
                
            with torch.no_grad():
                self.mcts_engine = MCTS.Root(self.board, self.model)
                num_iterations = max(1, rollouts // self.threads)
                remainder = rollouts % self.threads
                
                start_time = time.time()
                
                for i in range(num_iterations):
                    if self.stop_search.is_set():
                        break
                    self.mcts_engine.parallelRollouts(self.board.copy(), self.model, self.threads)
                    
                    # Output progress periodically
                    if (i + 1) % max(1, num_iterations // 10) == 0 or i == num_iterations - 1:
                        current_rollouts = (i + 1) * self.threads
                        current_edge = self.mcts_engine.maxNSelect()
                        if current_edge:
                            move = current_edge.getMove()
                            score = int(current_edge.getQ() * 1000 - 500)
                            elapsed = time.time() - start_time
                            nps = int(current_rollouts / elapsed) if elapsed > 0 else 0
                            print(f"info depth {current_rollouts} score cp {score} nodes {current_rollouts} nps {nps} pv {move}")
                            sys.stdout.flush()
                
                # Handle remainder rollouts if any
                if remainder > 0 and not self.stop_search.is_set():
                    self.mcts_engine.parallelRollouts(self.board.copy(), self.model, remainder)
                
                elapsed_time = time.time() - start_time
                actual_rollouts = num_iterations * self.threads + (remainder if remainder > 0 else 0)
                print(actual_rollouts, "rollouts completed in", elapsed_time, "seconds")
                
                # Update time manager with actual performance
                if elapsed_time > 0:
                    self.time_manager.update_performance(actual_rollouts, elapsed_time)
                edge = self.mcts_engine.maxNSelect()

                bestmove = edge.getMove()        
                self.best_move = bestmove.uci()
                print(f"info string Completed {actual_rollouts} rollouts in {elapsed_time:.2f}s")
                print(f"info string Rollouts per second: {actual_rollouts/elapsed_time:.1f}")
                print(self.mcts_engine.getStatisticsString())
                sys.stdout.flush()
                        
                # Clean up
                self.mcts_engine.cleanup()
                MCTS.clear_caches()
                MCTS.clear_pools()
                
        except Exception as e:
            print(f"info string Error during search: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
                
    def go(self, args):
        """
        Handle 'go' command.
        
        Args:
            args: List of go arguments
        """
        # Parse time control parameters
        wtime = None
        btime = None
        winc = 0
        binc = 0
        movestogo = 0
        movetime = None
        infinite = False
        
        i = 0
        while i < len(args):
            if args[i] == "wtime" and i + 1 < len(args):
                wtime = int(args[i + 1])
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                btime = int(args[i + 1])
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                winc = int(args[i + 1])
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                binc = int(args[i + 1])
                i += 2
            elif args[i] == "movestogo" and i + 1 < len(args):
                movestogo = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1])
                i += 2
            elif args[i] == "infinite":
                infinite = True
                i += 1
            else:
                i += 1
                
        # Calculate rollouts based on time
        if infinite:
            rollouts = 100000  # High number for analysis
        elif movetime:
            # Fixed time per move
            time_sec = movetime / 1000.0
            # Account for move overhead
            time_sec = max(0.1, time_sec - self.move_overhead / 1000.0)
            rollouts = int(self.time_manager.get_rollouts_per_second() * time_sec * 0.95)
        else:
            # Calculate based on game time
            # Adjust time for move overhead
            if wtime is not None:
                wtime = max(100, wtime - self.move_overhead)
            if btime is not None:
                btime = max(100, btime - self.move_overhead)
            rollouts = self.time_manager.calculate_rollouts(
                wtime, btime, winc, binc, movestogo, self.board.turn
            )
            
        if self.verbose:
            print(f"info string Calculating with {rollouts} rollouts")
            
        # Start search in separate thread
        self.search_thread = threading.Thread(
            target=self.search_position, args=(rollouts,)
        )
        self.search_thread.start()
        
        # Wait for search to complete
        self.search_thread.join()
        
        # Output best move
        if self.best_move:
            print(f"bestmove {self.best_move}")
        else:
            # Fallback: pick first legal move that doesn't cause threefold repetition
            print("FALLING BACK TO RANDOM MOVE BECAUSE SEARCH IS BROKEN")
            legal_moves = list(self.board.legal_moves)
            fallback_move = None
            
            for move in legal_moves:
                test_board = self.board.copy()
                test_board.push(move)
                if not test_board.can_claim_threefold_repetition():
                    fallback_move = move
                    break
            
            # If all moves lead to threefold repetition, use the first legal move
            if fallback_move is None and legal_moves:
                fallback_move = legal_moves[0]
                
            if fallback_move:
                print(f"bestmove {fallback_move}")
        sys.stdout.flush()
                
    def stop(self):
        """Handle 'stop' command."""
        self.stop_search.set()
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join()
        if self.best_move:
            print(f"bestmove {self.best_move}")
            sys.stdout.flush()
            
    def quit(self):
        """Handle 'quit' command."""
        self.stop_search.set()
        sys.exit(0)
        
    def setoption(self, args):
        """
        Handle 'setoption' command.
        
        Args:
            args: List of option arguments
        """
        if len(args) < 4 or args[0] != "name":
            return
            
        # Find where "value" appears in args
        value_idx = -1
        for i, arg in enumerate(args):
            if arg == "value":
                value_idx = i
                break
                
        if value_idx == -1 or value_idx < 2:
            return
            
        # Reconstruct name and value allowing for multi-word names
        name = " ".join(args[1:value_idx]).lower()
        value = " ".join(args[value_idx + 1:])
        
        if name == "threads":
            try:
                if self.device.type == "mps":
                    self.threads = int(value) * 8
                else:
                    self.threads = int(value) * 32
                self.time_manager.threads = self.threads
            except:
                pass
        elif name == "model":
            self.model_path = value
            self.model = None  # Force reload on next isready
        elif name == "verbose":
            self.verbose = value.lower() in ["true", "yes", "1"]
        elif name == "move overhead":
            try:
                self.move_overhead = int(value)
            except:
                pass
            
    def run(self):
        """Main UCI protocol loop."""
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                    
                parts = line.split()
                command = parts[0].lower()
                
                if command == "uci":
                    self.uci()
                elif command == "isready":
                    self.isready()
                elif command == "position":
                    self.position(parts[1:])
                elif command == "go":
                    self.go(parts[1:])
                elif command == "stop":
                    self.stop()
                elif command == "quit":
                    self.quit()
                elif command == "setoption":
                    self.setoption(parts[1:])
                elif command == "ucinewgame":
                    # Reset board for new game
                    self.board = chess.Board()
                elif self.verbose:
                    print(f"info string Unknown command: {command}")
                    sys.stdout.flush()
                    
            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()


def main():
    """Main entry point."""
    # Ensure stdout is line-buffered for proper UCI communication
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(
        description="UCI Protocol wrapper for AlphaZero chess engine"
    )
    parser.add_argument("--model", help="Path to model file", 
                       default="AlphaZeroNet_20x256_distributed.pt")
    parser.add_argument("--threads", type=int, help="Number of threads", 
                       default=128)
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    engine = UCIEngine(
        model_path=args.model,
        threads=args.threads,
        verbose=args.verbose
    )
    
    try:
        engine.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"info string Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()