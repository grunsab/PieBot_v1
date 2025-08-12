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
import multiprocessing as mp
from model_utils import load_model, detect_model_type, clean_state_dict
from device_utils import get_optimal_device

import sys

device, device_str = get_optimal_device()

import MCTS_profiling_speedups_v2 as MCTS
#import searchless_value as MCTS

class TimeManager:
    """Manages time allocation for moves based on game time constraints."""
    
    def __init__(self, base_rollouts=10000, base_time=1.0, threads=64):
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
        # My Macbook Mini M4 Runs around 600 rollouts per second, and my RTX 4080 does about 4000-5000 using MCTS_root_parallel.
        if device.type == "mps":
            self.rollouts_per_second = 600
        else:
            self.rollouts_per_second = 4000  # Average for RTX 4080
        
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
        rollouts = min(200000, rollouts)
        
        return rollouts


class UCIEngine:
    """UCI Protocol handler for AlphaZero chess engine."""
    
    def __init__(self, model_path=None, threads=64, verbose=False, use_multiprocess=False):
        """
        Initialize UCI engine.
        
        Args:
            model_path: Path to the neural network model file
            threads: Number of threads to use for MCTS
            verbose: Whether to output debug information
            use_multiprocess: Whether to use multi-process MCTS
        """
        self.model_path = model_path
        self.device, self.device_str = get_optimal_device() 
        if self.device.type == "mps":
            self.threads = 64
        else:
            self.threads = 64

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
        self.use_multiprocess = use_multiprocess
        
    
    def _create_pienano_from_weights(self, weights):
        """Create PieNano model with correct architecture from weights."""
        # Try to detect architecture from weights
        if isinstance(weights, dict):
            # Check if it has args from training
            if 'args' in weights:
                args = weights['args']
                num_blocks = getattr(args, 'num_blocks', 8)
                num_filters = getattr(args, 'num_filters', 128)
                num_input_planes = 112 if getattr(args, 'use_enhanced_encoder', False) else 16
                policy_hidden_dim = getattr(args, 'policy_hidden_dim', None)
            else:
                # Try to infer from state dict
                state_dict = weights.get('model_state_dict', weights)
                
                # Count residual blocks
                residual_keys = [k for k in state_dict.keys() if 'residual_tower' in k and 'conv1' in k]
                num_blocks = len(set(k.split('.')[1] for k in residual_keys if len(k.split('.')) > 1))
                
                # Get number of filters and input planes from conv_block
                if 'conv_block.0.weight' in state_dict:
                    num_filters = state_dict['conv_block.0.weight'].shape[0]
                    num_input_planes = state_dict['conv_block.0.weight'].shape[1]
                else:
                    # Default values
                    num_filters = 128
                    num_input_planes = 16
                
                # Detect if it's V2 by checking for policy_head.fc1
                policy_hidden_dim = None
                if 'policy_head.fc1.weight' in state_dict:
                    policy_hidden_dim = state_dict['policy_head.fc1.weight'].shape[0]
                
                # Default to 8 blocks if detection failed
                if num_blocks == 0:
                    num_blocks = 8
        else:
            # Default PieNano configuration
            num_blocks = 8
            num_filters = 128
            num_input_planes = 16
            policy_hidden_dim = None
        
        # Create V2 if policy_hidden_dim is detected, otherwise V1
        if policy_hidden_dim is not None:
            return PieNanoNetwork_v2.PieNanoV2(
                num_blocks=num_blocks,
                num_filters=num_filters,
                num_input_planes=num_input_planes,
                policy_hidden_dim=policy_hidden_dim
            )
        else:
            # Fallback to V1 for older models
            return PieNanoNetwork.PieNano(
                num_blocks=num_blocks,
                num_filters=num_filters,
                num_input_planes=num_input_planes
            )
    
    def detect_model_type(self, weights, model_path=None):
        """Detect whether this is an AlphaZeroNet, PieBotNet, PieNano, PieNanoV2, or TitanMini model."""
        # Check if it's a quantized model by filename
        if model_path and ('quantized' in model_path.lower() or 'quant' in model_path.lower()):
            # Try to determine the base model type from filename
            if 'titan' in model_path.lower() or 'titanmini' in model_path.lower():
                return 'TitanMini_Quantized'
            elif 'pienano' in model_path.lower() or 'pie_nano' in model_path.lower():
                return 'PieNano_Quantized'
            elif 'alphazero' in model_path.lower() or 'alpha_zero' in model_path.lower():
                return 'AlphaZeroNet_Quantized'
            # If can't determine from filename, assume AlphaZeroNet
            return 'AlphaZeroNet_Quantized'
        
        if isinstance(weights, dict):
            # Check state dict keys to determine model type
            state_dict = weights.get('model_state_dict', weights)
            
            # TitanMini has specific modules like chess_positional_encoding and relative_position_bias
            has_chess_pos_encoding = any('chess_positional_encoding' in key for key in state_dict.keys())
            has_relative_pos_bias = any('relative_position_bias' in key for key in state_dict.keys())
            has_geglu = any('geglu' in key.lower() for key in state_dict.keys())
            has_cls_token = any('cls_token' in key for key in state_dict.keys())
            
            # PieBotNet has specific modules like positional_encoding and transformer_blocks
            has_positional_encoding = any('positional_encoding' in key for key in state_dict.keys())
            has_transformer = any('transformer_blocks' in key for key in state_dict.keys())
            
            # PieNano models have SE (Squeeze-Excitation) modules and depthwise convolutions
            has_se = any('se.' in key or 'squeeze' in key or 'excitation' in key for key in state_dict.keys())
            has_depthwise = any('depthwise' in key for key in state_dict.keys())
            has_wdl_value = any('value_head.fc2' in key for key in state_dict.keys())
            
            # PieNanoV2 has the improved policy head with fc1 and fc2 in policy_head
            has_improved_policy = any('policy_head.fc1' in key or 'policy_head.fc2' in key for key in state_dict.keys())
            
            if has_chess_pos_encoding or has_relative_pos_bias or has_geglu or has_cls_token:
                return 'TitanMini'
            elif has_positional_encoding or has_transformer:
                return 'PieBotNet'
            elif has_improved_policy and (has_se or has_depthwise):
                return 'PieNanoV2'
            elif (has_se or has_depthwise) and has_wdl_value:
                # Check the shape of value head output to distinguish PieNano
                value_fc2_weight = state_dict.get('value_head.fc2.weight')
                if value_fc2_weight is not None and value_fc2_weight.shape[0] == 3:
                    return 'PieNano'
        return 'AlphaZeroNet'
    
    def load_model(self):
        """Load the neural network model."""
        try:
            if not self.model_path:
                self.model_path = "weights/AlphaZeroNet_20x256.pt"
            
            if not os.path.isabs(self.model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(script_dir, self.model_path)
            else:
                full_path = self.model_path

            if not os.path.exists(full_path):
                print(f"info string ERROR: Model file not found: {full_path}")
                sys.stdout.flush()
                return False
                
            if self.verbose:
                print(f"info string Loading model from: {full_path}")
                sys.stdout.flush()
            
            # Use the shared model loading utility
            self.model, self.device, is_quantized = load_model(full_path)
            
            if is_quantized and self.verbose:
                print(f"info string Note: Quantized model will run on CPU")
                sys.stdout.flush()
            
            self.model.eval()
            
            for param in self.model.parameters():
                param.requires_grad = False
                            
            if self.verbose:
                print("info string Model loaded successfully")
                sys.stdout.flush()

            if self.verbose:
                print("info string Initializing persistent multi-process MCTS engine...")
                sys.stdout.flush()
            self.mcts_engine = MCTS.Root(self.board, self.model)

            return True
            
        except Exception as e:
            print(f"info string ERROR loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return False
            
    def uci(self):
        """Handle 'uci' command."""
        print("id name AlphaZero UCI Engine (Persistent MP)")
        print("id author Rishi Sachdev")
        print("option name Threads type spin default 8 min 1 max 128")
        print("option name Model type string default weights/AlphaZeroNet_20x256.pt")
        print("option name Verbose type check default false")
        print("option name Move Overhead type spin default 30 min 0 max 5000")
        print("option name UseMultiprocess type check default true")
        print("uciok")
        sys.stdout.flush()
        
    def isready(self):
        """Handle 'isready' command."""
        if self.model is None:
            if not self.load_model():
                return
        print("readyok")
        sys.stdout.flush()
        
    def position(self, args):
        """Handle 'position' command."""
        if len(args) < 1:
            return
            
        if args[0] == "startpos":
            self.board = chess.Board()
            moves_start = 1
        elif args[0] == "fen":
            fen = " ".join(args[1:7])
            try:
                self.board = chess.Board(fen)
                moves_start = 7
            except ValueError:
                print(f"ERROR Invalid FEN: {fen}")
                return
        else:
            return
            
        if self.mcts_engine:
            if self.verbose:
                print("info string Resetting MCTS engine for new position")
                sys.stdout.flush()
            
        if len(args) > moves_start and args[moves_start] == "moves":
            for move_str in args[moves_start + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    if self.verbose:
                        print(f"info string Invalid move: {move_str}")
                        
    def search_position(self, rollouts):
        """Search current position using MCTS."""
        if self.verbose:
            print(f"info string Searching position: {self.board.fen()}")
            sys.stdout.flush()
        self.stop_search.clear()
        self.best_move = None
        
        try:
            if self.model is None:
                print("ERROR: No model loaded")
                sys.stdout.flush()
                return
                
            with torch.no_grad():
                start_time = time.time()
                best_move = None
                
                # Always create a fresh MCTS engine for each search to avoid state corruption
                if self.verbose:
                    print("info string Creating fresh MCTS engine for search")
                    sys.stdout.flush()
                
                # Create a new MCTS engine with the current board position
                # This ensures no stale tree state from previous positions
                self.mcts_engine = MCTS.Root(self.board.copy(), self.model)

                self.mcts_engine.parallelRolloutsTotal(self.board.copy(), self.model, rollouts, self.threads)
                actual_rollouts = rollouts
                
                elapsed_time = time.time() - start_time
                
                if elapsed_time > 0:
                    self.time_manager.update_performance(actual_rollouts, elapsed_time)
                
                edge = self.mcts_engine.maxNSelect()
                if not edge:
                    if self.verbose:
                        print("info string Warning: No edge selected from MCTS")
                        sys.stdout.flush()
                    return

                best_move = edge.getMove()
                self.best_move = best_move.uci()
                
                if self.verbose:
                    print(f"info string Selected move: {self.best_move}")
                    
                nps = int(actual_rollouts / elapsed_time) if elapsed_time > 0 else 0
                print(f"info depth {actual_rollouts} nodes {actual_rollouts} nps {nps} pv {best_move}")
                sys.stdout.flush()

                
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    def go(self, args):
        """Handle 'go' command."""
        wtime, btime, winc, binc, movestogo, movetime, infinite = self.parse_go_args(args)
            
        if infinite:
            rollouts = 1_000_000
        elif movetime:
            time_sec = max(0.1, movetime / 1000.0 - self.move_overhead / 1000.0)
            rollouts = int(self.time_manager.get_rollouts_per_second() * time_sec)
        else:
            rollouts = self.time_manager.calculate_rollouts(wtime, btime, winc, binc, movestogo, self.board.turn)
            
        rollouts = max(100, rollouts) # Ensure a minimum number of rollouts
        if self.verbose:
            print(f"info string Thinking for {rollouts} rollouts")
            
        self.search_thread = threading.Thread(target=self.search_position, args=(rollouts,))
        self.search_thread.start()
        self.search_thread.join()
        
        if self.best_move:
            print(f"bestmove {self.best_move}")
        else:
            # Fallback if search fails
            print("WARNING: NO BEST MOVE FOUND")
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                print(f"bestmove {legal_moves[0].uci()}")
        sys.stdout.flush()

    def parse_go_args(self, args):
        wtime = btime = None
        winc = binc = movestogo = 0
        movetime = None
        infinite = "infinite" in args
        
        arg_map = {
            "wtime": lambda x: int(x), "btime": lambda x: int(x),
            "winc": lambda x: int(x), "binc": lambda x: int(x),
            "movestogo": lambda x: int(x), "movetime": lambda x: int(x)
        }
        
        i = 0
        while i < len(args):
            if args[i] in arg_map:
                if i + 1 < len(args):
                    try:
                        val = arg_map[args[i]](args[i+1])
                        if args[i] == "wtime": wtime = val
                        elif args[i] == "btime": btime = val
                        elif args[i] == "winc": winc = val
                        elif args[i] == "binc": binc = val
                        elif args[i] == "movestogo": movestogo = val
                        elif args[i] == "movetime": movetime = val
                    except (ValueError, IndexError):
                        pass
                i += 2
            else:
                i += 1
        return wtime, btime, winc, binc, movestogo, movetime, infinite

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
        if self.use_multiprocess:
            if self.verbose:
                print("info string Shutting down multi-process engine...")
                sys.stdout.flush()
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
                    self.threads = 64
                else:
                    self.threads = 64
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
        elif name == "usemultiprocess":
            self.use_multiprocess = value.lower() in ["true", "yes", "1"]
            
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
                    # Clean up MCTS engine for new game
                    if self.mcts_engine:
                        if self.verbose:
                            print("info string Resetting engine for new game")
                            sys.stdout.flush()
                        import MCTS_root_parallel as MCTS
                        MCTS.Root.cleanup_engine()
                        self.mcts_engine = None
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
    # Set multiprocessing start method to 'spawn' for compatibility across platforms
    # This is crucial for macOS and Windows.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # The start method can only be set once. This is fine.
        pass

    parser = argparse.ArgumentParser(
        description="UCI Protocol wrapper for AlphaZero chess engine"
    )
    parser.add_argument("--model", help="Path to model file", 
                       default="weights/AlphaZeroNet_20x256.pt")
    parser.add_argument("--threads", type=int, help="Number of threads for threaded MCTS", 
                       default=64)
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--multiprocess", action="store_true",
                       help="Use persistent multi-process MCTS")
    
    args = parser.parse_args()
    
    engine = UCIEngine(
        model_path=args.model,
        threads=args.threads,
        verbose=args.verbose,
        use_multiprocess=args.multiprocess
    )
    
    # Ensure cleanup is called on exit
    import atexit
    atexit.register(engine.quit)

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
