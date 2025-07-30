#!/usr/bin/env python3
"""
Test suite to compare original MCTS vs Ultra-Performance MCTS.

This script plays games between the two MCTS implementations to ensure
the Ultra-Performance version maintains similar playing strength while
achieving higher NPS (nodes per second).
"""

import chess
import torch
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from AlphaZeroNetwork import AlphaZeroNet
import MCTS
from MCTS_ultra_performance import UltraPerformanceMCTSEngine
import device_utils

class MCTSComparison:
    """Handles comparison between MCTS implementations"""
    
    def __init__(self, model_path, device, num_games=20, rollouts=5000, verbose=True):
        self.model_path = model_path
        self.device = device
        self.num_games = num_games
        self.rollouts = rollouts
        self.verbose = verbose
        
        # Load model
        self.model = self._load_model()
        
        # Initialize engines
        self.ultra_engine = UltraPerformanceMCTSEngine(
            self.model,
            device=device,
            batch_size=512,
            num_workers=20,
            verbose=False
        )
        self.ultra_engine.start()
        
        # Results tracking
        self.results = {
            'ultra_wins': 0,
            'original_wins': 0,
            'draws': 0,
            'games': []
        }
        
    def _load_model(self):
        """Load the neural network model"""
        if self.verbose:
            print(f"Loading model: {self.model_path}")
            
        # Infer model architecture from filename
        if '20x256' in self.model_path:
            num_blocks, num_filters = 20, 256
        elif '10x128' in self.model_path:
            num_blocks, num_filters = 10, 128
        else:
            num_blocks, num_filters = 20, 256
            
        model = AlphaZeroNet(num_blocks, num_filters)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def play_game(self, ultra_plays_white):
        """Play a single game between the two engines"""
        board = chess.Board()
        moves = []
        
        game_info = {
            'moves': [],
            'ultra_white': ultra_plays_white,
            'result': None,
            'ultra_times': [],
            'original_times': [],
            'ultra_nps': [],
            'original_nps': []
        }
        
        if self.verbose:
            print(f"\nGame {len(self.results['games']) + 1}: ", end="")
            print(f"Ultra-Performance plays {'White' if ultra_plays_white else 'Black'}")
        
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            
            # Determine which engine plays
            is_ultra_turn = (board.turn == chess.WHITE) == ultra_plays_white
            
            if is_ultra_turn:
                # Ultra-Performance MCTS
                start_time = time.time()
                move = self.ultra_engine.search(board, self.rollouts)
                elapsed = time.time() - start_time
                
                if move:
                    nps = self.rollouts / elapsed if elapsed > 0 else 0
                    game_info['ultra_times'].append(elapsed)
                    game_info['ultra_nps'].append(nps)
                    
                    if self.verbose and move_count % 10 == 0:
                        print(f"  Move {move_count}: Ultra {move} ({nps:.0f} NPS)")
            else:
                # Original MCTS
                start_time = time.time()
                root = MCTS.Root(board, self.model)
                
                # Run rollouts
                for _ in range(self.rollouts // 20):  # 20 threads per iteration
                    root.parallelRollouts(board.copy(), self.model, 20)
                
                elapsed = time.time() - start_time
                
                # Select best move
                best_edge = root.maxNSelect()
                if best_edge:
                    move = best_edge.getMove()
                    nps = self.rollouts / elapsed if elapsed > 0 else 0
                    game_info['original_times'].append(elapsed)
                    game_info['original_nps'].append(nps)
                    
                    if self.verbose and move_count % 10 == 0:
                        print(f"  Move {move_count}: Original {move} ({nps:.0f} NPS)")
                else:
                    move = None
                    
                # Clean up
                root.cleanup()
            
            if move and move in board.legal_moves:
                board.push(move)
                game_info['moves'].append(move.uci())
            else:
                break
        
        # Game result
        result = board.result()
        game_info['result'] = result
        
        if result == "1-0":  # White wins
            if ultra_plays_white:
                self.results['ultra_wins'] += 1
                winner = "Ultra-Performance"
            else:
                self.results['original_wins'] += 1
                winner = "Original"
        elif result == "0-1":  # Black wins
            if not ultra_plays_white:
                self.results['ultra_wins'] += 1
                winner = "Ultra-Performance"
            else:
                self.results['original_wins'] += 1
                winner = "Original"
        else:  # Draw
            self.results['draws'] += 1
            winner = "Draw"
            
        if self.verbose:
            print(f"  Result: {result} ({winner})")
            print(f"  Moves: {len(game_info['moves'])}")
            if game_info['ultra_nps']:
                print(f"  Ultra avg NPS: {np.mean(game_info['ultra_nps']):,.0f}")
            if game_info['original_nps']:
                print(f"  Original avg NPS: {np.mean(game_info['original_nps']):,.0f}")
        
        self.results['games'].append(game_info)
        return result
    
    def run_comparison(self):
        """Run the full comparison"""
        print(f"Running {self.num_games} games with {self.rollouts} rollouts per move")
        print(f"Device: {self.device}")
        print("="*60)
        
        # Play games alternating colors
        for i in range(self.num_games):
            ultra_plays_white = (i % 2 == 0)
            self.play_game(ultra_plays_white)
        
        # Summary
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Total games: {self.num_games}")
        print(f"Ultra-Performance wins: {self.results['ultra_wins']} ({self.results['ultra_wins']/self.num_games*100:.1f}%)")
        print(f"Original MCTS wins: {self.results['original_wins']} ({self.results['original_wins']/self.num_games*100:.1f}%)")
        print(f"Draws: {self.results['draws']} ({self.results['draws']/self.num_games*100:.1f}%)")
        
        # Performance comparison
        all_ultra_nps = []
        all_original_nps = []
        for game in self.results['games']:
            all_ultra_nps.extend(game['ultra_nps'])
            all_original_nps.extend(game['original_nps'])
        
        if all_ultra_nps and all_original_nps:
            ultra_avg_nps = np.mean(all_ultra_nps)
            original_avg_nps = np.mean(all_original_nps)
            speedup = ultra_avg_nps / original_avg_nps
            
            print("\nPERFORMANCE COMPARISON")
            print("-"*30)
            print(f"Ultra-Performance avg NPS: {ultra_avg_nps:,.0f}")
            print(f"Original MCTS avg NPS: {original_avg_nps:,.0f}")
            print(f"Speedup: {speedup:.1f}x")
        
        # Stop engine
        self.ultra_engine.stop()
        
        return self.results

def main():
    parser = argparse.ArgumentParser(
        description='Compare Original MCTS vs Ultra-Performance MCTS'
    )
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--rollouts', type=int, default=5000, help='Rollouts per move')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.device)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Run comparison
    comparison = MCTSComparison(
        args.model,
        device,
        num_games=args.games,
        rollouts=args.rollouts,
        verbose=not args.quiet
    )
    
    results = comparison.run_comparison()
    
    # Check if performance is maintained
    win_rate = results['ultra_wins'] / args.games
    if win_rate >= 0.45:  # At least 45% win rate (accounting for draws)
        print("\n✅ Ultra-Performance MCTS maintains playing strength!")
    else:
        print("\n⚠️  Ultra-Performance MCTS shows reduced playing strength")

if __name__ == "__main__":
    main()