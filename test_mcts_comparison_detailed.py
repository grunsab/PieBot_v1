#!/usr/bin/env python3
"""
Detailed test suite to compare original MCTS vs Ultra-Performance MCTS with extensive logging.
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

class DetailedMCTSComparison:
    """Handles comparison between MCTS implementations with detailed logging"""
    
    def __init__(self, model_path, device, num_games=10, rollouts=1000, verbose=True):
        self.model_path = model_path
        self.device = device
        self.num_games = num_games
        self.rollouts = rollouts
        self.verbose = verbose
        
        # Load model
        print(f"\n{'='*80}")
        print("LOADING MODEL")
        print(f"{'='*80}")
        self.model = self._load_model()
        
        # Initialize engines
        print(f"\n{'='*80}")
        print("INITIALIZING ENGINES")
        print(f"{'='*80}")
        
        print("Initializing Ultra-Performance MCTS Engine...")
        self.ultra_engine = UltraPerformanceMCTSEngine(
            self.model,
            device=device,
            batch_size=256,
            num_workers=16,
            verbose=False
        )
        self.ultra_engine.start()
        print("✓ Ultra-Performance engine ready")
        
        # Results tracking
        self.results = {
            'ultra_wins': 0,
            'original_wins': 0,
            'draws': 0,
            'games': []
        }
        
    def _load_model(self):
        """Load the neural network model"""
        print(f"Loading model from: {self.model_path}")
        
        # Infer model architecture from filename
        if '20x256' in self.model_path:
            num_blocks, num_filters = 20, 256
        elif '10x128' in self.model_path:
            num_blocks, num_filters = 10, 128
        else:
            num_blocks, num_filters = 20, 256
            
        print(f"Model architecture: {num_blocks} blocks × {num_filters} filters")
        
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
            
        print("✓ Model loaded successfully")
        return model
    
    def analyze_mcts_state(self, implementation, board, move, elapsed, rollouts, is_ultra=True):
        """Analyze and log MCTS state after move selection"""
        if not self.verbose:
            return
            
        print(f"\n  {implementation} Analysis:")
        print(f"  Selected move: {move}")
        print(f"  Time: {elapsed:.2f}s, NPS: {rollouts/elapsed:.0f}")
        
        if is_ultra:
            # Analyze Ultra-Performance tree
            root_hash = board.fen()
            if root_hash in self.ultra_engine.node_lookup:
                root_idx = self.ultra_engine.node_lookup[root_hash]
                root = self.ultra_engine.nodes[root_idx]
                
                print(f"  Root visits: {root.visits}")
                
                if root_idx in self.ultra_engine.children:
                    children = self.ultra_engine.children[root_idx]
                    
                    # Count visited children
                    visited_children = 0
                    child_visits = []
                    
                    for child_idx in children:
                        if child_idx < len(self.ultra_engine.nodes):
                            child = self.ultra_engine.nodes[child_idx]
                            if child.visits > 0:
                                visited_children += 1
                                child_visits.append((child.move, child.visits))
                    
                    print(f"  Explored {visited_children}/{len(children)} legal moves")
                    
                    # Show top 5 moves
                    child_visits.sort(key=lambda x: x[1], reverse=True)
                    print("  Top 5 moves by visits:")
                    for i, (mv, visits) in enumerate(child_visits[:5]):
                        print(f"    {i+1}. {mv}: {visits} visits")
    
    def play_game(self, game_num, ultra_plays_white):
        """Play a single game between the two engines with detailed logging"""
        print(f"\n{'='*80}")
        print(f"GAME {game_num}")
        print(f"{'='*80}")
        print(f"Ultra-Performance plays: {'WHITE ⚪' if ultra_plays_white else 'BLACK ⚫'}")
        print(f"Original MCTS plays: {'BLACK ⚫' if ultra_plays_white else 'WHITE ⚪'}")
        print(f"Rollouts per move: {self.rollouts}")
        
        board = chess.Board()
        moves = []
        
        game_info = {
            'moves': [],
            'ultra_white': ultra_plays_white,
            'result': None,
            'ultra_times': [],
            'original_times': [],
            'ultra_nps': [],
            'original_nps': [],
            'positions': []
        }
        
        move_count = 0
        print("\nGame progress:")
        
        while not board.is_game_over() and move_count < 200:  # Prevent infinite games
            move_count += 1
            
            # Log current position every 10 moves
            if move_count % 10 == 1:
                print(f"\n--- Move {move_count} ---")
                print(f"FEN: {board.fen()}")
            
            # Determine which engine plays
            is_ultra_turn = (board.turn == chess.WHITE) == ultra_plays_white
            
            if is_ultra_turn:
                # Ultra-Performance MCTS
                engine_name = "Ultra-Performance"
                start_time = time.time()
                
                # Get move
                move = self.ultra_engine.search(board, self.rollouts)
                
                elapsed = time.time() - start_time
                
                if move:
                    nps = self.rollouts / elapsed if elapsed > 0 else 0
                    game_info['ultra_times'].append(elapsed)
                    game_info['ultra_nps'].append(nps)
                    
                    if move_count % 5 == 0 or move_count <= 10:
                        self.analyze_mcts_state(engine_name, board, move, elapsed, self.rollouts, is_ultra=True)
                        
            else:
                # Original MCTS
                engine_name = "Original"
                start_time = time.time()
                
                root = MCTS.Root(board, self.model)
                
                # Run rollouts
                batch_size = 8
                for _ in range(self.rollouts // batch_size):
                    root.parallelRollouts(board.copy(), self.model, batch_size)
                
                elapsed = time.time() - start_time
                
                # Select best move
                best_edge = root.maxNSelect()
                if best_edge:
                    move = best_edge.getMove()
                    nps = self.rollouts / elapsed if elapsed > 0 else 0
                    game_info['original_times'].append(elapsed)
                    game_info['original_nps'].append(nps)
                    
                    if move_count % 5 == 0 or move_count <= 10:
                        print(f"\n  {engine_name} Analysis:")
                        print(f"  Selected move: {move}")
                        print(f"  Time: {elapsed:.2f}s, NPS: {nps:.0f}")
                        print(f"  Root visits: {root.getN()}")
                        
                        # Show top moves
                        edges = [(e, e.getN()) for e in root.edges if e.getN() > 0]
                        edges.sort(key=lambda x: x[1], reverse=True)
                        print("  Top 5 moves by visits:")
                        for i, (edge, visits) in enumerate(edges[:5]):
                            print(f"    {i+1}. {edge.getMove()}: {visits:.0f} visits")
                else:
                    move = None
                    
                # Clean up
                root.cleanup()
            
            if move and move in board.legal_moves:
                board.push(move)
                game_info['moves'].append(move.uci())
                
                # Log the move
                if move_count <= 20 or move_count % 10 == 0:
                    symbol = "⚪" if board.turn == chess.BLACK else "⚫"
                    print(f"{symbol} Move {move_count}: {engine_name} plays {move}")
            else:
                print(f"\n⚠️  {engine_name} returned invalid move: {move}")
                break
        
        # Game result
        result = board.result()
        game_info['result'] = result
        
        print(f"\n{'='*60}")
        print("GAME RESULT")
        print(f"{'='*60}")
        print(f"Result: {result}")
        print(f"Total moves: {len(game_info['moves'])}")
        
        if result == "1-0":  # White wins
            if ultra_plays_white:
                self.results['ultra_wins'] += 1
                winner = "Ultra-Performance (White)"
                print("🏆 Winner: Ultra-Performance MCTS")
            else:
                self.results['original_wins'] += 1
                winner = "Original (White)"
                print("🏆 Winner: Original MCTS")
        elif result == "0-1":  # Black wins
            if not ultra_plays_white:
                self.results['ultra_wins'] += 1
                winner = "Ultra-Performance (Black)"
                print("🏆 Winner: Ultra-Performance MCTS")
            else:
                self.results['original_wins'] += 1
                winner = "Original (Black)"
                print("🏆 Winner: Original MCTS")
        else:  # Draw
            self.results['draws'] += 1
            winner = "Draw"
            print("🤝 Game ended in a draw")
            
        # Performance statistics
        if game_info['ultra_nps'] and game_info['original_nps']:
            print(f"\nPerformance Statistics:")
            print(f"Ultra-Performance avg: {np.mean(game_info['ultra_nps']):,.0f} NPS")
            print(f"Original MCTS avg: {np.mean(game_info['original_nps']):,.0f} NPS")
            print(f"Speedup: {np.mean(game_info['ultra_nps']) / np.mean(game_info['original_nps']):.1f}x")
        
        # Final position
        print(f"\nFinal position:")
        print(board)
        
        self.results['games'].append(game_info)
        return result
    
    def run_comparison(self):
        """Run the full comparison with detailed logging"""
        print(f"\n{'='*80}")
        print("STARTING COMPARISON TEST")
        print(f"{'='*80}")
        print(f"Total games: {self.num_games}")
        print(f"Rollouts per move: {self.rollouts}")
        print(f"Device: {self.device}")
        print(f"Batch size: 256")
        print(f"Workers: 16")
        
        # Play games alternating colors
        for i in range(self.num_games):
            ultra_plays_white = (i % 2 == 0)
            self.play_game(i + 1, ultra_plays_white)
            
            # Running tally
            print(f"\n📊 Running Score: Ultra {self.results['ultra_wins']} - "
                  f"{self.results['original_wins']} Original (Draws: {self.results['draws']})")
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Total games played: {self.num_games}")
        print(f"\n🏆 Ultra-Performance wins: {self.results['ultra_wins']} "
              f"({self.results['ultra_wins']/self.num_games*100:.1f}%)")
        print(f"🏆 Original MCTS wins: {self.results['original_wins']} "
              f"({self.results['original_wins']/self.num_games*100:.1f}%)")
        print(f"🤝 Draws: {self.results['draws']} "
              f"({self.results['draws']/self.num_games*100:.1f}%)")
        
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
            
            print(f"\n{'='*60}")
            print("PERFORMANCE ANALYSIS")
            print(f"{'='*60}")
            print(f"Ultra-Performance average: {ultra_avg_nps:,.0f} NPS")
            print(f"Original MCTS average: {original_avg_nps:,.0f} NPS")
            print(f"Average speedup: {speedup:.1f}x")
        
        # Stop engine
        self.ultra_engine.stop()
        
        # Final verdict
        win_rate = self.results['ultra_wins'] / self.num_games
        print(f"\n{'='*80}")
        if win_rate >= 0.45:
            print("✅ SUCCESS: Ultra-Performance MCTS maintains competitive playing strength!")
        elif win_rate >= 0.30:
            print("⚠️  PARTIAL SUCCESS: Ultra-Performance MCTS shows reasonable strength")
        else:
            print("❌ FAILURE: Ultra-Performance MCTS shows significantly reduced strength")
        print(f"{'='*80}")
        
        return self.results

def main():
    parser = argparse.ArgumentParser(
        description='Detailed comparison of Original MCTS vs Ultra-Performance MCTS'
    )
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--rollouts', type=int, default=1000, help='Rollouts per move')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    
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
    comparison = DetailedMCTSComparison(
        args.model,
        device,
        num_games=args.games,
        rollouts=args.rollouts,
        verbose=True
    )
    
    results = comparison.run_comparison()

if __name__ == "__main__":
    main()