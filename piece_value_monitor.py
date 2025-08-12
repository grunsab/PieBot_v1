"""
Monitor piece values learned by the AlphaZero network.
This module analyzes how the network values different pieces during training.
"""

import torch
import chess
import numpy as np
from typing import Dict, List, Tuple
import encoder

class PieceValueMonitor:
    """
    Monitors the implicit piece values learned by the neural network.
    Estimates values by comparing positions with and without specific pieces.
    """
    
    # Standard piece values for reference (in pawns)
    STANDARD_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0
    }
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: The AlphaZero network to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.piece_names = {
            chess.PAWN: 'Pawn',
            chess.KNIGHT: 'Knight',
            chess.BISHOP: 'Bishop',
            chess.ROOK: 'Rook',
            chess.QUEEN: 'Queen'
        }
    
    def create_test_positions(self) -> List[Tuple[chess.Board, chess.Board, int]]:
        """
        Create pairs of test positions that differ by exactly one piece.
        Returns list of (board_with_piece, board_without_piece, piece_type) tuples.
        """
        test_positions = []
        
        # Test position 1: Starting position variations
        base_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Remove different pieces to test their values
        test_fens = [
            # Remove white pieces
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", chess.ROOK),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", chess.QUEEN),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1", chess.KNIGHT),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1", chess.BISHOP),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1", chess.PAWN),
        ]
        
        for fen_with, fen_without, piece_type in test_fens:
            board_with = chess.Board(fen_with)
            board_without = chess.Board(fen_without)
            test_positions.append((board_with, board_without, piece_type))
        
        # Test position 2: Middle game positions
        middle_game_base = "r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 0 8"
        
        middle_game_tests = [
            # Remove pieces from middle game
            ("r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 0 8",
             "r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNBQK21R w KQkq - 0 8", chess.ROOK),
            ("r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 0 8",
             "r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNB1K2R w KQkq - 0 8", chess.QUEEN),
            ("r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 0 8",
             "r1bqk2r/pp2nppp/2n1p3/3p4/1b1P4/3BP3/PPP2PPP/RNBQK2R w KQkq - 0 8", chess.KNIGHT),
        ]
        
        for fen_with, fen_without, piece_type in middle_game_tests:
            board_with = chess.Board(fen_with)
            board_without = chess.Board(fen_without)
            test_positions.append((board_with, board_without, piece_type))
        
        return test_positions
    
    def estimate_piece_values(self) -> Dict[int, float]:
        """
        Estimate piece values by comparing network evaluations of positions
        with and without specific pieces.
        
        Returns:
            Dictionary mapping piece types to estimated values (in pawns)
        """
        self.model.eval()
        piece_values = {piece: [] for piece in self.piece_names.keys()}
        
        test_positions = self.create_test_positions()
        
        with torch.no_grad():
            for board_with, board_without, piece_type in test_positions:
                # Encode positions
                pos_with = encoder.encode(board_with)
                pos_without = encoder.encode(board_without)
                
                # Convert to tensors
                pos_with_tensor = torch.tensor(pos_with, dtype=torch.float32).unsqueeze(0).to(self.device)
                pos_without_tensor = torch.tensor(pos_without, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get network evaluations
                value_with, _ = self.model(pos_with_tensor)
                value_without, _ = self.model(pos_without_tensor)
                
                # Calculate difference (this is the piece's contribution to evaluation)
                value_diff = (value_with - value_without).item()
                
                # Store the value difference
                piece_values[piece_type].append(value_diff)
        
        # Average the estimates for each piece type
        estimated_values = {}
        for piece_type, values in piece_values.items():
            if values:
                # Average and normalize relative to pawn value
                avg_value = np.mean(values)
                estimated_values[piece_type] = avg_value
        
        # Normalize values relative to pawn if we have pawn values
        if chess.PAWN in estimated_values and estimated_values[chess.PAWN] != 0:
            pawn_value = abs(estimated_values[chess.PAWN])
            for piece_type in estimated_values:
                estimated_values[piece_type] = abs(estimated_values[piece_type]) / pawn_value
        else:
            # If no good pawn reference, scale based on queen = 9
            if chess.QUEEN in estimated_values and estimated_values[chess.QUEEN] != 0:
                queen_value = abs(estimated_values[chess.QUEEN])
                scale_factor = 9.0 / queen_value
                for piece_type in estimated_values:
                    estimated_values[piece_type] = abs(estimated_values[piece_type]) * scale_factor
        
        return estimated_values
    
    def get_value_convergence_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics showing how close the learned values are to standard values.
        
        Returns:
            Dictionary with convergence metrics
        """
        estimated = self.estimate_piece_values()
        
        metrics = {}
        total_error = 0
        count = 0
        
        for piece_type, standard_value in self.STANDARD_VALUES.items():
            if piece_type in estimated:
                est_value = estimated[piece_type]
                error = abs(est_value - standard_value)
                relative_error = error / standard_value
                
                piece_name = self.piece_names[piece_type]
                metrics[f'{piece_name}_value'] = est_value
                metrics[f'{piece_name}_error'] = error
                metrics[f'{piece_name}_relative_error'] = relative_error
                
                total_error += error
                count += 1
        
        if count > 0:
            metrics['mean_absolute_error'] = total_error / count
            metrics['convergence_score'] = 1.0 / (1.0 + metrics['mean_absolute_error'])
        
        return metrics
    
    def print_piece_value_report(self):
        """Print a detailed report of learned piece values."""
        estimated = self.estimate_piece_values()
        metrics = self.get_value_convergence_metrics()
        
        print("\n" + "="*60)
        print("PIECE VALUE ANALYSIS")
        print("="*60)
        
        print("\n{:<12} {:<12} {:<12} {:<12}".format(
            "Piece", "Learned", "Standard", "Error"
        ))
        print("-"*48)
        
        for piece_type, piece_name in self.piece_names.items():
            if piece_type in estimated:
                learned = estimated[piece_type]
                standard = self.STANDARD_VALUES[piece_type]
                error = abs(learned - standard)
                
                print("{:<12} {:<12.2f} {:<12.2f} {:<12.2f}".format(
                    piece_name, learned, standard, error
                ))
        
        print("-"*48)
        if 'mean_absolute_error' in metrics:
            print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.3f}")
            print(f"Convergence Score: {metrics['convergence_score']:.3f}")
        
        print("="*60 + "\n")
    
    def log_to_tensorboard(self, writer, epoch):
        """
        Log piece value metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            epoch: Current training epoch
        """
        metrics = self.get_value_convergence_metrics()
        
        for key, value in metrics.items():
            writer.add_scalar(f'PieceValues/{key}', value, epoch)
        
        # Log individual piece values
        estimated = self.estimate_piece_values()
        for piece_type, value in estimated.items():
            piece_name = self.piece_names[piece_type]
            writer.add_scalar(f'PieceValues/Raw/{piece_name}', value, epoch)