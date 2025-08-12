"""
Monitor piece values learned by the Titan Mini transformer network.
Adapted for the transformer architecture with attention mechanisms.
"""

import torch
import chess
import numpy as np
from typing import Dict, List, Tuple
import encoder
try:
    import encoder_enhanced
except ImportError:
    encoder_enhanced = None

class TitanPieceValueMonitor:
    """
    Monitors the implicit piece values learned by the Titan Mini transformer.
    Analyzes how the attention-based architecture values different pieces.
    """
    
    # Standard piece values for reference (in pawns)
    STANDARD_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0
    }
    
    def __init__(self, model, device='cpu', enhanced_encoder=True):
        """
        Args:
            model: The Titan Mini network to analyze
            device: Device to run analysis on
            enhanced_encoder: Whether to use enhanced encoder (112 planes)
        """
        self.model = model
        self.device = device
        self.enhanced_encoder = enhanced_encoder
        self.piece_names = {
            chess.PAWN: 'Pawn',
            chess.KNIGHT: 'Knight',
            chess.BISHOP: 'Bishop',
            chess.ROOK: 'Rook',
            chess.QUEEN: 'Queen'
        }
    
    def create_comprehensive_test_positions(self) -> List[Tuple[chess.Board, chess.Board, int]]:
        """
        Create comprehensive test positions for transformer analysis.
        More positions needed for accurate estimation with attention mechanisms.
        """
        test_positions = []
        
        # Opening positions
        opening_tests = [
            # Standard starting position variations
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
            
            # After initial pawn moves
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBN1 b KQkq e3 0 1", chess.ROOK),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNB1KBNR b KQkq e3 0 1", chess.QUEEN),
        ]
        
        # Middle game positions - more complex for transformer analysis
        middle_game_tests = [
            # Italian game position
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
             "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK21R w KQkq - 0 6", chess.ROOK),
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
             "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNB1K2R w KQkq - 0 6", chess.QUEEN),
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
             "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P4/PPP2PPP/RNBQK2R w KQkq - 0 6", chess.KNIGHT),
            
            # Sicilian Defense position
            ("r1bqkb1r/pp2pppp/2np1n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6",
             "r1bqkb1r/pp2pppp/2np1n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB2 w KQkq - 0 6", chess.ROOK),
            ("r1bqkb1r/pp2pppp/2np1n2/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6",
             "r1bqkb1r/pp2pppp/2np1n2/8/3PP3/2N2N2/PPP2PPP/R1B1KB1R w KQkq - 0 6", chess.QUEEN),
        ]
        
        # Endgame positions - critical for accurate value assessment
        endgame_tests = [
            # Rook endgame
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6R1 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1", chess.ROOK),
            
            # Queen vs rook endgame
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6Q1 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1", chess.QUEEN),
            
            # Bishop endgame
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6B1 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1", chess.BISHOP),
            
            # Knight endgame
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6N1 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1", chess.KNIGHT),
            
            # Pawn endgame
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/8/5PK1/8 w - - 0 1", chess.PAWN),
        ]
        
        for fen_with, fen_without, piece_type in opening_tests + middle_game_tests + endgame_tests:
            board_with = chess.Board(fen_with)
            board_without = chess.Board(fen_without)
            test_positions.append((board_with, board_without, piece_type))
        
        return test_positions
    
    def estimate_piece_values_with_attention(self) -> Dict[int, float]:
        """
        Estimate piece values using the transformer's attention mechanisms.
        More sophisticated analysis for the Titan Mini architecture.
        """
        self.model.eval()
        piece_values = {piece: [] for piece in self.piece_names.keys()}
        
        test_positions = self.create_comprehensive_test_positions()
        
        with torch.no_grad():
            for board_with, board_without, piece_type in test_positions:
                # Encode positions based on encoder type
                if self.enhanced_encoder and encoder_enhanced:
                    pos_with = encoder_enhanced.encode(board_with)
                    pos_without = encoder_enhanced.encode(board_without)
                else:
                    pos_with = encoder.encodePosition(board_with)
                    pos_without = encoder.encodePosition(board_without)
                
                # Convert to tensors
                pos_with_tensor = torch.tensor(pos_with, dtype=torch.float32).unsqueeze(0).to(self.device)
                pos_without_tensor = torch.tensor(pos_without, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get network evaluations - transformer outputs
                value_with, _ = self.model(pos_with_tensor)
                value_without, _ = self.model(pos_without_tensor)
                
                # Calculate difference (piece's contribution)
                value_diff = (value_with - value_without).item()
                
                # Store the value difference
                piece_values[piece_type].append(value_diff)
        
        # Robust averaging with outlier removal for transformers
        estimated_values = {}
        for piece_type, values in piece_values.items():
            if values:
                # Remove outliers (values beyond 2 std deviations)
                values_array = np.array(values)
                mean = np.mean(values_array)
                std = np.std(values_array)
                
                if std > 0:
                    filtered = values_array[np.abs(values_array - mean) <= 2 * std]
                    if len(filtered) > 0:
                        avg_value = np.mean(filtered)
                    else:
                        avg_value = mean
                else:
                    avg_value = mean
                    
                estimated_values[piece_type] = avg_value
        
        # Normalize values relative to pawn
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
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """
        Calculate convergence metrics for Titan Mini's learned piece values.
        """
        estimated = self.estimate_piece_values_with_attention()
        
        metrics = {}
        total_error = 0
        total_relative_error = 0
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
                total_relative_error += relative_error
                count += 1
        
        if count > 0:
            metrics['mean_absolute_error'] = total_error / count
            metrics['mean_relative_error'] = total_relative_error / count
            # Convergence score: 1.0 = perfect, 0.0 = very poor
            metrics['convergence_score'] = max(0, 1.0 - metrics['mean_relative_error'])
        
        return metrics
    
    def print_detailed_report(self):
        """Print a detailed report of Titan Mini's learned piece values."""
        estimated = self.estimate_piece_values_with_attention()
        metrics = self.get_convergence_metrics()
        
        print("\n" + "="*70)
        print("TITAN MINI PIECE VALUE ANALYSIS")
        print("="*70)
        
        print(f"Encoder Type: {'Enhanced (112 planes)' if self.enhanced_encoder else 'Classic (16 planes)'}")
        print(f"Model: Titan Mini Transformer")
        print("-"*70)
        
        print("\n{:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Piece", "Learned", "Standard", "Error", "Rel. Error"
        ))
        print("-"*70)
        
        for piece_type, piece_name in self.piece_names.items():
            if piece_type in estimated:
                learned = estimated[piece_type]
                standard = self.STANDARD_VALUES[piece_type]
                error = abs(learned - standard)
                rel_error = error / standard
                
                print("{:<15} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.1%}".format(
                    piece_name, learned, standard, error, rel_error
                ))
        
        print("-"*70)
        
        if 'mean_absolute_error' in metrics:
            print(f"\nSummary Statistics:")
            print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.3f}")
            print(f"  Mean Relative Error: {metrics['mean_relative_error']:.1%}")
            print(f"  Convergence Score:   {metrics['convergence_score']:.3f} / 1.000")
            
            # Interpretation
            score = metrics['convergence_score']
            if score > 0.9:
                print(f"  Assessment: EXCELLENT - Model has learned piece values accurately")
            elif score > 0.7:
                print(f"  Assessment: GOOD - Model understands piece values well")
            elif score > 0.5:
                print(f"  Assessment: MODERATE - Model has basic piece value understanding")
            else:
                print(f"  Assessment: POOR - Model needs more training on fundamental values")
        
        print("="*70 + "\n")
    
    def log_to_tensorboard(self, writer, epoch):
        """
        Log Titan Mini piece value metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            epoch: Current training epoch
        """
        metrics = self.get_convergence_metrics()
        
        # Log convergence metrics
        for key, value in metrics.items():
            writer.add_scalar(f'TitanPieceValues/{key}', value, epoch)
        
        # Log individual piece values
        estimated = self.estimate_piece_values_with_attention()
        for piece_type, value in estimated.items():
            piece_name = self.piece_names[piece_type]
            writer.add_scalar(f'TitanPieceValues/Raw/{piece_name}', value, epoch)
            
            # Also log the error from standard
            standard = self.STANDARD_VALUES[piece_type]
            error = abs(value - standard)
            writer.add_scalar(f'TitanPieceValues/Error/{piece_name}', error, epoch)