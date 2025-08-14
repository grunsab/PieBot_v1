"""
Monitor piece values learned by the Titan Mini transformer network.
Revised to correctly handle non-linear AlphaZero-style evaluations.
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

# Helper function to linearize the [-1, 1] evaluation score
def linearize_evaluation(value, C=350.0):
    """
    Converts AlphaZero evaluation V (in [-1, 1]) to a centipawn-like scale (CP).
    Uses the approximation CP = C * ArcTanh(V).
    C is a scaling constant (adjust if needed, 350 is a reasonable default).
    """
    # Clamp value to avoid ArcTanh(1) or ArcTanh(-1)
    clamped_value = np.clip(value, -0.999, 0.999)
    # Calculate CP value
    cp_value = C * np.arctanh(clamped_value)
    return cp_value

class TitanPieceValueMonitor:
    
    # Standard piece values for reference (in pawns)
    STANDARD_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0
    }
    
    def __init__(self, model, device='cpu', enhanced_encoder=True):
        self.model = model
        self.device = device
        
        # Check if the model is actually using sparse input (16 planes)
        if hasattr(model, 'input_planes') and model.input_planes <= 16:
            self.enhanced_encoder = False
        else:
            self.enhanced_encoder = enhanced_encoder

        # Create encoder instance if using enhanced encoder
        if self.enhanced_encoder and encoder_enhanced:
            self.encoder_instance = encoder_enhanced.EnhancedEncoder(use_enhanced=True)
        else:
            self.encoder_instance = None
        
        self.piece_names = {
            chess.PAWN: 'Pawn', chess.KNIGHT: 'Knight', chess.BISHOP: 'Bishop',
            chess.ROOK: 'Rook', chess.QUEEN: 'Queen'
        }
    
    def create_comprehensive_test_positions(self) -> List[Tuple[chess.Board, chess.Board, int]]:
        # (Test positions remain the same as provided in the prompt, including mixed turns)
        test_positions = []
        
        # Opening positions
        opening_tests = [
            # WTM
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", chess.ROOK),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", chess.QUEEN),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1", chess.PAWN),
            
            # BTM
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBN1 b KQkq e3 0 1", chess.ROOK),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNB1KBNR b KQkq e3 0 1", chess.QUEEN),
        ]
        
        # Middle game positions
        middle_game_tests = [
            # WTM
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
             "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK3 w KQkq - 0 6", chess.ROOK),
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
             "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P4/PPP2PPP/RNBQK2R w KQkq - 0 6", chess.KNIGHT),
        ]
        
        # Endgame positions
        endgame_tests = [
            # WTM
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6R1 w - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 w - - 0 1", chess.ROOK),
            # BTM
            ("8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/6B1 b - - 0 1",
             "8/5pk1/4p1p1/3p2P1/3P4/4P3/5PK1/8 b - - 0 1", chess.BISHOP),
        ]
        
        for fen_with, fen_without, piece_type in opening_tests + middle_game_tests + endgame_tests:
            board_with = chess.Board(fen_with)
            board_without = chess.Board(fen_without)
            if board_with.turn != board_without.turn:
                 continue
            test_positions.append((board_with, board_without, piece_type))
        
        return test_positions
    
    def estimate_piece_values_with_attention(self) -> Dict[int, float]:
        """
        Estimate piece values by linearizing the network's evaluation scores.
        """
        self.model.eval()
        # Store values in centipawn scale
        piece_values_cp = {piece: [] for piece in self.piece_names.keys()}
        
        test_positions = self.create_comprehensive_test_positions()
        
        with torch.no_grad():
            for board_with, board_without, piece_type in test_positions:
                # Encode positions (Handles perspective normalization internally via encoder.py)
                if self.enhanced_encoder and self.encoder_instance:
                    # (Enhanced encoder path remains the same)
                    self.encoder_instance.reset_history()
                    pos_with_tensor_raw = self.encoder_instance.encode(board_with)
                    self.encoder_instance.reset_history()
                    pos_without_tensor_raw = self.encoder_instance.encode(board_without)
                    pos_with = pos_with_tensor_raw.cpu().numpy()
                    pos_without = pos_without_tensor_raw.cpu().numpy()
                    _, mask_with = encoder.encodePositionForInference(board_with)
                    _, mask_without = encoder.encodePositionForInference(board_without)
                else:
                    # Standard encoder path
                    pos_with, mask_with = encoder.encodePositionForInference(board_with)
                    pos_without, mask_without = encoder.encodePositionForInference(board_without)
                
                # Convert to tensors and flatten masks
                pos_with_tensor = torch.from_numpy(pos_with if isinstance(pos_with, np.ndarray) else np.array(pos_with)).unsqueeze(0).to(self.device)
                pos_without_tensor = torch.from_numpy(pos_without if isinstance(pos_without, np.ndarray) else np.array(pos_without)).unsqueeze(0).to(self.device)
                mask_with_tensor = torch.from_numpy(mask_with).unsqueeze(0).to(self.device)
                mask_without_tensor = torch.from_numpy(mask_without).unsqueeze(0).to(self.device)
                
                mask_with_flat = mask_with_tensor.view(mask_with_tensor.shape[0], -1)
                mask_without_flat = mask_without_tensor.view(mask_without_tensor.shape[0], -1)
                
                # Get network evaluations (Raw [-1, 1] scores)
                value_with_raw, _ = self.model(pos_with_tensor, policyMask=mask_with_flat)
                value_without_raw, _ = self.model(pos_without_tensor, policyMask=mask_without_flat)
                
                # FIX: Linearize evaluations and calculate absolute difference
                value_with_cp = linearize_evaluation(value_with_raw.item())
                value_without_cp = linearize_evaluation(value_without_raw.item())

                # Calculate difference in centipawns
                value_diff_cp = value_with_cp - value_without_cp
                
                # Store the absolute value difference. This handles perspective correctly, 
                # as the magnitude represents the piece value.
                piece_values_cp[piece_type].append(abs(value_diff_cp))
        
        # Robust averaging of CP values
        estimated_values_cp = {}
        for piece_type, values in piece_values_cp.items():
            if values:
                # (Averaging logic remains the same)
                values_array = np.array(values)
                mean = np.mean(values_array)
                std = np.std(values_array)
                
                if std > 0.01 and len(values_array) > 3:
                    filtered = values_array[np.abs(values_array - mean) <= 2 * std]
                    if len(filtered) > 0:
                        avg_value = np.mean(filtered)
                    else:
                        avg_value = mean
                else:
                    avg_value = mean
                    
                estimated_values_cp[piece_type] = avg_value
        
        # Normalize values relative to the estimated pawn value (in pawns)
        estimated_values_normalized = {}
        if chess.PAWN in estimated_values_cp and estimated_values_cp[chess.PAWN] > 10.0: # Ensure pawn value is reasonable
            pawn_value_cp = estimated_values_cp[chess.PAWN]
            for piece_type in estimated_values_cp:
                estimated_values_normalized[piece_type] = estimated_values_cp[piece_type] / pawn_value_cp
        else:
            # Fallback
            print("Warning: Pawn value estimation failed or is too low. Using fallback normalization (1 Pawn = 100 CP).")
            for piece_type in estimated_values_cp:
                 estimated_values_normalized[piece_type] = estimated_values_cp[piece_type] / 100.0

        
        return estimated_values_normalized
    
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        # (Metrics calculation updated to exclude Pawn from error metrics)
        estimated = self.estimate_piece_values_with_attention()
        
        metrics = {}
        total_error = 0
        total_relative_error = 0
        count = 0
        
        for piece_type, standard_value in self.STANDARD_VALUES.items():
            if piece_type in estimated:
                est_value = estimated[piece_type]
                error = abs(est_value - standard_value)
                relative_error = error / standard_value if standard_value != 0 else 0
                
                piece_name = self.piece_names[piece_type]
                metrics[f'{piece_name}_value'] = est_value
                
                # Exclude Pawn from MAE/MRE calculation as it's the normalization unit
                if piece_type != chess.PAWN:
                    total_error += error
                    total_relative_error += relative_error
                    count += 1
        
        if count > 0:
            metrics['mean_absolute_error'] = total_error / count
            metrics['mean_relative_error'] = total_relative_error / count
            metrics['convergence_score'] = max(0, 1.0 - metrics['mean_relative_error'])
        else:
            metrics['mean_absolute_error'] = 0
            metrics['mean_relative_error'] = 0
            metrics['convergence_score'] = 1.0
        
        return metrics
    
    def print_detailed_report(self):
        # (Reporting remains the same, title updated)
        estimated = self.estimate_piece_values_with_attention()
        metrics = self.get_convergence_metrics()
        
        print("\n" + "="*70)
        print("TITAN MINI PIECE VALUE ANALYSIS (Linearized)")
        print("="*70)
        
        print(f"Encoder Type: {'Enhanced (112 planes)' if self.enhanced_encoder else 'Classic (16 planes)'}")
        print(f"Model: Titan Mini Transformer")
        print("-"*70)
        
        print("\n{:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Piece", "Learned", "Standard", "Error", "Rel. Error"
        ))
        print("-"*70)
        
        for piece_type in sorted(self.piece_names.keys()):
            piece_name = self.piece_names[piece_type]
            if piece_type in estimated:
                learned = estimated[piece_type]
                standard = self.STANDARD_VALUES[piece_type]
                error = abs(learned - standard)
                rel_error = error / standard if standard != 0 else 0
                
                print("{:<15} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.1%}".format(
                    piece_name, learned, standard, error, rel_error
                ))
        
        print("-"*70)
        
        if 'mean_absolute_error' in metrics:
            print(f"\nSummary Statistics (Excluding Pawn):")
            print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.3f}")
            print(f"  Mean Relative Error: {metrics['mean_relative_error']:.1%}")
            print(f"  Convergence Score:   {metrics['convergence_score']:.3f} / 1.000")
            
            score = metrics['convergence_score']
            if score > 0.9:
                print(f"  Assessment: EXCELLENT - Model has learned piece values accurately")
            elif score > 0.75:
                print(f"  Assessment: GOOD - Model understands piece values well")
            elif score > 0.5:
                print(f"  Assessment: MODERATE - Model has basic piece value understanding")
            else:
                print(f"  Assessment: POOR - Model needs more training on fundamental values")
        
        print("="*70 + "\n")

    def log_to_tensorboard(self, writer, epoch):
        # (Implementation remains the same)
        metrics = self.get_convergence_metrics()
        for key, value in metrics.items():
            writer.add_scalar(f'TitanPieceValues/{key}', value, epoch)
        estimated = self.estimate_piece_values_with_attention()
        for piece_type, value in estimated.items():
            piece_name = self.piece_names[piece_type]
            writer.add_scalar(f'TitanPieceValues/Raw/{piece_name}', value, epoch)
            standard = self.STANDARD_VALUES[piece_type]
            error = abs(value - standard)
            writer.add_scalar(f'TitanPieceValues/Error/{piece_name}', error, epoch)