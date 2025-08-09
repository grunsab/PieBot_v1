import chess
import numpy as np
import torch
from device_utils import get_optimal_device

# Get the optimal device once at module level
DEVICE, DEVICE_STR = get_optimal_device()

class PositionHistory:
    """
    Maintains a history of board positions for enhanced encoding.
    """
    def __init__(self, history_length=8):
        self.history_length = history_length
        self.history = []
        
    def add_position(self, board):
        """Add a board position to history."""
        self.history.append(board.copy())
        if len(self.history) > self.history_length:
            self.history.pop(0)
            
    def get_history(self):
        """Get the position history, padding with None if needed."""
        padded = self.history.copy()
        while len(padded) < self.history_length:
            padded.insert(0, None)
        return padded

def encode_piece_positions(board, planes, offset=0):
    """
    Encode piece positions for a single board state into planes.
    
    Args:
        board: chess.Board object
        planes: numpy array to fill
        offset: starting plane index
    
    Returns:
        Next available plane index
    """
    piece_map = {
        (chess.PAWN, chess.WHITE): offset + 0,
        (chess.PAWN, chess.BLACK): offset + 1,
        (chess.KNIGHT, chess.WHITE): offset + 2,
        (chess.KNIGHT, chess.BLACK): offset + 3,
        (chess.BISHOP, chess.WHITE): offset + 4,
        (chess.BISHOP, chess.BLACK): offset + 5,
        (chess.ROOK, chess.WHITE): offset + 6,
        (chess.ROOK, chess.BLACK): offset + 7,
        (chess.QUEEN, chess.WHITE): offset + 8,
        (chess.QUEEN, chess.BLACK): offset + 9,
        (chess.KING, chess.WHITE): offset + 10,
        (chess.KING, chess.BLACK): offset + 11,
    }
    
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color
        plane_idx = piece_map[(piece_type, color)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Mirror position if it's black's turn for consistency
        if not board.turn:
            rank = 7 - rank
            file = 7 - file
            
        planes[plane_idx, rank, file] = 1.0
    
    return offset + 12

def encode_enhanced_position(board, history=None):
    """
    Enhanced position encoding with 112 planes for richer representation.
    
    Plane allocation:
    - 0-11: Current position pieces (12 planes)
    - 12-107: Historical positions (8 positions × 12 planes = 96 planes)
    - 108: Castling rights (K-side white)
    - 109: Castling rights (Q-side white)
    - 110: Castling rights (K-side black)
    - 111: Castling rights (Q-side black)
    
    Additional features encoded in existing planes:
    - En passant squares marked in pawn planes
    - Move counters normalized and added to king planes
    
    Args:
        board: chess.Board object
        history: PositionHistory object or None
        
    Returns:
        numpy array of shape (112, 8, 8)
    """
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    
    # Encode current position
    encode_piece_positions(board, planes, 0)
    
    # Encode position history
    if history is not None:
        historical_positions = history.get_history()
        for i, hist_board in enumerate(historical_positions):
            if hist_board is not None:
                encode_piece_positions(hist_board, planes, 12 + i * 12)
    
    # Encode castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[108, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[109, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[110, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[111, :, :] = 1.0
    
    # Encode en passant square
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        if not board.turn:
            ep_rank = 7 - ep_rank
            ep_file = 7 - ep_file
        # Mark en passant in appropriate pawn plane
        pawn_plane = 0 if board.turn else 1
        planes[pawn_plane, ep_rank, ep_file] = 0.5  # Use 0.5 to distinguish from regular pawns
    
    # Add move counters as continuous features
    # Normalize to [0, 1] range
    halfmove_clock = min(board.halfmove_clock / 100.0, 1.0)  # Cap at 100 halfmoves
    fullmove_number = min(board.fullmove_number / 200.0, 1.0)  # Cap at 200 moves
    
    # Add to king planes as they're least likely to be full
    planes[10, :, :] += halfmove_clock * 0.1  # White king plane
    planes[11, :, :] += fullmove_number * 0.1  # Black king plane
    
    # Mirror the entire position if it's black's turn
    if not board.turn:
        # Swap white and black pieces
        for i in range(0, 12, 2):
            planes[[i, i+1]] = planes[[i+1, i]]
        # Do the same for historical positions
        for hist_idx in range(1, 9):
            base = 12 * hist_idx
            for i in range(0, 12, 2):
                if base + i + 1 < 108:  # Safety check
                    planes[[base + i, base + i + 1]] = planes[[base + i + 1, base + i]]
        # Swap castling rights
        planes[[108, 110]] = planes[[110, 108]]
        planes[[109, 111]] = planes[[111, 109]]
    
    return planes

def encode_simple_position(board):
    """
    Simple 16-plane encoding for backward compatibility with original AlphaZero.
    
    Args:
        board: chess.Board object
        
    Returns:
        numpy array of shape (16, 8, 8)
    """
    planes = np.zeros((16, 8, 8), dtype=np.float32)
    
    # Encode pieces (12 planes)
    encode_piece_positions(board, planes, 0)
    
    # Encode castling rights (4 planes)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    
    # Mirror if black's turn
    if not board.turn:
        # Swap white and black pieces
        for i in range(0, 12, 2):
            planes[[i, i+1]] = planes[[i+1, i]]
        # Swap castling rights
        planes[[12, 14]] = planes[[14, 12]]
        planes[[13, 15]] = planes[[15, 13]]
        # Flip the board vertically
        planes = np.flip(planes, axis=1)
    
    return planes

def encode_legal_moves(board):
    """
    Encode legal moves as a mask for the policy head.
    
    Args:
        board: chess.Board object
        
    Returns:
        numpy array of shape (72, 8, 8) representing legal moves
    """
    # This follows the original AlphaZero move encoding
    # 72 move types: 56 queen moves + 8 knight moves + 8 underpromotions
    legal_moves = np.zeros((72, 8, 8), dtype=np.float32)
    
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)
        
        # Mirror if black's turn
        if not board.turn:
            from_rank = 7 - from_rank
            from_file = 7 - from_file
        
        # Calculate move type index
        # This is simplified - actual implementation would need full move encoding
        # For now, just mark the from square in first plane
        legal_moves[0, from_rank, from_file] = 1.0
    
    return legal_moves

def parseResult(result):
    """
    Map the result string to a value in {-1, 0, 1}.
    
    Args:
        result: string representation of game result
        
    Returns:
        int: -1 for black win, 0 for draw, 1 for white win
    """
    if result == "1-0":
        return 1
    elif result == "1/2-1/2":
        return 0
    elif result == "0-1":
        return -1
    else:
        raise Exception(f"Unexpected result string {result}")

# Maintain backward compatibility
encodePosition = encode_simple_position

class EnhancedEncoder:
    """
    Stateful encoder that maintains position history for enhanced encoding.
    """
    def __init__(self, history_length=8, use_enhanced=True):
        self.history = PositionHistory(history_length)
        self.use_enhanced = use_enhanced
        
    def encode(self, board):
        """
        Encode a board position with history.
        
        Args:
            board: chess.Board object
            
        Returns:
            torch.Tensor: encoded position
        """
        if self.use_enhanced:
            encoded = encode_enhanced_position(board, self.history)
        else:
            encoded = encode_simple_position(board)
        
        # Add to history for next encoding
        self.history.add_position(board)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(encoded).float()
        return tensor.to(DEVICE)
    
    def reset_history(self):
        """Reset the position history."""
        self.history = PositionHistory(self.history.history_length)