
import chess.pgn
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import encoder
try:
    import encoder_enhanced
except ImportError:
    encoder_enhanced = None

def tolist( mainline_moves ):
    """
    Change an iterable object of moves to a list of moves.
    
    Args:
        mainline_moves (Mainline object) iterable list of moves

    Returns:
        moves (list of chess.Move) list version of the input moves
    """
    moves = []
    for move in mainline_moves:
        moves.append( move )
    return moves

class CCRLDataset( Dataset ):
    """
    Subclass of torch.utils.data.Dataset for the ccrl dataset.
    """

    def __init__( self, ccrl_dir, soft_targets=True, temperature=0.1, enhanced_encoder=False ):
        """
        Args:
            ccrl_dir (string) Path to directory containing
                pgn files with names 0.pgn, 1.pgn, 2.pgn, etc.
            soft_targets (bool) If True, return probability distribution instead of move index
            temperature (float) Temperature for label smoothing when using soft targets
            enhanced_encoder (bool) If True, use enhanced encoder with 112 planes
        """
        self.ccrl_dir = ccrl_dir
        self.pgn_file_names = os.listdir( ccrl_dir )
        self.soft_targets = soft_targets
        self.temperature = temperature
        self.enhanced_encoder = enhanced_encoder
        
        if enhanced_encoder and encoder_enhanced is None:
            raise ImportError("encoder_enhanced module not found but enhanced_encoder=True")

    def __len__( self ):
        """
        Get length of dataset
        """
        return len( self.pgn_file_names )

    def __getitem__( self, idx ):
        """
        Load the game in idx.pgn
        Get a random position, the move made from it, and the winner
        Encode these as numpy arrays
        
        Args:
            idx (int) the index into the dataset.
        
        Returns:
           position (torch.Tensor (16, 8, 8) float32) the encoded position
           policy (torch.Tensor (1) long) the target move's index
           value (torch.Tensor (1) float) the encoded winner of the game
           mask (torch.Tensor (72, 8, 8) int) the legal move mask
        """
        pgn_file_name = self.pgn_file_names[ idx ]
        pgn_file_name = os.path.join( self.ccrl_dir, pgn_file_name )
        pgn_fh = open( pgn_file_name )
        game = chess.pgn.read_game( pgn_fh )
        pgn_fh.close()
        
        if game is None:
            # Try next file if this one failed to parse
            return self.__getitem__((idx + 1) % len(self))

        moves = tolist( game.mainline_moves() )
        
        if len(moves) < 2:
            # Need at least 2 moves to get a position and next move
            return self.__getitem__((idx + 1) % len(self))

        randIdx = int( np.random.random() * ( len( moves ) - 1 ) )

        board = game.board()

        for idx, move in enumerate( moves ):
            board.push( move )
            if( randIdx == idx ):
                next_move = moves[ idx + 1 ]
                break

        if 'Result' not in game.headers:
            # Skip games without result
            return self.__getitem__((idx + 1) % len(self))
            
        winner = encoder.parseResult( game.headers[ 'Result' ] )

        if self.enhanced_encoder:
            # Use enhanced encoder - need to encode components separately
            # Encode position with enhanced features (112 planes)
            position = encoder_enhanced.encode_enhanced_position(board)
            
            # Encode the move to get policy index
            # Mirror the move if it's black's turn (to match encoder behavior)
            move_to_encode = next_move
            if not board.turn:
                move_to_encode = encoder.mirrorMove(next_move)
            planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move_to_encode)
            policy_idx = planeIdx * 64 + rankIdx * 8 + fileIdx
            
            # Encode value (winner from perspective of current player)
            if board.turn:
                value = (winner + 1.) / 2.
            else:
                value = (-winner + 1.) / 2.
            
            # Encode legal moves mask
            mask = encoder.getLegalMoveMask(board)
        else:
            # Use regular encoder
            position, policy_idx, value, mask = encoder.encodeTrainingPoint( board, next_move, winner )
        
        if self.soft_targets:
            # Convert to probability distribution with label smoothing
            policy = np.zeros(4608, dtype=np.float32)
            
            # Set the played move to high probability
            policy[policy_idx] = 1.0
            
            # Apply temperature-based label smoothing
            if self.temperature > 0:
                # Add small probability to all legal moves
                legal_indices = []
                for move in board.legal_moves:
                    # Mirror the move if it's black's turn (to match encoder behavior)
                    if not board.turn:
                        from encoder import mirrorMove
                        move = mirrorMove(move)
                    planeIdx, rankIdx, fileIdx = encoder.moveToIdx(move)
                    moveIdx = planeIdx * 64 + rankIdx * 8 + fileIdx
                    legal_indices.append(moveIdx)
                
                # Apply smoothing only to legal moves
                num_legal = len(legal_indices)
                if num_legal > 1:
                    smoothing_prob = self.temperature / num_legal
                    policy[policy_idx] = 1.0 - self.temperature + smoothing_prob
                    for idx in legal_indices:
                        if idx != policy_idx:
                            policy[idx] = smoothing_prob
            
            return { 'position': torch.from_numpy( position ),
                     'policy': torch.from_numpy( policy ),
                     'value': torch.Tensor( [value] ),
                     'mask': torch.from_numpy( mask ) }
        else:
            # Original behavior: return move index
            return { 'position': torch.from_numpy( position ),
                     'policy': torch.Tensor( [policy_idx] ).type( dtype=torch.long ),
                     'value': torch.Tensor( [value] ),
                     'mask': torch.from_numpy( mask ) }
