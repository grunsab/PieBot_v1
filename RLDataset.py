import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import chess
import encoder
try:
    import encoder_enhanced
except ImportError:
    encoder_enhanced = None
import time


class RLDataset(Dataset):
    """
    Dataset for reinforcement learning training data from self-play games.
    Unlike CCRLDataset, this loads pre-computed training positions with MCTS visit counts.
    """
    
    def __init__(self, data_dir, file_pattern='**/selfplay_*.h5', weight_recent=False, weight_decay=0.1, enhanced_encoder=False):
        """
        Args:
            data_dir (str): Directory containing self-play data files
            file_pattern (str): Pattern for self-play data files (supports recursive search)
            weight_recent (bool): If True, weight recent games more heavily
            weight_decay (float): Decay factor for weighting older games (lambda in exp(-lambda * age))
            enhanced_encoder (bool): If True, use enhanced 112-plane encoder
        """
        self.data_dir = data_dir
        self.data_files = []
        self.file_sizes = []
        self.file_iterations = []
        self.file_weights = []
        self.file_encoding_types = []  # Track encoding type per file
        self.cumulative_sizes = [0]
        self.weight_recent = weight_recent
        self.weight_decay = weight_decay
        self.enhanced_encoder = enhanced_encoder
        
        if enhanced_encoder and encoder_enhanced is None:
            raise ImportError("encoder_enhanced module not found but enhanced_encoder=True")
        
        # Find all matching data files (including in subdirectories)
        import glob
        pattern = os.path.join(data_dir, file_pattern)
        files = sorted(glob.glob(pattern, recursive=True))
        
        if not files:
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        # Extract iteration numbers and find max iteration
        max_iteration = 0
        file_info = []
        for file_path in files:
            # Try to extract iteration number from path or filename
            import re
            iter_match = re.search(r'iter[_\-]?(\d+)', file_path)
            iteration = int(iter_match.group(1)) if iter_match else 0
            max_iteration = max(max_iteration, iteration)
            file_info.append((file_path, iteration))
        
        # Load file metadata and calculate weights
        for file_path, iteration in file_info:
            with h5py.File(file_path, 'r') as f:
                size = f['positions'].shape[0]
                self.data_files.append(file_path)
                self.file_sizes.append(size)
                self.file_iterations.append(iteration)
                
                # Check encoding type from file metadata or shape
                if 'encoding_type' in f.attrs:
                    encoding_type = f.attrs['encoding_type']
                else:
                    # Infer from shape: 16 planes = standard, 112 = enhanced
                    num_planes = f['positions'].shape[1] if len(f['positions'].shape) > 1 else 16
                    encoding_type = 'enhanced' if num_planes == 112 else 'standard'
                self.file_encoding_types.append(encoding_type)
                
                # Warn if file encoding doesn't match requested
                if self.enhanced_encoder and encoding_type != 'enhanced':
                    print(f"Warning: File {file_path} has standard encoding but enhanced encoder requested")
                elif not self.enhanced_encoder and encoding_type == 'enhanced':
                    print(f"Warning: File {file_path} has enhanced encoding but standard encoder requested")
                
                # Calculate weight based on iteration age
                if self.weight_recent and max_iteration > 0:
                    age = max_iteration - iteration
                    weight = np.exp(-self.weight_decay * age)
                else:
                    weight = 1.0
                self.file_weights.append(weight)
                
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
        
        self.total_size = self.cumulative_sizes[-1]
        
        # Create weighted sampling distribution if needed
        if self.weight_recent:
            # Create position weights for all positions
            self.position_weights = []
            for i, (size, weight) in enumerate(zip(self.file_sizes, self.file_weights)):
                self.position_weights.extend([weight] * size)
            self.position_weights = np.array(self.position_weights)
            self.position_weights /= self.position_weights.sum()  # Normalize
            
            print(f"Loaded {len(self.data_files)} files with {self.total_size} positions")
            print(f"Iteration range: {min(self.file_iterations)} to {max(self.file_iterations)}")
            print(f"Weight range: {min(self.file_weights):.3f} to {max(self.file_weights):.3f}")
        else:
            print(f"Loaded {len(self.data_files)} files with {self.total_size} positions")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Load a training position from the self-play data.
        
        Returns:
            dict with keys:
                - position: torch.Tensor (16 or 112, 8, 8) encoded board position
                - policy: torch.Tensor (4608,) MCTS visit count distribution
                - value: torch.Tensor (1,) game outcome from this position
                - mask: torch.Tensor (72, 8, 8) legal move mask
        """
        # Find which file contains this index
        file_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                file_idx = i
                break
        
        # Calculate index within the file
        local_idx = idx - self.cumulative_sizes[file_idx]
        
        # Load data from the file
        with h5py.File(self.data_files[file_idx], 'r') as f:
            position = f['positions'][local_idx]  # (16 or 112, 8, 8)
            policy = f['policies'][local_idx]     # (4608,) - MCTS visit counts
            value = f['values'][local_idx]        # scalar - game outcome
            mask = f['masks'][local_idx]          # (72, 8, 8) - legal moves
            
            # Handle encoding type conversion if needed
            file_encoding = self.file_encoding_types[file_idx]
            if self.enhanced_encoder and file_encoding == 'standard':
                # Convert 16-plane to 112-plane (pad with zeros for missing history)
                enhanced_position = np.zeros((112, 8, 8), dtype=np.float32)
                enhanced_position[:16] = position  # Current position in first 16 planes
                # Castling rights are in planes 108-111 (copy from 12-15)
                enhanced_position[108:112] = position[12:16]
                position = enhanced_position
            elif not self.enhanced_encoder and file_encoding == 'enhanced':
                # Extract just the current position from enhanced encoding
                position = position[:16]  # Use only first 16 planes
        
        # Normalize policy to probabilities
        policy = policy.astype(np.float32)
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        
        return {
            'position': torch.from_numpy(position.astype(np.float32)),
            'policy': torch.from_numpy(policy),
            'value': torch.tensor([value], dtype=torch.float32),
            'mask': torch.from_numpy(mask.astype(np.int32))
        }


class SelfPlayDataCollector:
    """
    Collects training data during self-play games.
    Saves positions, MCTS visit counts, and game outcomes.
    """
    
    def __init__(self, output_file, iteration=0, enhanced_encoder=False):
        """
        Args:
            output_file (str): Path to output HDF5 file
            iteration (int): Training iteration number for metadata
            enhanced_encoder (bool): Use enhanced 112-plane encoder
        """
        self.output_file = output_file
        self.iteration = iteration
        self.enhanced_encoder = enhanced_encoder
        self.positions = []
        self.policies = []
        self.values = []
        self.masks = []
        self.game_positions = []  # Temporary storage for current game
        
        if enhanced_encoder and encoder_enhanced is None:
            raise ImportError("encoder_enhanced module not found but enhanced_encoder=True")
        
        # Initialize history for enhanced encoding
        if self.enhanced_encoder:
            self.history = encoder_enhanced.PositionHistory(history_length=8)
        else:
            self.history = None
    
    def add_position(self, board, mcts_visits, legal_moves):
        """
        Add a position from the current game.
        
        Args:
            board (chess.Board): Current board position
            mcts_visits (np.array): MCTS visit counts for each move (4608,)
            legal_moves (np.array): Legal move mask (72, 8, 8)
        """
        # Encode the position
        if self.enhanced_encoder:
            # Use enhanced encoding with history
            position_encoded = encoder_enhanced.encode_enhanced_position(board, self.history)
            # Add position to history for next encoding
            self.history.add_position(board)
        else:
            # Use standard encoding
            position_encoded = encoder.encodePosition(board)
        
        # Store for this game (value will be filled in when game ends)
        self.game_positions.append({
            'position': position_encoded,
            'policy': mcts_visits,
            'mask': legal_moves,
            'turn': board.turn  # Track whose turn it was
        })
    
    def end_game(self, result):
        """
        Called when a game ends. Assigns values to all positions in the game.
        
        Args:
            result (str): Game result ("1-0", "0-1", "1/2-1/2")
        """
        # Parse result
        if result == "1-0":
            white_value = 1.0
        elif result == "0-1":
            white_value = -1.0
        else:
            white_value = 0.0
        
        # Add all positions from this game with appropriate values
        for pos_data in self.game_positions:
            # Value is from the perspective of the player to move
            if pos_data['turn']:  # White to move
                value = white_value
            else:  # Black to move
                value = -white_value
            
            self.positions.append(pos_data['position'])
            self.policies.append(pos_data['policy'])
            self.values.append(value)
            self.masks.append(pos_data['mask'])
        
        # Clear game positions and reset history for next game
        self.game_positions = []
        if self.history:
            self.history = encoder_enhanced.PositionHistory(history_length=8)
    
    def save(self):
        """
        Save all collected data to HDF5 file.
        """
        if not self.positions:
            print("No data to save")
            return
        
        # Convert lists to arrays
        positions = np.array(self.positions, dtype=np.float32)
        policies = np.array(self.policies, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        masks = np.array(self.masks, dtype=np.int8)
        
        # Save to HDF5
        with h5py.File(self.output_file, 'w') as f:
            f.create_dataset('positions', data=positions, compression='gzip')
            f.create_dataset('policies', data=policies, compression='gzip')
            f.create_dataset('values', data=values)
            f.create_dataset('masks', data=masks, compression='gzip')
            
            # Save metadata
            f.attrs['num_positions'] = len(positions)
            f.attrs['num_games'] = len(self.values) // (len(self.game_positions) or 1)
            f.attrs['iteration'] = self.iteration
            f.attrs['timestamp'] = time.time()
            f.attrs['encoding_type'] = 'enhanced' if self.enhanced_encoder else 'standard'
            f.attrs['num_planes'] = 112 if self.enhanced_encoder else 16
        
        print(f"Saved {len(positions)} positions to {self.output_file} (iteration {self.iteration})")


class WeightedRLSampler(Sampler):
    """
    Custom sampler for RLDataset that samples positions based on game recency weights.
    Used to implement Leela Chess Zero's approach of weighting recent games more heavily.
    """
    
    def __init__(self, dataset, num_samples=None):
        """
        Args:
            dataset (RLDataset): Dataset with position weights
            num_samples (int): Number of samples per epoch (default: len(dataset))
        """
        if not hasattr(dataset, 'position_weights') or dataset.position_weights is None:
            raise ValueError("Dataset must have weight_recent=True for weighted sampling")
        
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.weights = dataset.position_weights
        
    def __iter__(self):
        # Sample indices based on weights
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=True,  # Allow replacement for proper weighting
            p=self.weights
        )
        return iter(indices)
    
    def __len__(self):
        return self.num_samples