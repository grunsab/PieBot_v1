"""
Curriculum Dataset for progressive Titan Mini chess training.
Optimized for transformer-based architecture with enhanced positional encoding.
"""

import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from CCRLDataset import CCRLDataset
import json
from typing import Dict, List, Tuple, Optional

class TitanCurriculumStage:
    """Represents a single stage in the Titan Mini curriculum training."""
    
    def __init__(self, name: str, data_dir: str, elo_range: Tuple[int, int], 
                 epochs: int, value_weight: float = 1.0, 
                 soft_targets: bool = True, temperature: float = 0.15,
                 lr_multiplier: float = 1.0, enhanced_encoder: bool = True):
        """
        Args:
            name: Stage name (e.g., 'beginner', 'intermediate', 'expert', 'computer')
            data_dir: Directory containing games for this stage
            elo_range: Tuple of (min_elo, max_elo) for this stage
            epochs: Number of epochs to train on this stage
            value_weight: Weight for value loss relative to policy loss
            soft_targets: Whether to use soft targets for policy
            temperature: Temperature for label smoothing
            lr_multiplier: Learning rate multiplier for this stage
            enhanced_encoder: Use enhanced encoder with 112 planes (Titan Mini default)
        """
        self.name = name
        self.data_dir = data_dir
        self.elo_range = elo_range
        self.epochs = epochs
        self.value_weight = value_weight
        self.soft_targets = soft_targets
        self.temperature = temperature
        self.lr_multiplier = lr_multiplier
        self.enhanced_encoder = enhanced_encoder
        self.dataset = None
        
    def load_dataset(self):
        """Load the dataset for this stage."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        self.dataset = CCRLDataset(
            self.data_dir, 
            soft_targets=self.soft_targets,
            temperature=self.temperature,
            enhanced_encoder=self.enhanced_encoder
        )
        return self.dataset
    
    def __repr__(self):
        return (f"TitanCurriculumStage(name={self.name}, elo={self.elo_range}, "
                f"epochs={self.epochs}, value_weight={self.value_weight}, "
                f"lr_mult={self.lr_multiplier})")


class TitanCurriculumDataset(Dataset):
    """
    Dataset that manages curriculum learning for Titan Mini transformer model.
    Optimized for the deeper architecture and attention mechanisms of Titan Mini.
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None, enhanced_encoder: bool = True):
        """
        Args:
            curriculum_config: Configuration dict with stages or path to config file
            enhanced_encoder: Use enhanced encoder with 112 planes (default True for Titan)
        """
        self.stages: List[TitanCurriculumStage] = []
        self.current_stage_idx = 0
        self.current_dataset = None
        self.total_epochs_trained = 0
        self.stage_epochs_trained = 0
        self.enhanced_encoder = enhanced_encoder
        
        if curriculum_config:
            if isinstance(curriculum_config, str):
                # Load from file
                with open(curriculum_config, 'r') as f:
                    config = json.load(f)
            else:
                config = curriculum_config
                
            self._setup_stages(config)
        else:
            # Use default curriculum optimized for Titan Mini
            self._setup_default_titan_curriculum()
    
    def _setup_default_titan_curriculum(self):
        """
        Setup default 4-stage curriculum optimized for Titan Mini's architecture.
        Titan Mini benefits from longer training and gradual difficulty progression.
        """
        base_dir = 'games_training_data/curriculum'
        
        # Stage 1: Beginner (750-1500 ELO)
        # Focus on learning basic piece values and simple tactics
        self.stages.append(TitanCurriculumStage(
            name='beginner',
            data_dir=os.path.join(base_dir, 'beginner'),
            elo_range=(750, 1500),
            epochs=80,  # More epochs for transformer to learn basics
            value_weight=2.5,  # Strong emphasis on value learning
            soft_targets=True,
            temperature=0.20,  # Higher temperature for smoother learning
            lr_multiplier=0.5,  # Start with lower LR for stability
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 2: Intermediate (1500-2400 ELO)
        # Develop positional understanding and strategic concepts
        self.stages.append(TitanCurriculumStage(
            name='intermediate',
            data_dir=os.path.join(base_dir, 'intermediate'),
            elo_range=(1500, 2400),
            epochs=80,  # More epochs for complex patterns
            value_weight=1.8,
            soft_targets=True,
            temperature=0.15,
            lr_multiplier=1.0,  # Normal learning rate
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 3: Expert (2400-3000 ELO)
        # Master advanced strategies and endgame understanding
        self.stages.append(TitanCurriculumStage(
            name='expert',
            data_dir=os.path.join(base_dir, 'expert'),
            elo_range=(2400, 3000),
            epochs=150,  # Extended training for expert play
            value_weight=1.2,
            soft_targets=True,
            temperature=0.12,
            lr_multiplier=0.8,  # Slightly reduced LR for fine-tuning
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 4: Computer (3000-4000 ELO)
        # Refine to engine-level play with sophisticated evaluation
        self.stages.append(TitanCurriculumStage(
            name='computer',
            data_dir=os.path.join(base_dir, 'computer'),
            elo_range=(3000, 4000),
            epochs=150,  # Extensive training for peak performance
            value_weight=0.8,  # Focus on policy for sophisticated play
            soft_targets=True,
            temperature=0.08,
            lr_multiplier=0.5,  # Low LR for final refinement
            enhanced_encoder=self.enhanced_encoder
        ))
    
    def _setup_stages(self, config: Dict):
        """Setup stages from configuration."""
        for stage_config in config.get('stages', []):
            # Set enhanced encoder default if not specified
            if 'enhanced_encoder' not in stage_config:
                stage_config['enhanced_encoder'] = self.enhanced_encoder
            stage = TitanCurriculumStage(**stage_config)
            self.stages.append(stage)
    
    def get_current_stage(self) -> Optional[TitanCurriculumStage]:
        """Get the current training stage."""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def get_lr_multiplier(self) -> float:
        """Get the learning rate multiplier for current stage."""
        current_stage = self.get_current_stage()
        if current_stage:
            return current_stage.lr_multiplier
        return 1.0
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        Returns True if successfully advanced, False if no more stages.
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.stage_epochs_trained = 0
            self._load_current_stage()
            return True
        return False
    
    def should_advance(self) -> bool:
        """Check if it's time to advance to the next stage."""
        current_stage = self.get_current_stage()
        if current_stage:
            return self.stage_epochs_trained >= current_stage.epochs
        return False
    
    def _load_current_stage(self):
        """Load the dataset for the current stage."""
        current_stage = self.get_current_stage()
        if current_stage:
            self.current_dataset = current_stage.load_dataset()
            print(f"\n{'='*60}")
            print(f"Loading Titan Mini curriculum stage: {current_stage.name.upper()}")
            print(f"  ELO range: {current_stage.elo_range}")
            print(f"  Epochs: {current_stage.epochs}")
            print(f"  Value weight: {current_stage.value_weight}")
            print(f"  LR multiplier: {current_stage.lr_multiplier}")
            print(f"  Temperature: {current_stage.temperature}")
            print(f"  Enhanced encoder: {current_stage.enhanced_encoder}")
            print(f"  Dataset size: {len(self.current_dataset)}")
            print(f"{'='*60}\n")
    
    def on_epoch_end(self):
        """Called at the end of each training epoch."""
        self.total_epochs_trained += 1
        self.stage_epochs_trained += 1
        
        current_stage = self.get_current_stage()
        if current_stage:
            print(f"[Titan Curriculum] Stage '{current_stage.name}': "
                  f"Epoch {self.stage_epochs_trained}/{current_stage.epochs} "
                  f"(Global: {self.total_epochs_trained})")
    
    def get_value_weight(self) -> float:
        """Get the current value loss weight."""
        current_stage = self.get_current_stage()
        if current_stage:
            return current_stage.value_weight
        return 1.0
    
    def get_current_stage_size(self) -> int:
        """Get the number of games in the current stage."""
        current_stage = self.get_current_stage()
        if current_stage and current_stage.data_dir:
            # Count PGN files in the stage directory
            if os.path.exists(current_stage.data_dir):
                pgn_files = [f for f in os.listdir(current_stage.data_dir) if f.endswith('.pgn')]
                return len(pgn_files)
        # If we have a loaded dataset, return its size
        if self.current_dataset:
            return len(self.current_dataset)
        return 100000  # Default estimate
    
    def get_stage_info(self) -> Dict:
        """Get information about current training progress."""
        current_stage = self.get_current_stage()
        if current_stage:
            return {
                'current_stage': current_stage.name,
                'stage_index': self.current_stage_idx,
                'total_stages': len(self.stages),
                'stage_epochs': self.stage_epochs_trained,
                'stage_total_epochs': current_stage.epochs,
                'total_epochs': self.total_epochs_trained,
                'elo_range': current_stage.elo_range,
                'value_weight': current_stage.value_weight,
                'lr_multiplier': current_stage.lr_multiplier,
                'temperature': current_stage.temperature,
                'stage_size': self.get_current_stage_size()
            }
        return {}
    
    def __len__(self):
        """Return length of current dataset."""
        if self.current_dataset is None:
            self._load_current_stage()
        return len(self.current_dataset) if self.current_dataset else 0
    
    def __getitem__(self, idx):
        """Get item from current dataset."""
        if self.current_dataset is None:
            self._load_current_stage()
        return self.current_dataset[idx] if self.current_dataset else None
    
    def save_state(self, filepath: str):
        """Save curriculum state for resuming training."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'total_epochs_trained': self.total_epochs_trained,
            'stage_epochs_trained': self.stage_epochs_trained,
            'enhanced_encoder': self.enhanced_encoder,
            'stages': [
                {
                    'name': stage.name,
                    'data_dir': stage.data_dir,
                    'elo_range': stage.elo_range,
                    'epochs': stage.epochs,
                    'value_weight': stage.value_weight,
                    'soft_targets': stage.soft_targets,
                    'temperature': stage.temperature,
                    'lr_multiplier': stage.lr_multiplier,
                    'enhanced_encoder': stage.enhanced_encoder
                }
                for stage in self.stages
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Titan curriculum state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load curriculum state for resuming training."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage_idx = state['current_stage_idx']
        self.total_epochs_trained = state['total_epochs_trained']
        self.stage_epochs_trained = state['stage_epochs_trained']
        self.enhanced_encoder = state.get('enhanced_encoder', True)
        
        # Rebuild stages
        self.stages = []
        for stage_config in state['stages']:
            self.stages.append(TitanCurriculumStage(**stage_config))
        
        self._load_current_stage()
        print(f"Titan curriculum state loaded from {filepath}")
        print(f"Resuming from stage '{self.get_current_stage().name}', "
              f"epoch {self.stage_epochs_trained}")


class TitanMixedCurriculumDataset(Dataset):
    """
    Dataset that mixes games from different skill levels for Titan Mini.
    Prevents catastrophic forgetting while maintaining progressive learning.
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None, 
                 mixing_ratio: Optional[Dict] = None,
                 enhanced_encoder: bool = True):
        """
        Args:
            curriculum_config: Configuration for curriculum stages
            mixing_ratio: Dict mapping stage names to sampling ratios
            enhanced_encoder: Use enhanced encoder with 112 planes
        """
        self.curriculum = TitanCurriculumDataset(curriculum_config, enhanced_encoder)
        
        # Optimized mixing ratio for Titan Mini
        self.mixing_ratio = mixing_ratio or {
            'beginner': 0.10,      # Less emphasis on basic games
            'intermediate': 0.25,  # Moderate sampling of mid-level
            'expert': 0.35,        # Strong focus on expert play
            'computer': 0.30       # Significant engine-level games
        }
        
        # Load all datasets
        self.datasets = {}
        for stage in self.curriculum.stages:
            if stage.name in self.mixing_ratio and self.mixing_ratio[stage.name] > 0:
                stage.load_dataset()
                self.datasets[stage.name] = stage.dataset
        
        # Calculate dataset sizes based on ratios
        self._calculate_sampling_indices()
    
    def _calculate_sampling_indices(self):
        """Calculate indices for sampling from each dataset."""
        self.indices = []
        self.dataset_mapping = []
        
        # Find minimum dataset size for balanced sampling
        min_size = min(len(ds) for ds in self.datasets.values())
        
        for stage_name, dataset in self.datasets.items():
            ratio = self.mixing_ratio.get(stage_name, 0)
            if ratio > 0:
                sample_size = int(min_size * ratio)
                
                # Add indices from this dataset
                for i in range(sample_size):
                    idx = i % len(dataset)
                    self.indices.append(len(self.dataset_mapping))
                    self.dataset_mapping.append((stage_name, idx))
        
        print(f"\nTitan Mini mixed curriculum dataset created:")
        print(f"Total samples: {len(self.indices)}")
        for stage_name in self.datasets:
            count = sum(1 for name, _ in self.dataset_mapping if name == stage_name)
            print(f"  {stage_name}: {count} samples ({count/len(self.indices)*100:.1f}%)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stage_name, dataset_idx = self.dataset_mapping[idx]
        return self.datasets[stage_name][dataset_idx]