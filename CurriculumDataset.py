"""
Curriculum Dataset for progressive chess training.
Manages training data across different skill levels (ELO ranges).
"""

import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from CCRLDataset import CCRLDataset
import json
from typing import Dict, List, Tuple, Optional

class CurriculumStage:
    """Represents a single stage in the curriculum training."""
    
    def __init__(self, name: str, data_dir: str, elo_range: Tuple[int, int], 
                 epochs: int, value_weight: float = 1.0, 
                 soft_targets: bool = True, temperature: float = 0.1):
        """
        Args:
            name: Stage name (e.g., 'beginner', 'intermediate', 'expert')
            data_dir: Directory containing games for this stage
            elo_range: Tuple of (min_elo, max_elo) for this stage
            epochs: Number of epochs to train on this stage
            value_weight: Weight for value loss relative to policy loss
            soft_targets: Whether to use soft targets for policy
            temperature: Temperature for label smoothing
        """
        self.name = name
        self.data_dir = data_dir
        self.elo_range = elo_range
        self.epochs = epochs
        self.value_weight = value_weight
        self.soft_targets = soft_targets
        self.temperature = temperature
        self.dataset = None
        
    def load_dataset(self):
        """Load the dataset for this stage."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        self.dataset = CCRLDataset(
            self.data_dir, 
            soft_targets=self.soft_targets,
            temperature=self.temperature
        )
        return self.dataset
    
    def __repr__(self):
        return (f"CurriculumStage(name={self.name}, elo={self.elo_range}, "
                f"epochs={self.epochs}, value_weight={self.value_weight})")


class CurriculumDataset(Dataset):
    """
    Dataset that manages curriculum learning across multiple skill levels.
    Automatically switches between datasets based on training progress.
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None):
        """
        Args:
            curriculum_config: Configuration dict with stages or path to config file
        """
        self.stages: List[CurriculumStage] = []
        self.current_stage_idx = 0
        self.current_dataset = None
        self.total_epochs_trained = 0
        self.stage_epochs_trained = 0
        
        if curriculum_config:
            if isinstance(curriculum_config, str):
                # Load from file
                with open(curriculum_config, 'r') as f:
                    config = json.load(f)
            else:
                config = curriculum_config
                
            self._setup_stages(config)
        else:
            # Use default curriculum
            self._setup_default_curriculum()
    
    def _setup_default_curriculum(self):
        """Setup default 3-stage curriculum."""
        base_dir = 'games_training_data/curriculum'
        
        # Stage 1: Beginner (750-1500 ELO)
        self.stages.append(CurriculumStage(
            name='beginner',
            data_dir=os.path.join(base_dir, 'beginner'),
            elo_range=(750, 1500),
            epochs=80,
            value_weight=3.0,  # Higher weight on value loss to learn piece values
            soft_targets=True,
            temperature=0.15
        ))
        
        # Stage 2: Intermediate (1500-2400 ELO)
        self.stages.append(CurriculumStage(
            name='intermediate',
            data_dir=os.path.join(base_dir, 'intermediate'),
            elo_range=(1500, 2400),
            epochs=80,
            value_weight=2.5,
            soft_targets=True,
            temperature=0.12
        ))
        
        # Stage 3: Expert (2400-3000 ELO, strong humans and titled players)
        self.stages.append(CurriculumStage(
            name='expert',
            data_dir=os.path.join(base_dir, 'expert'),
            elo_range=(2400, 3000),
            epochs=150,
            value_weight=1.8,
            soft_targets=True,
            temperature=0.1
        ))
        
        # Stage 4: Computer (3000-4000 ELO, computer chess engines)
        self.stages.append(CurriculumStage(
            name='computer',
            data_dir=os.path.join(base_dir, 'computer'),
            elo_range=(3000, 4000),
            epochs=300,
            value_weight=0.7,  # Lower value weight for sophisticated play
            soft_targets=True,
            temperature=0.08
        ))
    
    def _setup_stages(self, config: Dict):
        """Setup stages from configuration."""
        for stage_config in config.get('stages', []):
            stage = CurriculumStage(**stage_config)
            self.stages.append(stage)
    
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get the current training stage."""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
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
            print(f"\nLoading curriculum stage: {current_stage.name}")
            print(f"  ELO range: {current_stage.elo_range}")
            print(f"  Epochs: {current_stage.epochs}")
            print(f"  Value weight: {current_stage.value_weight}")
            print(f"  Dataset size: {len(self.current_dataset)}")
    
    def on_epoch_end(self):
        """Called at the end of each training epoch."""
        self.total_epochs_trained += 1
        self.stage_epochs_trained += 1
        
        current_stage = self.get_current_stage()
        if current_stage:
            print(f"Stage '{current_stage.name}': Epoch {self.stage_epochs_trained}/{current_stage.epochs}")
    
    def get_value_weight(self) -> float:
        """Get the current value loss weight."""
        current_stage = self.get_current_stage()
        if current_stage:
            return current_stage.value_weight
        return 1.0
    
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
                'value_weight': current_stage.value_weight
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
            'stages': [
                {
                    'name': stage.name,
                    'data_dir': stage.data_dir,
                    'elo_range': stage.elo_range,
                    'epochs': stage.epochs,
                    'value_weight': stage.value_weight,
                    'soft_targets': stage.soft_targets,
                    'temperature': stage.temperature
                }
                for stage in self.stages
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Curriculum state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load curriculum state for resuming training."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage_idx = state['current_stage_idx']
        self.total_epochs_trained = state['total_epochs_trained']
        self.stage_epochs_trained = state['stage_epochs_trained']
        
        # Rebuild stages
        self.stages = []
        for stage_config in state['stages']:
            self.stages.append(CurriculumStage(**stage_config))
        
        self._load_current_stage()
        print(f"Curriculum state loaded from {filepath}")
        print(f"Resuming from stage '{self.get_current_stage().name}', "
              f"epoch {self.stage_epochs_trained}")


class MixedCurriculumDataset(Dataset):
    """
    Dataset that mixes games from different skill levels in controlled proportions.
    Useful for preventing catastrophic forgetting.
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None, 
                 mixing_ratio: Optional[Dict] = None):
        """
        Args:
            curriculum_config: Configuration for curriculum stages
            mixing_ratio: Dict mapping stage names to sampling ratios
                         e.g., {'beginner': 0.2, 'intermediate': 0.3, 'expert': 0.5}
        """
        self.curriculum = CurriculumDataset(curriculum_config)
        
        # Default mixing ratio
        self.mixing_ratio = mixing_ratio or {
            'beginner': 0.10,
            'intermediate': 0.10,
            'expert': 0.15,
            'computer': 0.65
        }
        
        # Load all datasets and store stages
        self.datasets = {}
        self.stages = {}  # Store stage objects for value_weight access
        for stage in self.curriculum.stages:
            if stage.name in self.mixing_ratio and self.mixing_ratio[stage.name] > 0:
                stage.load_dataset()
                self.datasets[stage.name] = stage.dataset
                self.stages[stage.name] = stage  # Store the stage object
        
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
        
        print(f"Mixed curriculum dataset created with {len(self.indices)} samples")
        for stage_name in self.datasets:
            count = sum(1 for name, _ in self.dataset_mapping if name == stage_name)
            print(f"  {stage_name}: {count} samples ({count/len(self.indices)*100:.1f}%)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stage_name, dataset_idx = self.dataset_mapping[idx]
        data = self.datasets[stage_name][dataset_idx]
        # Add value_weight information from the corresponding stage
        if isinstance(data, tuple):
            # Assuming data is (input, value_target, policy_target) or similar
            return data + (self.stages[stage_name].value_weight,)
        return data
    
    def get_value_weight_for_stage(self, stage_name: str) -> float:
        """Get the value weight for a specific stage."""
        if stage_name in self.stages:
            return self.stages[stage_name].value_weight
        return 1.0