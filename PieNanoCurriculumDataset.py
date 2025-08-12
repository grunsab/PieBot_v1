"""
Curriculum Dataset for progressive PieNano V2 chess training.
Optimized for the enhanced 8M parameter architecture with powerful policy head.
"""

import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from CCRLDataset import CCRLDataset
import json
from typing import Dict, List, Tuple, Optional

class PieNanoCurriculumStage:
    """Represents a single stage in the PieNano V2 curriculum training."""
    
    def __init__(self, name: str, data_dir: str, elo_range: Tuple[int, int], 
                 epochs: int, value_weight: float = 1.0, 
                 soft_targets: bool = True, temperature: float = 0.1,
                 lr_multiplier: float = 1.0, enhanced_encoder: bool = False):
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
            enhanced_encoder: Use enhanced encoder with 112 planes
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
        return (f"PieNanoCurriculumStage(name={self.name}, elo={self.elo_range}, "
                f"epochs={self.epochs}, value_weight={self.value_weight}, "
                f"lr_mult={self.lr_multiplier})")


class PieNanoCurriculumDataset(Dataset):
    """
    Dataset that manages curriculum learning for PieNano V2 model.
    Optimized for the 20x256 architecture with 768-dim policy head (~8M params).
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None, enhanced_encoder: bool = False):
        """
        Args:
            curriculum_config: Configuration dict with stages or path to config file
            enhanced_encoder: Use enhanced encoder with 112 planes
        """
        self.stages: List[PieNanoCurriculumStage] = []
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
            # Use default curriculum optimized for PieNano V2
            self._setup_default_pie_nano_curriculum()
    
    def _setup_default_pie_nano_curriculum(self):
        """
        Setup default 4-stage curriculum optimized for PieNano V2's architecture.
        Balanced training duration for 8M parameter model.
        """
        base_dir = 'games_training_data/curriculum'
        
        # Stage 1: Beginner (750-1500 ELO)
        # Focus on learning basic piece values and simple tactics
        self.stages.append(PieNanoCurriculumStage(
            name='beginner',
            data_dir=os.path.join(base_dir, 'beginner'),
            elo_range=(750, 1500),
            epochs=80,  # Moderate epochs for foundation
            value_weight=2.0,  # Strong emphasis on value learning
            soft_targets=True,
            temperature=0.15,  # Moderate temperature for smooth learning
            lr_multiplier=0.7,  # Slightly lower LR for stability
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 2: Intermediate (1500-2400 ELO)
        # Develop positional understanding and tactical patterns
        self.stages.append(PieNanoCurriculumStage(
            name='intermediate',
            data_dir=os.path.join(base_dir, 'intermediate'),
            elo_range=(1500, 2400),
            epochs=80,  # More epochs for complex patterns
            value_weight=1.5,
            soft_targets=True,
            temperature=0.12,
            lr_multiplier=1.0,  # Normal learning rate
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 3: Expert (2400-3000 ELO)
        # Master advanced strategies and deep calculation
        self.stages.append(PieNanoCurriculumStage(
            name='expert',
            data_dir=os.path.join(base_dir, 'expert'),
            elo_range=(2400, 3000),
            epochs=120,  # Extended training for expert play
            value_weight=1.0,
            soft_targets=True,
            temperature=0.10,
            lr_multiplier=0.8,  # Slightly reduced LR for fine-tuning
            enhanced_encoder=self.enhanced_encoder
        ))
        
        # Stage 4: Computer (3000-4000 ELO)
        # Refine to engine-level play with sophisticated evaluation
        self.stages.append(PieNanoCurriculumStage(
            name='computer',
            data_dir=os.path.join(base_dir, 'computer'),
            elo_range=(3000, 4000),
            epochs=120,  # Extensive training for peak performance
            value_weight=0.7,  # Focus more on policy for sophisticated play
            soft_targets=True,
            temperature=0.08,  # Lower temperature for sharper decisions
            lr_multiplier=0.6,  # Lower LR for careful refinement
            enhanced_encoder=self.enhanced_encoder
        ))
    
    def _setup_stages(self, config: Dict):
        """Setup stages from configuration dictionary."""
        for stage_config in config.get('stages', []):
            stage = PieNanoCurriculumStage(
                name=stage_config['name'],
                data_dir=stage_config['data_dir'],
                elo_range=tuple(stage_config['elo_range']),
                epochs=stage_config['epochs'],
                value_weight=stage_config.get('value_weight', 1.0),
                soft_targets=stage_config.get('soft_targets', True),
                temperature=stage_config.get('temperature', 0.1),
                lr_multiplier=stage_config.get('lr_multiplier', 1.0),
                enhanced_encoder=stage_config.get('enhanced_encoder', self.enhanced_encoder)
            )
            self.stages.append(stage)
    
    def get_current_stage(self) -> PieNanoCurriculumStage:
        """Get the current training stage."""
        if self.current_stage_idx >= len(self.stages):
            return None
        return self.stages[self.current_stage_idx]
    
    def advance_stage(self):
        """Advance to the next training stage."""
        self.current_stage_idx += 1
        self.stage_epochs_trained = 0
        
        if self.current_stage_idx < len(self.stages):
            stage = self.stages[self.current_stage_idx]
            print(f"\n{'='*60}")
            print(f"Advancing to stage: {stage.name}")
            print(f"ELO range: {stage.elo_range[0]}-{stage.elo_range[1]}")
            print(f"Training for {stage.epochs} epochs")
            print(f"Value weight: {stage.value_weight}")
            print(f"Learning rate multiplier: {stage.lr_multiplier}")
            print(f"{'='*60}\n")
            
            # Load the new stage's dataset
            self.current_dataset = stage.load_dataset()
            return True
        else:
            print("\n" + "="*60)
            print("Curriculum training complete!")
            print("="*60 + "\n")
            return False
    
    def should_advance(self) -> bool:
        """Check if it's time to advance to the next stage."""
        if self.current_stage_idx >= len(self.stages):
            return False
        
        current_stage = self.stages[self.current_stage_idx]
        return self.stage_epochs_trained >= current_stage.epochs
    
    def increment_epoch(self):
        """Increment epoch counters."""
        self.stage_epochs_trained += 1
        self.total_epochs_trained += 1
    
    def load_initial_stage(self):
        """Load the first stage's dataset."""
        if len(self.stages) == 0:
            raise ValueError("No stages configured")
        
        stage = self.stages[0]
        print(f"Starting curriculum with stage: {stage.name}")
        print(f"ELO range: {stage.elo_range[0]}-{stage.elo_range[1]}")
        print(f"Training for {stage.epochs} epochs")
        
        self.current_dataset = stage.load_dataset()
        return self.current_dataset
    
    def save_state(self, filepath: str):
        """Save current training state."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'total_epochs_trained': self.total_epochs_trained,
            'stage_epochs_trained': self.stage_epochs_trained,
            'enhanced_encoder': self.enhanced_encoder
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved curriculum state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load training state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage_idx = state['current_stage_idx']
        self.total_epochs_trained = state['total_epochs_trained']
        self.stage_epochs_trained = state['stage_epochs_trained']
        
        # Load the current stage's dataset
        if self.current_stage_idx < len(self.stages):
            stage = self.stages[self.current_stage_idx]
            self.current_dataset = stage.load_dataset()
            print(f"Resumed at stage: {stage.name}")
            print(f"Stage progress: {self.stage_epochs_trained}/{stage.epochs} epochs")
    
    def __len__(self):
        """Return length of current dataset."""
        if self.current_dataset is None:
            self.load_initial_stage()
        return len(self.current_dataset)
    
    def __getitem__(self, idx):
        """Get item from current dataset."""
        if self.current_dataset is None:
            self.load_initial_stage()
        return self.current_dataset[idx]
    
    def get_training_info(self) -> Dict:
        """Get current training information."""
        stage = self.get_current_stage()
        if stage is None:
            return {
                'completed': True,
                'total_epochs': self.total_epochs_trained
            }
        
        return {
            'stage_name': stage.name,
            'stage_idx': self.current_stage_idx,
            'total_stages': len(self.stages),
            'stage_epochs': self.stage_epochs_trained,
            'stage_total_epochs': stage.epochs,
            'total_epochs': self.total_epochs_trained,
            'elo_range': stage.elo_range,
            'value_weight': stage.value_weight,
            'lr_multiplier': stage.lr_multiplier
        }


class PieNanoMixedCurriculumDataset(Dataset):
    """
    Mixed curriculum dataset that combines multiple stages simultaneously.
    Useful for more robust training with varied difficulty levels.
    """
    
    def __init__(self, curriculum_config: Optional[Dict] = None, enhanced_encoder: bool = False):
        """
        Args:
            curriculum_config: Configuration dict or path to config file
            enhanced_encoder: Use enhanced encoder with 112 planes
        """
        self.enhanced_encoder = enhanced_encoder
        self.datasets = []
        self.weights = []
        
        if curriculum_config:
            if isinstance(curriculum_config, str):
                with open(curriculum_config, 'r') as f:
                    config = json.load(f)
            else:
                config = curriculum_config
            
            self._setup_mixed_stages(config)
        else:
            self._setup_default_mixed()
    
    def _setup_default_mixed(self):
        """Setup default mixed curriculum with weighted sampling."""
        base_dir = 'games_training_data/curriculum'
        
        # Load all stages with different weights
        stages_config = [
            ('beginner', 0.15),     # 15% from beginner games
            ('intermediate', 0.30), # 30% from intermediate
            ('expert', 0.35),       # 35% from expert
            ('computer', 0.20)      # 20% from computer games
        ]
        
        for stage_name, weight in stages_config:
            data_dir = os.path.join(base_dir, stage_name)
            if os.path.exists(data_dir):
                dataset = CCRLDataset(
                    data_dir,
                    soft_targets=True,
                    temperature=0.1,
                    enhanced_encoder=self.enhanced_encoder
                )
                self.datasets.append(dataset)
                self.weights.append(weight)
                print(f"Loaded {stage_name} dataset with weight {weight:.2%}")
    
    def _setup_mixed_stages(self, config: Dict):
        """Setup mixed stages from configuration."""
        for stage_config in config.get('mixed_stages', []):
            dataset = CCRLDataset(
                stage_config['data_dir'],
                soft_targets=stage_config.get('soft_targets', True),
                temperature=stage_config.get('temperature', 0.1),
                enhanced_encoder=stage_config.get('enhanced_encoder', self.enhanced_encoder)
            )
            self.datasets.append(dataset)
            self.weights.append(stage_config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def __len__(self):
        """Return total length of all datasets."""
        return sum(len(d) for d in self.datasets)
    
    def __getitem__(self, idx):
        """Sample from datasets according to weights."""
        # Use weighted random sampling
        dataset_idx = torch.multinomial(torch.tensor(self.weights), 1).item()
        dataset = self.datasets[dataset_idx]
        
        # Random sample from selected dataset
        sample_idx = torch.randint(0, len(dataset), (1,)).item()
        return dataset[sample_idx]