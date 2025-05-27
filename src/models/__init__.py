"""
Two-tower recommendation model implementation.

This package provides a complete implementation of a two-tower neural network
architecture for recommendation systems using TensorFlow Recommenders.
"""

from .builder import FeatureProcessor, ModelBuilder, build_model_from_config
from .config import (
    DataConfig,
    EmbeddingConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    TwoTowerConfig,
    create_baseline_config,
    create_large_config,
)
from .simple_two_tower import SimpleTwoTowerModel, create_simple_model
from .two_tower import ItemTower, TwoTowerModel, UserTower

__all__ = [
    # Model components
    "UserTower",
    "ItemTower", 
    "TwoTowerModel",
    "SimpleTwoTowerModel",
    "create_simple_model",
    # Configuration
    "EmbeddingConfig",
    "ModelArchitectureConfig",
    "TrainingConfig",
    "DataConfig",
    "TwoTowerConfig",
    "create_baseline_config",
    "create_large_config",
    # Builder utilities
    "FeatureProcessor",
    "ModelBuilder",
    "build_model_from_config",
]