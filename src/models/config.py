"""
Model configuration classes for two-tower recommendation system.

This module provides configuration classes and utilities for model hyperparameters,
architecture settings, and training configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layers."""
    
    user_id_dim: int = 128
    item_id_dim: int = 128
    category_dim: int = 32
    feature_dim: int = 16
    
    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format expected by model."""
        return {
            "user_id": self.user_id_dim,
            "item_id": self.item_id_dim,
            "category": self.category_dim,
            "user_review_count_bucket": self.feature_dim,
            "user_avg_rating_bucket": self.feature_dim,
            "item_review_count_bucket": self.feature_dim,
            "item_avg_rating_bucket": self.feature_dim,
        }


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    
    # Tower architecture
    hidden_units: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    
    # Task weights
    rating_weight: float = 1.0
    retrieval_weight: float = 1.0
    
    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Optimization
    learning_rate: float = 0.001
    batch_size: int = 1024
    epochs: int = 50
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_decay_factor: float = 0.8
    lr_patience: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    
    # Checkpointing
    save_best_only: bool = True
    save_weights_only: bool = False
    
    # Validation
    validation_split: float = 0.1
    shuffle_buffer_size: int = 10000


@dataclass
class DataConfig:
    """Configuration for data processing specific to model needs."""
    
    # Feature buckets for continuous variables
    user_review_count_buckets: list[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500]
    )
    item_review_count_buckets: list[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500]
    )
    user_rating_buckets: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    item_rating_buckets: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    
    # Text features
    max_text_length: int = 512
    text_vocab_size: int = 10000
    
    # Temporal features
    use_temporal_features: bool = True
    temporal_feature_names: list[str] = field(
        default_factory=lambda: ["hour", "day_of_week", "month", "is_weekend"]
    )


@dataclass
class TwoTowerConfig:
    """Complete configuration for two-tower model."""
    
    # Model architecture
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    
    # Training settings
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data processing
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment tracking
    experiment_name: str = "two_tower_baseline"
    run_name: Optional[str] = None
    
    # Model saving
    model_dir: str = "models/experiments"
    
    def get_vocabulary_sizes(self, data_stats: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
        """
        Extract vocabulary sizes for user and item features from data statistics.
        
        Args:
            data_stats: Dictionary containing dataset statistics
            
        Returns:
            Tuple of (user_vocab_sizes, item_vocab_sizes)
        """
        user_vocab_sizes = {
            "user_id": data_stats["n_users"],
            "user_review_count_bucket": len(self.data.user_review_count_buckets) + 1,
            "user_avg_rating_bucket": len(self.data.user_rating_buckets) + 1,
        }
        
        item_vocab_sizes = {
            "item_id": data_stats["n_items"],
            "category": data_stats.get("n_categories", 50),  # Default if not provided
            "item_review_count_bucket": len(self.data.item_review_count_buckets) + 1,
            "item_avg_rating_bucket": len(self.data.item_rating_buckets) + 1,
        }
        
        return user_vocab_sizes, item_vocab_sizes
    
    def get_embedding_dimensions(self) -> tuple[dict[str, int], dict[str, int]]:
        """
        Get embedding dimensions for user and item features.
        
        Returns:
            Tuple of (user_embedding_dims, item_embedding_dims)
        """
        embedding_dict = self.model.embedding.to_dict()
        
        user_embedding_dims = {
            "user_id": embedding_dict["user_id"],
            "user_review_count_bucket": embedding_dict["user_review_count_bucket"],
            "user_avg_rating_bucket": embedding_dict["user_avg_rating_bucket"],
        }
        
        item_embedding_dims = {
            "item_id": embedding_dict["item_id"],
            "category": embedding_dict["category"],
            "item_review_count_bucket": embedding_dict["item_review_count_bucket"],
            "item_avg_rating_bucket": embedding_dict["item_avg_rating_bucket"],
        }
        
        return user_embedding_dims, item_embedding_dims


def create_baseline_config() -> TwoTowerConfig:
    """Create a baseline configuration for initial experiments."""
    return TwoTowerConfig(
        model=ModelArchitectureConfig(
            hidden_units=[256, 128, 64],
            dropout_rate=0.2,
            rating_weight=1.0,
            retrieval_weight=1.0,
            embedding=EmbeddingConfig(
                user_id_dim=128,
                item_id_dim=128,
                category_dim=32,
                feature_dim=16,
            )
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            batch_size=1024,
            epochs=50,
            validation_split=0.1,
        ),
        experiment_name="two_tower_baseline"
    )


def create_large_config() -> TwoTowerConfig:
    """Create a larger configuration for production experiments."""
    return TwoTowerConfig(
        model=ModelArchitectureConfig(
            hidden_units=[512, 256, 128, 64],
            dropout_rate=0.3,
            rating_weight=1.0,
            retrieval_weight=1.5,
            embedding=EmbeddingConfig(
                user_id_dim=256,
                item_id_dim=256,
                category_dim=64,
                feature_dim=32,
            )
        ),
        training=TrainingConfig(
            learning_rate=0.0005,
            batch_size=2048,
            epochs=100,
            validation_split=0.1,
            early_stopping_patience=15,
        ),
        experiment_name="two_tower_large"
    )