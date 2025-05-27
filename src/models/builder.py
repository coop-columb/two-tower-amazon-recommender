"""
Model builder utilities for constructing two-tower models.

This module provides utilities for building and initializing two-tower models
with proper feature processing and vocabulary setup.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import TwoTowerConfig
from .two_tower import TwoTowerModel


class FeatureProcessor:
    """Processes features for two-tower model input."""

    def __init__(self, config: TwoTowerConfig):
        """Initialize feature processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def bucketize_continuous_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert continuous features to bucketized categorical features.
        
        Args:
            df: Input dataframe with continuous features
            
        Returns:
            Dataframe with added bucketized features
        """
        df = df.copy()

        # Bucketize user review counts
        if "user_review_count" in df.columns:
            df["user_review_count_bucket"] = pd.cut(
                df["user_review_count"],
                bins=[0] + self.config.data.user_review_count_buckets + [float("inf")],
                labels=range(len(self.config.data.user_review_count_buckets) + 1),
                include_lowest=True,
            ).astype(int)

        # Bucketize item review counts
        if "item_review_count" in df.columns:
            df["item_review_count_bucket"] = pd.cut(
                df["item_review_count"],
                bins=[0] + self.config.data.item_review_count_buckets + [float("inf")],
                labels=range(len(self.config.data.item_review_count_buckets) + 1),
                include_lowest=True,
            ).astype(int)

        # Bucketize user average ratings
        if "user_avg_rating" in df.columns:
            df["user_avg_rating_bucket"] = pd.cut(
                df["user_avg_rating"],
                bins=[0] + self.config.data.user_rating_buckets,
                labels=range(len(self.config.data.user_rating_buckets)),
                include_lowest=True,
            ).astype(int)

        # Bucketize item average ratings
        if "item_avg_rating" in df.columns:
            df["item_avg_rating_bucket"] = pd.cut(
                df["item_avg_rating"],
                bins=[0] + self.config.data.item_rating_buckets,
                labels=range(len(self.config.data.item_rating_buckets)),
                include_lowest=True,
            ).astype(int)

        return df

    def prepare_features(self, df: pd.DataFrame) -> dict[str, tf.Tensor]:
        """
        Prepare features for model input.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of features as tensorflow tensors
        """
        # Bucketize continuous features
        df_processed = self.bucketize_continuous_features(df)

        # Convert to tensors
        features = {}
        
        # Required features
        features["user_id"] = tf.constant(df_processed["user_id"].astype(str).values)
        features["item_id"] = tf.constant(df_processed["parent_asin"].astype(str).values)
        features["rating"] = tf.constant(df_processed["rating"].astype(np.float32).values)

        # Optional categorical features
        if "category" in df_processed.columns:
            features["category"] = tf.constant(df_processed["category"].astype(str).values)

        # Bucketized features
        bucket_features = [
            "user_review_count_bucket",
            "item_review_count_bucket", 
            "user_avg_rating_bucket",
            "item_avg_rating_bucket",
        ]
        
        for feature in bucket_features:
            if feature in df_processed.columns:
                features[feature] = tf.constant(df_processed[feature].astype(np.int32).values)

        # Temporal features if enabled
        if self.config.data.use_temporal_features:
            for temporal_feature in self.config.data.temporal_feature_names:
                if temporal_feature in df_processed.columns:
                    features[temporal_feature] = tf.constant(
                        df_processed[temporal_feature].astype(np.int32).values
                    )

        self.logger.info(f"Prepared {len(features)} features for model input")
        return features


class ModelBuilder:
    """Builds and initializes two-tower models."""

    def __init__(self, config: TwoTowerConfig):
        """Initialize model builder with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_data_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute data statistics needed for model configuration.
        
        Args:
            df: Training dataframe
            
        Returns:
            Dictionary of data statistics
        """
        stats = {
            "n_users": df["user_id"].nunique(),
            "n_items": df["parent_asin"].nunique(),
            "n_interactions": len(df),
            "rating_mean": df["rating"].mean(),
            "rating_std": df["rating"].std(),
        }

        # Category statistics if available
        if "category" in df.columns:
            stats["n_categories"] = df["category"].nunique()

        # Continuous feature statistics
        continuous_features = [
            "user_review_count", "item_review_count",
            "user_avg_rating", "item_avg_rating"
        ]
        
        for feature in continuous_features:
            if feature in df.columns:
                stats[f"{feature}_mean"] = df[feature].mean()
                stats[f"{feature}_std"] = df[feature].std()
                stats[f"{feature}_min"] = df[feature].min()
                stats[f"{feature}_max"] = df[feature].max()

        self.logger.info(f"Computed statistics for {stats['n_users']} users and {stats['n_items']} items")
        return stats

    def build_model(self, data_stats: dict[str, Any]) -> TwoTowerModel:
        """
        Build a two-tower model based on configuration and data statistics.
        
        Args:
            data_stats: Dictionary of data statistics
            
        Returns:
            Initialized TwoTowerModel
        """
        # Get vocabulary sizes and embedding dimensions
        user_vocab_sizes, item_vocab_sizes = self.config.get_vocabulary_sizes(data_stats)
        user_embedding_dims, item_embedding_dims = self.config.get_embedding_dimensions()

        self.logger.info(f"Building model with user vocab sizes: {user_vocab_sizes}")
        self.logger.info(f"Building model with item vocab sizes: {item_vocab_sizes}")

        # Create model
        model = TwoTowerModel(
            user_vocabulary_sizes=user_vocab_sizes,
            item_vocabulary_sizes=item_vocab_sizes,
            user_embedding_dimensions=user_embedding_dims,
            item_embedding_dimensions=item_embedding_dims,
            hidden_units=self.config.model.hidden_units,
            dropout_rate=self.config.model.dropout_rate,
            rating_weight=self.config.model.rating_weight,
            retrieval_weight=self.config.model.retrieval_weight,
        )

        self.logger.info("Successfully built two-tower model")
        return model

    def compile_model(self, model: TwoTowerModel) -> TwoTowerModel:
        """
        Compile model with optimizer and loss functions.
        
        Args:
            model: TwoTowerModel to compile
            
        Returns:
            Compiled model
        """
        # Create optimizer with learning rate schedule if configured
        if self.config.training.use_lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.ReduceLROnPlateau(
                factor=self.config.training.lr_decay_factor,
                patience=self.config.training.lr_patience,
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.training.learning_rate
            )

        # Compile model
        model.compile(optimizer=optimizer)
        
        self.logger.info(f"Compiled model with Adam optimizer (lr={self.config.training.learning_rate})")
        return model

    def create_callbacks(self, model_dir: Path) -> list:
        """
        Create training callbacks.
        
        Args:
            model_dir: Directory to save model checkpoints
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Model checkpointing
        checkpoint_path = model_dir / "checkpoints" / "model.{epoch:02d}-{val_loss:.2f}.weights.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=self.config.training.save_best_only,
            save_weights_only=self.config.training.save_weights_only,
            monitor=self.config.training.early_stopping_metric,
            mode=self.config.training.early_stopping_mode,
            verbose=1,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.training.early_stopping_metric,
            patience=self.config.training.early_stopping_patience,
            mode=self.config.training.early_stopping_mode,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        if self.config.training.use_lr_schedule:
            lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.training.lr_decay_factor,
                patience=self.config.training.lr_patience,
                min_lr=1e-7,
                verbose=1,
            )
            callbacks.append(lr_reduction)

        # TensorBoard logging
        tensorboard_dir = model_dir / "tensorboard"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
        )
        callbacks.append(tensorboard_callback)

        self.logger.info(f"Created {len(callbacks)} training callbacks")
        return callbacks

    def save_model_config(self, model_dir: Path) -> None:
        """
        Save model configuration to file.
        
        Args:
            model_dir: Directory to save configuration
        """
        import json
        from dataclasses import asdict

        config_path = model_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
            
        self.logger.info(f"Saved model configuration to {config_path}")


def build_model_from_config(
    config: TwoTowerConfig,
    train_df: pd.DataFrame,
    model_dir: Optional[Path] = None,
) -> tuple[TwoTowerModel, dict[str, Any], list]:
    """
    Build a complete model from configuration and training data.
    
    Args:
        config: Model configuration
        train_df: Training dataframe
        model_dir: Directory for model artifacts
        
    Returns:
        Tuple of (compiled_model, data_statistics, callbacks)
    """
    # Setup model directory
    if model_dir is None:
        model_dir = Path(config.model_dir) / config.experiment_name
    
    # Create builder
    builder = ModelBuilder(config)
    
    # Compute data statistics
    data_stats = builder.compute_data_statistics(train_df)
    
    # Build and compile model
    model = builder.build_model(data_stats)
    model = builder.compile_model(model)
    
    # Create callbacks
    callbacks = builder.create_callbacks(model_dir)
    
    # Save configuration
    builder.save_model_config(model_dir)
    
    return model, data_stats, callbacks