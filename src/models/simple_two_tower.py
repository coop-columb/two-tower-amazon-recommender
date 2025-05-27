"""
Simplified two-tower model implementation that works with current TF/TFR versions.

This provides a working baseline that can be extended once we resolve API compatibility.
"""

import logging
from typing import Any, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras


class SimpleTwoTowerModel(tfrs.Model):
    """Simplified two-tower model for initial development."""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int = 64,
        hidden_units: list[int] = [128, 64],
        **kwargs,
    ):
        """
        Initialize simplified two-tower model.

        Args:
            user_vocab_size: Number of unique users
            item_vocab_size: Number of unique items  
            embedding_dim: Embedding dimension for users and items
            hidden_units: Hidden layer sizes
        """
        super().__init__(**kwargs)
        
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.embedding_dim = embedding_dim

        # User tower
        self.user_embedding = keras.layers.Embedding(
            user_vocab_size, embedding_dim, name="user_embedding"
        )
        
        # Item tower
        self.item_embedding = keras.layers.Embedding(
            item_vocab_size, embedding_dim, name="item_embedding"
        )

        # Dense layers for user tower
        self.user_dense_layers = []
        for i, units in enumerate(hidden_units):
            self.user_dense_layers.append(
                keras.layers.Dense(units, activation="relu", name=f"user_dense_{i}")
            )

        # Dense layers for item tower
        self.item_dense_layers = []
        for i, units in enumerate(hidden_units):
            self.item_dense_layers.append(
                keras.layers.Dense(units, activation="relu", name=f"item_dense_{i}")
            )

        # Final projection layers
        self.user_projection = keras.layers.Dense(
            embedding_dim, activation=None, name="user_projection"
        )
        self.item_projection = keras.layers.Dense(
            embedding_dim, activation=None, name="item_projection"
        )

        # Tasks
        self.rating_task = tfrs.tasks.Ranking(
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        
        # Build candidate dataset for retrieval
        candidate_identifiers = tf.data.Dataset.range(item_vocab_size).batch(1000).map(
            lambda x: tf.strings.as_string(x)
        )
        
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_identifiers
            )
        )

    def call(self, features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Forward pass through the model."""
        # Convert string IDs to integers for embedding lookup
        user_ids = tf.strings.to_number(features["user_id"], out_type=tf.int32)
        item_ids = tf.strings.to_number(features["item_id"], out_type=tf.int32)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Pass through dense layers
        user_output = user_emb
        for layer in self.user_dense_layers:
            user_output = layer(user_output)
        user_output = self.user_projection(user_output)

        item_output = item_emb
        for layer in self.item_dense_layers:
            item_output = layer(item_output)
        item_output = self.item_projection(item_output)

        return {
            "user_embedding": user_output,
            "item_embedding": item_output,
        }

    def compute_loss(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Compute loss for training."""
        # Get embeddings
        model_outputs = self(features, training=training)
        user_embeddings = model_outputs["user_embedding"]
        item_embeddings = model_outputs["item_embedding"]

        # Rating prediction loss
        rating_predictions = tf.reduce_sum(
            user_embeddings * item_embeddings, axis=1, keepdims=True
        )
        rating_loss = self.rating_task(
            labels=features["rating"],
            predictions=rating_predictions,
        )

        # Retrieval loss
        retrieval_loss = self.retrieval_task(
            query_embeddings=user_embeddings,
            candidate_identifiers=features["item_id"],
        )

        # Total loss
        total_loss = rating_loss + retrieval_loss

        return total_loss


def create_simple_model(
    user_vocab_size: int,
    item_vocab_size: int,
    embedding_dim: int = 64,
    hidden_units: list[int] = [128, 64],
    learning_rate: float = 0.001,
) -> SimpleTwoTowerModel:
    """
    Create and compile a simple two-tower model.
    
    Args:
        user_vocab_size: Number of unique users
        item_vocab_size: Number of unique items
        embedding_dim: Embedding dimension
        hidden_units: Hidden layer sizes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled SimpleTwoTowerModel
    """
    model = SimpleTwoTowerModel(
        user_vocab_size=user_vocab_size,
        item_vocab_size=item_vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
    )
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate))
    
    return model