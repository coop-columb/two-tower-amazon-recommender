"""
Two-tower neural network architecture for recommendation systems.

This module implements a state-of-the-art two-tower model using TensorFlow Recommenders
for learning user and item representations in a shared embedding space.
"""

import logging
from typing import Any, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras


class UserTower(keras.Model):
    """User tower for encoding user features into embeddings."""

    def __init__(
        self,
        vocabulary_sizes: dict[str, int],
        embedding_dimensions: dict[str, int],
        hidden_units: list[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """
        Initialize the user tower.

        Args:
            vocabulary_sizes: Dictionary mapping feature names to vocabulary sizes
            embedding_dimensions: Dictionary mapping feature names to embedding dimensions
            hidden_units: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(**kwargs)
        self.vocabulary_sizes = vocabulary_sizes
        self.embedding_dimensions = embedding_dimensions
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        # User ID embedding
        self.user_embedding = keras.Sequential([
            keras.utils.StringLookup(
                vocabulary=None, mask_token=None, num_oov_indices=1
            ),
            keras.layers.Embedding(
                vocabulary_sizes["user_id"],
                embedding_dimensions["user_id"],
                name="user_id_embedding"
            ),
        ])

        # Feature embeddings for additional user features
        self.feature_embeddings = {}
        for feature_name in ["user_review_count_bucket", "user_avg_rating_bucket"]:
            if feature_name in vocabulary_sizes:
                self.feature_embeddings[feature_name] = keras.Sequential([
                    keras.utils.IntegerLookup(
                        vocabulary=None, mask_value=0, num_oov_indices=1
                    ),
                    keras.layers.Embedding(
                        vocabulary_sizes[feature_name],
                        embedding_dimensions.get(feature_name, 16),
                        name=f"{feature_name}_embedding"
                    ),
                ])

        # Dense layers for user tower
        self.dense_layers = []
        for i, units in enumerate(hidden_units):
            self.dense_layers.extend([
                keras.layers.Dense(units, activation="relu", name=f"user_dense_{i}"),
                keras.layers.Dropout(dropout_rate, name=f"user_dropout_{i}"),
            ])

        # Final output layer
        self.output_layer = keras.layers.Dense(
            hidden_units[-1], activation=None, name="user_output"
        )

        # Layer normalization
        self.layer_norm = keras.layers.LayerNormalization(name="user_layer_norm")

    def call(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through user tower.

        Args:
            features: Dictionary of user features
            training: Whether in training mode

        Returns:
            User embedding tensor
        """
        # Get user ID embedding
        user_emb = self.user_embedding(features["user_id"])

        # Collect all embeddings
        embeddings = [user_emb]

        # Add feature embeddings if available
        for feature_name, embedding_layer in self.feature_embeddings.items():
            if feature_name in features:
                feat_emb = embedding_layer(features[feature_name])
                embeddings.append(feat_emb)

        # Concatenate all embeddings
        if len(embeddings) > 1:
            concatenated = keras.layers.Concatenate()(embeddings)
        else:
            concatenated = embeddings[0]

        # Pass through dense layers
        x = concatenated
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # Final output with layer normalization
        output = self.output_layer(x)
        return self.layer_norm(output)


class ItemTower(keras.Model):
    """Item tower for encoding item features into embeddings."""

    def __init__(
        self,
        vocabulary_sizes: dict[str, int],
        embedding_dimensions: dict[str, int],
        hidden_units: list[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """
        Initialize the item tower.

        Args:
            vocabulary_sizes: Dictionary mapping feature names to vocabulary sizes
            embedding_dimensions: Dictionary mapping feature names to embedding dimensions
            hidden_units: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(**kwargs)
        self.vocabulary_sizes = vocabulary_sizes
        self.embedding_dimensions = embedding_dimensions
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        # Item ID embedding
        self.item_embedding = keras.Sequential([
            keras.utils.StringLookup(
                vocabulary=None, mask_token=None, num_oov_indices=1
            ),
            keras.layers.Embedding(
                vocabulary_sizes["item_id"],
                embedding_dimensions["item_id"],
                name="item_id_embedding"
            ),
        ])

        # Category embedding
        if "category" in vocabulary_sizes:
            self.category_embedding = keras.Sequential([
                keras.utils.StringLookup(
                    vocabulary=None, mask_token=None, num_oov_indices=1
                ),
                keras.layers.Embedding(
                    vocabulary_sizes["category"],
                    embedding_dimensions.get("category", 32),
                    name="category_embedding"
                ),
            ])
        else:
            self.category_embedding = None

        # Feature embeddings for additional item features
        self.feature_embeddings = {}
        for feature_name in ["item_review_count_bucket", "item_avg_rating_bucket"]:
            if feature_name in vocabulary_sizes:
                self.feature_embeddings[feature_name] = keras.Sequential([
                    keras.utils.IntegerLookup(
                        vocabulary=None, mask_value=0, num_oov_indices=1
                    ),
                    keras.layers.Embedding(
                        vocabulary_sizes[feature_name],
                        embedding_dimensions.get(feature_name, 16),
                        name=f"{feature_name}_embedding"
                    ),
                ])

        # Dense layers for item tower
        self.dense_layers = []
        for i, units in enumerate(hidden_units):
            self.dense_layers.extend([
                keras.layers.Dense(units, activation="relu", name=f"item_dense_{i}"),
                keras.layers.Dropout(dropout_rate, name=f"item_dropout_{i}"),
            ])

        # Final output layer
        self.output_layer = keras.layers.Dense(
            hidden_units[-1], activation=None, name="item_output"
        )

        # Layer normalization
        self.layer_norm = keras.layers.LayerNormalization(name="item_layer_norm")

    def call(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass through item tower.

        Args:
            features: Dictionary of item features
            training: Whether in training mode

        Returns:
            Item embedding tensor
        """
        # Get item ID embedding
        item_emb = self.item_embedding(features["item_id"])

        # Collect all embeddings
        embeddings = [item_emb]

        # Add category embedding if available
        if self.category_embedding is not None and "category" in features:
            cat_emb = self.category_embedding(features["category"])
            embeddings.append(cat_emb)

        # Add feature embeddings if available
        for feature_name, embedding_layer in self.feature_embeddings.items():
            if feature_name in features:
                feat_emb = embedding_layer(features[feature_name])
                embeddings.append(feat_emb)

        # Concatenate all embeddings
        if len(embeddings) > 1:
            concatenated = keras.layers.Concatenate()(embeddings)
        else:
            concatenated = embeddings[0]

        # Pass through dense layers
        x = concatenated
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # Final output with layer normalization
        output = self.output_layer(x)
        return self.layer_norm(output)


class TwoTowerModel(tfrs.Model):
    """
    Complete two-tower recommendation model.
    
    Combines user and item towers with retrieval and ranking tasks.
    """

    def __init__(
        self,
        user_vocabulary_sizes: dict[str, int],
        item_vocabulary_sizes: dict[str, int],
        user_embedding_dimensions: dict[str, int],
        item_embedding_dimensions: dict[str, int],
        hidden_units: list[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        rating_weight: float = 1.0,
        retrieval_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the two-tower model.

        Args:
            user_vocabulary_sizes: User feature vocabulary sizes
            item_vocabulary_sizes: Item feature vocabulary sizes
            user_embedding_dimensions: User feature embedding dimensions
            item_embedding_dimensions: Item feature embedding dimensions
            hidden_units: Hidden layer sizes for both towers
            dropout_rate: Dropout rate for regularization
            rating_weight: Weight for rating prediction task
            retrieval_weight: Weight for retrieval task
        """
        super().__init__(**kwargs)

        # Store configuration
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

        # Initialize towers
        self.user_tower = UserTower(
            vocabulary_sizes=user_vocabulary_sizes,
            embedding_dimensions=user_embedding_dimensions,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )

        self.item_tower = ItemTower(
            vocabulary_sizes=item_vocabulary_sizes,
            embedding_dimensions=item_embedding_dimensions,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )

        # Rating prediction task
        self.rating_task = tfrs.tasks.Ranking(
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
            prediction_layer=keras.Sequential([
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(1, activation="sigmoid"),
                keras.layers.Lambda(lambda x: x * 5.0)  # Scale to 1-5 rating
            ])
        )

        # Retrieval task for candidate item retrieval
        item_candidates = item_vocabulary_sizes["item_id"]
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.range(item_candidates).batch(1000).map(
                    lambda x: tf.strings.as_string(x)
                )
            )
        )

    def call(self, features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            features: Dictionary containing user and item features

        Returns:
            Dictionary with user embeddings, item embeddings, and ratings
        """
        # Extract user and item features
        user_features = {k: v for k, v in features.items() 
                        if k.startswith("user_") or k == "user_id"}
        item_features = {k: v for k, v in features.items() 
                        if k.startswith("item_") or k in ["item_id", "category"]}

        # Get embeddings from towers
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)

        # Compute rating prediction
        rating_prediction = self.rating_task(
            query_embeddings=user_embedding,
            candidate_embeddings=item_embedding,
        )

        return {
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
            "rating_prediction": rating_prediction,
        }

    def compute_loss(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Compute the total loss for training.

        Args:
            features: Dictionary containing features and labels
            training: Whether in training mode

        Returns:
            Total loss tensor
        """
        # Get model outputs
        outputs = self(features, training=training)

        # Compute rating loss
        rating_loss = self.rating_task(
            query_embeddings=outputs["user_embedding"],
            candidate_embeddings=outputs["item_embedding"],
            true_ratings=features["rating"],
        )

        # Compute retrieval loss
        retrieval_loss = self.retrieval_task(
            query_embeddings=outputs["user_embedding"],
            candidate_identifiers=features["item_id"],
        )

        # Total loss
        total_loss = (
            self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss
        )

        return total_loss