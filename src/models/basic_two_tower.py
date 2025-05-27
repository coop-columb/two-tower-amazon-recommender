"""
Basic two-tower model implementation with minimal dependencies.

This provides a very simple working model for initial development.
"""

import tensorflow as tf
from tensorflow import keras


class BasicTwoTowerModel(keras.Model):
    """Basic two-tower model for recommendation."""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int = 64,
        hidden_units: list[int] = [128, 64],
        **kwargs,
    ):
        """
        Initialize basic two-tower model.

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

        # Rating prediction head
        self.rating_head = keras.layers.Dense(1, activation="sigmoid", name="rating_head")

    def call(self, features: dict[str, tf.Tensor], training: bool = False) -> dict[str, tf.Tensor]:
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
            user_output = layer(user_output, training=training)
        user_output = self.user_projection(user_output)

        item_output = item_emb
        for layer in self.item_dense_layers:
            item_output = layer(item_output, training=training)
        item_output = self.item_projection(item_output)

        # Rating prediction (dot product + dense layer)
        interaction = user_output * item_output
        rating_pred = self.rating_head(interaction) * 5.0  # Scale to 1-5

        return {
            "user_embedding": user_output,
            "item_embedding": item_output,
            "rating_prediction": rating_pred,
        }

    def compute_loss(self, x, y, sample_weight=None):
        """Compute loss for training."""
        outputs = self(x, training=True)
        
        # Mean squared error for rating prediction
        rating_loss = keras.losses.mean_squared_error(
            y_true=x["rating"], 
            y_pred=tf.squeeze(outputs["rating_prediction"])
        )
        
        return tf.reduce_mean(rating_loss)

    def train_step(self, data):
        """Custom training step."""
        x, y = data if isinstance(data, tuple) else (data, None)
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(
            y_true=x["rating"], 
            y_pred=tf.squeeze(self(x, training=False)["rating_prediction"])
        )
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}


def create_basic_model(
    user_vocab_size: int,
    item_vocab_size: int,
    embedding_dim: int = 64,
    hidden_units: list[int] = [128, 64],
    learning_rate: float = 0.001,
) -> BasicTwoTowerModel:
    """
    Create and compile a basic two-tower model.
    
    Args:
        user_vocab_size: Number of unique users
        item_vocab_size: Number of unique items
        embedding_dim: Embedding dimension
        hidden_units: Hidden layer sizes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled BasicTwoTowerModel
    """
    model = BasicTwoTowerModel(
        user_vocab_size=user_vocab_size,
        item_vocab_size=item_vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        metrics=[keras.metrics.RootMeanSquaredError()]
    )
    
    return model