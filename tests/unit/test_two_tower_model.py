"""Unit tests for two-tower model components."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.models import (
    TwoTowerConfig,
    TwoTowerModel,
    UserTower,
    ItemTower,
    create_baseline_config,
    build_model_from_config,
    FeatureProcessor,
    ModelBuilder,
)


class TestUserTower:
    """Test UserTower model component."""

    @pytest.fixture
    def user_tower(self):
        """Create a UserTower instance for testing."""
        vocabulary_sizes = {"user_id": 1000}
        embedding_dimensions = {"user_id": 64}
        return UserTower(
            vocabulary_sizes=vocabulary_sizes,
            embedding_dimensions=embedding_dimensions,
            hidden_units=[128, 64],
            dropout_rate=0.1,
        )

    def test_user_tower_initialization(self, user_tower):
        """Test UserTower initializes correctly."""
        assert user_tower.vocabulary_sizes["user_id"] == 1000
        assert user_tower.embedding_dimensions["user_id"] == 64
        assert user_tower.hidden_units == [128, 64]
        assert user_tower.dropout_rate == 0.1

    def test_user_tower_forward_pass(self, user_tower):
        """Test UserTower forward pass."""
        # Create mock user features
        features = {
            "user_id": tf.constant(["user_1", "user_2", "user_3"])
        }
        
        # Forward pass
        output = user_tower(features, training=False)
        
        # Check output shape
        assert output.shape == (3, 64)  # batch_size=3, embedding_dim=64
        assert output.dtype == tf.float32


class TestItemTower:
    """Test ItemTower model component."""

    @pytest.fixture
    def item_tower(self):
        """Create an ItemTower instance for testing."""
        vocabulary_sizes = {"item_id": 5000, "category": 50}
        embedding_dimensions = {"item_id": 64, "category": 32}
        return ItemTower(
            vocabulary_sizes=vocabulary_sizes,
            embedding_dimensions=embedding_dimensions,
            hidden_units=[128, 64],
            dropout_rate=0.1,
        )

    def test_item_tower_initialization(self, item_tower):
        """Test ItemTower initializes correctly."""
        assert item_tower.vocabulary_sizes["item_id"] == 5000
        assert item_tower.vocabulary_sizes["category"] == 50
        assert item_tower.hidden_units == [128, 64]

    def test_item_tower_forward_pass(self, item_tower):
        """Test ItemTower forward pass."""
        # Create mock item features
        features = {
            "item_id": tf.constant(["item_1", "item_2", "item_3"]),
            "category": tf.constant(["Books", "Electronics", "Sports"])
        }
        
        # Forward pass
        output = item_tower(features, training=False)
        
        # Check output shape
        assert output.shape == (3, 64)  # batch_size=3, embedding_dim=64
        assert output.dtype == tf.float32


class TestTwoTowerModel:
    """Test complete TwoTowerModel."""

    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing."""
        user_vocab_sizes = {"user_id": 1000}
        item_vocab_sizes = {"item_id": 5000, "category": 50}
        user_embedding_dims = {"user_id": 64}
        item_embedding_dims = {"item_id": 64, "category": 32}
        
        return {
            "user_vocabulary_sizes": user_vocab_sizes,
            "item_vocabulary_sizes": item_vocab_sizes,
            "user_embedding_dimensions": user_embedding_dims,
            "item_embedding_dimensions": item_embedding_dims,
            "hidden_units": [128, 64],
            "dropout_rate": 0.1,
        }

    def test_two_tower_model_initialization(self, model_config):
        """Test TwoTowerModel initializes correctly."""
        model = TwoTowerModel(**model_config)
        
        assert model.rating_weight == 1.0
        assert model.retrieval_weight == 1.0
        assert isinstance(model.user_tower, UserTower)
        assert isinstance(model.item_tower, ItemTower)

    def test_two_tower_model_call(self, model_config):
        """Test TwoTowerModel forward pass."""
        model = TwoTowerModel(**model_config)
        
        # Create mock features
        features = {
            "user_id": tf.constant(["user_1", "user_2"]),
            "item_id": tf.constant(["item_1", "item_2"]),
            "category": tf.constant(["Books", "Electronics"]),
            "rating": tf.constant([4.0, 5.0])
        }
        
        # Forward pass
        outputs = model(features)
        
        # Check outputs
        assert "user_embedding" in outputs
        assert "item_embedding" in outputs
        assert "rating_prediction" in outputs
        
        assert outputs["user_embedding"].shape == (2, 64)
        assert outputs["item_embedding"].shape == (2, 64)


class TestTwoTowerConfig:
    """Test TwoTowerConfig configuration class."""

    def test_baseline_config_creation(self):
        """Test creating baseline configuration."""
        config = create_baseline_config()
        
        assert config.model.hidden_units == [256, 128, 64]
        assert config.model.dropout_rate == 0.2
        assert config.training.learning_rate == 0.001
        assert config.training.batch_size == 1024
        assert config.experiment_name == "two_tower_baseline"

    def test_config_vocabulary_sizes(self):
        """Test vocabulary size extraction."""
        config = create_baseline_config()
        data_stats = {
            "n_users": 10000,
            "n_items": 50000,
            "n_categories": 25,
        }
        
        user_vocab, item_vocab = config.get_vocabulary_sizes(data_stats)
        
        assert user_vocab["user_id"] == 10000
        assert item_vocab["item_id"] == 50000
        assert item_vocab["category"] == 25

    def test_config_embedding_dimensions(self):
        """Test embedding dimension extraction."""
        config = create_baseline_config()
        
        user_dims, item_dims = config.get_embedding_dimensions()
        
        assert user_dims["user_id"] == 128
        assert item_dims["item_id"] == 128
        assert item_dims["category"] == 32


class TestFeatureProcessor:
    """Test FeatureProcessor utility."""

    @pytest.fixture
    def processor(self):
        """Create FeatureProcessor for testing."""
        config = create_baseline_config()
        return FeatureProcessor(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "parent_asin": ["i1", "i2", "i3"],
            "rating": [4.0, 5.0, 3.0],
            "category": ["Books", "Electronics", "Sports"],
            "user_review_count": [10, 50, 200],
            "item_review_count": [5, 25, 100],
            "user_avg_rating": [4.2, 3.8, 4.5],
            "item_avg_rating": [4.0, 4.1, 3.9],
        })

    def test_bucketize_continuous_features(self, processor, sample_data):
        """Test continuous feature bucketization."""
        processed = processor.bucketize_continuous_features(sample_data)
        
        # Check that bucketized columns are added
        assert "user_review_count_bucket" in processed.columns
        assert "item_review_count_bucket" in processed.columns
        assert "user_avg_rating_bucket" in processed.columns
        assert "item_avg_rating_bucket" in processed.columns
        
        # Check data types
        assert processed["user_review_count_bucket"].dtype == int
        assert processed["item_review_count_bucket"].dtype == int

    def test_prepare_features(self, processor, sample_data):
        """Test feature preparation for model input."""
        features = processor.prepare_features(sample_data)
        
        # Check required features
        assert "user_id" in features
        assert "item_id" in features
        assert "rating" in features
        assert "category" in features
        
        # Check tensor types
        assert isinstance(features["user_id"], tf.Tensor)
        assert isinstance(features["rating"], tf.Tensor)
        
        # Check shapes
        assert features["user_id"].shape[0] == 3
        assert features["rating"].shape[0] == 3


class TestModelBuilder:
    """Test ModelBuilder utility."""

    @pytest.fixture
    def builder(self):
        """Create ModelBuilder for testing."""
        config = create_baseline_config()
        return ModelBuilder(config)

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return pd.DataFrame({
            "user_id": [f"u{i}" for i in range(100)],
            "parent_asin": [f"i{i % 50}" for i in range(100)],
            "rating": np.random.uniform(1, 5, 100),
            "category": np.random.choice(["Books", "Electronics", "Sports"], 100),
            "user_review_count": np.random.randint(1, 100, 100),
            "item_review_count": np.random.randint(1, 200, 100),
        })

    def test_compute_data_statistics(self, builder, sample_train_data):
        """Test data statistics computation."""
        stats = builder.compute_data_statistics(sample_train_data)
        
        assert stats["n_users"] == 100
        assert stats["n_items"] == 50
        assert stats["n_interactions"] == 100
        assert stats["n_categories"] == 3
        assert "rating_mean" in stats
        assert "rating_std" in stats

    def test_build_model_from_config(self, sample_train_data):
        """Test complete model building from configuration."""
        config = create_baseline_config()
        
        model, data_stats, callbacks = build_model_from_config(
            config=config,
            train_df=sample_train_data,
        )
        
        # Check model
        assert isinstance(model, TwoTowerModel)
        
        # Check data statistics
        assert data_stats["n_users"] == 100
        assert data_stats["n_items"] == 50
        
        # Check callbacks
        assert len(callbacks) > 0
        assert any("ModelCheckpoint" in str(type(cb)) for cb in callbacks)
        assert any("EarlyStopping" in str(type(cb)) for cb in callbacks)