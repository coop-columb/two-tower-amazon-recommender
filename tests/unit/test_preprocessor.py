"""Unit tests for Amazon Reviews preprocessor."""

import logging
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.base import DatasetConfig
from src.data.preprocessor import (
    AmazonReviewsPreprocessor,
    FeatureEngineer,
    InteractionFilter,
    TextProcessor,
)


class TestTextProcessor:
    """Test TextProcessor class."""

    @pytest.fixture
    def text_processor(self):
        """Create a TextProcessor instance."""
        return TextProcessor()

    def test_clean_text_basic(self, text_processor):
        """Test basic text cleaning."""
        text = "This is a GREAT product!!! 😊"
        cleaned = text_processor.clean_text(text)

        assert cleaned == "This is a GREAT product"
        assert "!" not in cleaned
        assert "😊" not in cleaned

    def test_clean_text_html(self, text_processor):
        """Test HTML tag removal."""
        text = "<p>This is <b>bold</b> text</p>"
        cleaned = text_processor.clean_text(text)

        assert cleaned == "This is bold text"
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_clean_text_urls(self, text_processor):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        cleaned = text_processor.clean_text(text)

        assert "https://example.com" not in cleaned
        assert "Check out for more info" in cleaned

    def test_remove_stopwords(self, text_processor):
        """Test stopword removal."""
        text = "This is a test of the system"
        result = text_processor.remove_stopwords(text)

        # Common stopwords should be removed
        assert "is" not in result.lower()
        assert "a" not in result.lower()
        assert "the" not in result.lower()
        assert "test" in result.lower()
        assert "system" in result.lower()

    def test_stem_text(self, text_processor):
        """Test text stemming."""
        text = "running runs runner"
        stemmed = text_processor.stem_text(text)

        # All forms should be reduced to same stem
        words = stemmed.split()
        assert len(set(words)) == 1  # All should have same stem

    def test_process_pipeline(self, text_processor):
        """Test full text processing pipeline."""
        text = "<p>This product is absolutely AMAZING!!! Check https://example.com</p>"
        processed = text_processor.process(text)

        assert "<p>" not in processed
        assert "https://example.com" not in processed
        assert "amaz" in processed.lower()  # Stemmed form of amazing


class TestInteractionFilter:
    """Test InteractionFilter class."""

    @pytest.fixture
    def interaction_filter(self):
        """Create an InteractionFilter instance."""
        return InteractionFilter(min_user_interactions=2, min_item_interactions=2)

    @pytest.fixture
    def sample_data(self):
        """Create sample interaction data."""
        return pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
                "parent_asin": ["i1", "i2", "i3", "i1", "i2", "i1"],
                "rating": [5, 4, 3, 5, 3, 1],
            }
        )

    def test_filter_by_frequency(self, interaction_filter, sample_data):
        """Test filtering by interaction frequency."""
        filtered = interaction_filter.filter_by_frequency(sample_data)

        # u3 has only 1 interaction, should be filtered out
        assert "u3" not in filtered["user_id"].values

        # i3 has only 1 interaction, should be filtered out
        assert "i3" not in filtered["parent_asin"].values

        # Remaining data should have adequate interactions
        assert len(filtered) == 4

    def test_filter_by_rating_default(self, interaction_filter, sample_data):
        """Test filtering by rating with default (no filtering)."""
        filtered = interaction_filter.filter_by_rating(sample_data)

        assert len(filtered) == len(sample_data)

    def test_filter_by_rating_threshold(self, sample_data):
        """Test filtering by rating with threshold."""
        filter_with_rating = InteractionFilter(
            min_user_interactions=1,
            min_item_interactions=1,
            min_rating=3,
        )

        filtered = filter_with_rating.filter_by_rating(sample_data)

        # Rating 1 should be filtered out
        assert 1 not in filtered["rating"].values
        assert len(filtered) == 5

    def test_filter_pipeline(self, interaction_filter, sample_data):
        """Test full filtering pipeline."""
        filtered = interaction_filter.filter(sample_data)

        # Should apply both frequency and rating filters
        assert len(filtered) == 4
        assert "u3" not in filtered["user_id"].values
        assert "i3" not in filtered["parent_asin"].values


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def feature_engineer(self):
        """Create a FeatureEngineer instance."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with required columns."""
        return pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "parent_asin": ["i1", "i2", "i1"],
                "rating": [5, 4, 3],
                "text": ["Great product", "Good item", "Average"],
                "timestamp": [1609459200, 1609545600, 1609632000],
            }
        )

    def test_create_temporal_features(self, feature_engineer, sample_data):
        """Test temporal feature creation."""
        result = feature_engineer.create_temporal_features(sample_data)

        # Check new columns exist
        assert "hour" in result.columns
        assert "day" in result.columns
        assert "month" in result.columns
        assert "year" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns

        # Verify values are reasonable
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_create_text_features(self, feature_engineer, sample_data):
        """Test text feature creation."""
        result = feature_engineer.create_text_features(sample_data)

        # Check new columns exist
        assert "text_length" in result.columns
        assert "word_count" in result.columns
        assert "exclamation_count" in result.columns
        assert "question_count" in result.columns
        assert "caps_ratio" in result.columns

        # Verify values
        assert result["text_length"].iloc[0] == len("Great product")
        assert result["word_count"].iloc[0] == 2

    def test_create_user_features(self, feature_engineer, sample_data):
        """Test user feature creation."""
        result = feature_engineer.create_user_features(sample_data)

        # Check new columns exist
        assert "user_review_count" in result.columns
        assert "user_avg_rating" in result.columns
        assert "user_rating_std" in result.columns

        # Verify values for u1 (2 reviews with ratings 5 and 4)
        u1_data = result[result["user_id"] == "u1"].iloc[0]
        assert u1_data["user_review_count"] == 2
        assert u1_data["user_avg_rating"] == 4.5

    def test_create_item_features(self, feature_engineer, sample_data):
        """Test item feature creation."""
        result = feature_engineer.create_item_features(sample_data)

        # Check new columns exist
        assert "item_review_count" in result.columns
        assert "item_avg_rating" in result.columns
        assert "item_rating_std" in result.columns

        # Verify values for i1 (2 reviews with ratings 5 and 3)
        i1_data = result[result["parent_asin"] == "i1"].iloc[0]
        assert i1_data["item_review_count"] == 2
        assert i1_data["item_avg_rating"] == 4.0

    def test_engineer_features_pipeline(self, feature_engineer, sample_data):
        """Test full feature engineering pipeline."""
        result = feature_engineer.engineer_features(sample_data)

        # Check all feature groups are created
        temporal_cols = ["hour", "day", "month", "year", "day_of_week", "is_weekend"]
        text_cols = [
            "text_length",
            "word_count",
            "exclamation_count",
            "question_count",
            "caps_ratio",
        ]
        user_cols = ["user_review_count", "user_avg_rating", "user_rating_std"]
        item_cols = ["item_review_count", "item_avg_rating", "item_rating_std"]

        all_expected_cols = temporal_cols + text_cols + user_cols + item_cols
        for col in all_expected_cols:
            assert col in result.columns


class TestAmazonReviewsPreprocessor:
    """Test AmazonReviewsPreprocessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DatasetConfig."""
        return DatasetConfig(
            name="amazon_reviews",
            source="test",
            categories=["All_Beauty"],
            preprocessing={
                "min_interactions_per_user": 2,
                "min_interactions_per_item": 2,
                "min_user_interactions": 2,
                "min_item_interactions": 2,
                "test_size": 0.2,
                "val_size": 0.1,
            },
            model={},
        )

    @pytest.fixture
    def preprocessor(self, mock_config):
        """Create a preprocessor instance."""
        return AmazonReviewsPreprocessor(mock_config)

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "user_id": [f"u{i % 10}" for i in range(n_samples)],
                "parent_asin": [f"i{i % 15}" for i in range(n_samples)],
                "rating": np.random.randint(1, 6, n_samples),
                "text": [
                    "This is a test review " * np.random.randint(1, 5) for _ in range(n_samples)
                ],
                "timestamp": np.random.randint(1609459200, 1640995200, n_samples),
            }
        )

    def test_basic_cleaning(self, preprocessor, sample_data):
        """Test basic data cleaning."""
        # Add duplicates and NaN values
        sample_data = pd.concat([sample_data, sample_data.iloc[:5]])  # Add duplicates
        sample_data.loc[10, "text"] = None  # Add NaN

        cleaned = preprocessor.basic_cleaning(sample_data)

        # Check duplicates removed
        assert len(cleaned) < len(sample_data)

        # Check NaN removed
        assert cleaned["text"].isna().sum() == 0

    def test_encode_categorical_ids(self, preprocessor, sample_data):
        """Test categorical ID encoding."""
        encoded = preprocessor.encode_categorical_ids(sample_data)

        # Check new columns exist
        assert "user_idx" in encoded.columns
        assert "item_idx" in encoded.columns

        # Check encoding starts from 0
        assert encoded["user_idx"].min() == 0
        assert encoded["item_idx"].min() == 0

        # Check mappings are stored
        assert hasattr(preprocessor, "user_encoder")
        assert hasattr(preprocessor, "item_encoder")

    def test_split_data_temporal(self, preprocessor, sample_data):
        """Test temporal data splitting."""
        train, val, test = preprocessor.split_data(sample_data, method="temporal")

        # Check split sizes
        assert len(train) + len(val) + len(test) == len(sample_data)

        # Check temporal ordering
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_split_data_random(self, preprocessor, sample_data):
        """Test random data splitting."""
        train, val, test = preprocessor.split_data(sample_data, method="random")

        # Check split sizes
        total = len(sample_data)
        assert len(test) == pytest.approx(total * 0.2, abs=2)
        assert len(val) == pytest.approx(total * 0.08, abs=2)  # 0.1 of remaining 0.8

    def test_process_pipeline(self, preprocessor, sample_data):
        """Test full preprocessing pipeline."""
        processed = preprocessor.process(sample_data)

        # Check all expected columns exist
        expected_columns = [
            "user_id",
            "parent_asin",
            "rating",
            "text",
            "timestamp",
            "user_idx",
            "item_idx",
            "hour",
            "day",
            "month",
            "year",
            "day_of_week",
            "is_weekend",
            "text_length",
            "word_count",
            "exclamation_count",
            "question_count",
            "caps_ratio",
            "user_review_count",
            "user_avg_rating",
            "user_rating_std",
            "item_review_count",
            "item_avg_rating",
            "item_rating_std",
        ]

        for col in expected_columns:
            assert col in processed.columns

        # Check data is filtered
        assert len(processed) <= len(sample_data)
