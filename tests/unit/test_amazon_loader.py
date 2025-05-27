"""Unit tests for Amazon Reviews data loader."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.data.amazon_loader import AmazonReviewsLoader, AmazonReviewsValidator
from src.data.base import DatasetConfig


class TestAmazonReviewsLoader:
    """Test AmazonReviewsLoader class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DatasetConfig."""
        return DatasetConfig(
            name="amazon_reviews",
            source="McAuley-Lab/Amazon-Reviews-2023",
            categories=["All_Beauty"],
            preprocessing={
                "min_interactions_per_user": 5,
                "min_interactions_per_item": 5,
            },
            model={},
        )

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        mock_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3"],
                "parent_asin": ["item1", "item2", "item3"],
                "rating": [5, 4, 3],
                "text": ["Great product", "Good item", "Average"],
                "timestamp": [1609459200, 1609545600, 1609632000],
            }
        )

        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value.to_pandas.return_value = mock_data
        return mock_dataset

    def test_loader_initialization(self, mock_config):
        """Test AmazonReviewsLoader initialization."""
        loader = AmazonReviewsLoader(mock_config)

        assert loader.config == mock_config
        assert loader.cache_dir == Path.home() / ".cache" / "amazon_reviews"
        assert loader.trust_remote_code is False
        assert loader.logger is not None

    def test_loader_with_custom_cache_dir(self, mock_config, tmp_path):
        """Test AmazonReviewsLoader with custom cache directory."""
        custom_cache = tmp_path / "custom_cache"
        loader = AmazonReviewsLoader(mock_config, cache_dir=custom_cache)

        assert loader.cache_dir == custom_cache
        assert custom_cache.exists()

    @patch("src.data.amazon_loader.load_dataset")
    def test_load_single_category(self, mock_load_dataset, mock_config, mock_dataset):
        """Test loading a single category."""
        mock_load_dataset.return_value = mock_dataset

        loader = AmazonReviewsLoader(mock_config)
        result = loader.load(Path("dummy_path"))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "user_id" in result.columns
        assert "rating" in result.columns

        # Verify load_dataset was called correctly
        mock_load_dataset.assert_called_once_with(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_All_Beauty",
            cache_dir=loader.cache_dir,
            trust_remote_code=False,
        )

    @patch("src.data.amazon_loader.load_dataset")
    def test_load_multiple_categories(self, mock_load_dataset, mock_config, mock_dataset):
        """Test loading multiple categories."""
        mock_config.categories = ["All_Beauty", "Electronics"]
        mock_load_dataset.return_value = mock_dataset

        loader = AmazonReviewsLoader(mock_config)
        result = loader.load(Path("dummy_path"))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 rows * 2 categories
        assert mock_load_dataset.call_count == 2

    @patch("src.data.amazon_loader.load_dataset")
    def test_load_with_exception(self, mock_load_dataset, mock_config):
        """Test load handles exceptions gracefully."""
        mock_load_dataset.side_effect = Exception("Download failed")

        loader = AmazonReviewsLoader(mock_config)

        with pytest.raises(Exception, match="Download failed"):
            loader.load(Path("dummy_path"))

    @patch("src.data.amazon_loader.load_dataset")
    def test_get_dataset_stats(self, mock_load_dataset, mock_config):
        """Test getting dataset statistics."""
        # Create mock dataset info
        mock_info = MagicMock()
        mock_info.dataset_size = 1024 * 1024 * 100  # 100 MB
        mock_info.download_size = 1024 * 1024 * 50  # 50 MB
        mock_info.features = {"user_id": "string", "rating": "int32"}

        mock_dataset = MagicMock()
        mock_dataset.info = mock_info
        mock_load_dataset.return_value = mock_dataset

        loader = AmazonReviewsLoader(mock_config)
        stats = loader.get_dataset_stats("All_Beauty")

        assert stats["category"] == "All_Beauty"
        assert stats["download_size_mb"] == 50.0
        assert stats["dataset_size_mb"] == 100.0
        assert stats["features"] == {"user_id": "string", "rating": "int32"}


class TestAmazonReviewsValidator:
    """Test AmazonReviewsValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return AmazonReviewsValidator()

    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        return pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3"],
                "parent_asin": ["item1", "item2", "item3"],
                "rating": [5, 4, 3],
                "text": ["Great product!", "Good item.", "Average product."],
                "timestamp": [1609459200, 1609545600, 1609632000],
            }
        )

    def test_validate_valid_data(self, validator, valid_data):
        """Test validation with valid data."""
        is_valid, errors = validator.validate(valid_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_columns(self, validator):
        """Test validation with missing required columns."""
        data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "rating": [5, 4],
            }
        )

        is_valid, errors = validator.validate(data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("Missing required columns" in error for error in errors)

    def test_validate_invalid_ratings(self, validator, valid_data):
        """Test validation with invalid ratings."""
        valid_data.loc[0, "rating"] = 6  # Invalid rating
        valid_data.loc[1, "rating"] = 0  # Invalid rating

        is_valid, errors = validator.validate(valid_data)

        assert is_valid is False
        assert any("Found 2 invalid ratings" in error for error in errors)

    def test_validate_short_text(self, validator, valid_data):
        """Test validation with short text reviews."""
        valid_data.loc[0, "text"] = "Bad"  # Too short
        valid_data.loc[1, "text"] = ""  # Empty

        is_valid, errors = validator.validate(valid_data)

        assert is_valid is False
        assert any(
            "Found 2 reviews with text shorter than 5 characters" in error for error in errors
        )

    def test_validate_duplicates(self, validator, valid_data):
        """Test validation with duplicate reviews."""
        # Add duplicate row
        duplicate_row = valid_data.iloc[0].copy()
        valid_data = pd.concat([valid_data, duplicate_row.to_frame().T], ignore_index=True)

        is_valid, errors = validator.validate(valid_data)

        assert is_valid is False
        assert any("Found 1 duplicate reviews" in error for error in errors)

    def test_validate_empty_dataframe(self, validator):
        """Test validation with empty dataframe."""
        empty_df = pd.DataFrame()

        is_valid, errors = validator.validate(empty_df)

        assert is_valid is False
        assert len(errors) > 0
