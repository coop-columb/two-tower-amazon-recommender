"""
Amazon Reviews 2023 dataset loader implementation.

This module handles downloading, caching, and loading of Amazon Reviews data
from HuggingFace Hub with robust error handling and data validation.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from .base import DataLoader, DatasetConfig, DataValidator


# TypedDict for Amazon dataset statistics
class CategoryStats(TypedDict, total=False):
    total_records: int
    unique_users: int
    unique_items: int
    avg_rating: float | None
    rating_distribution: dict[int, int]
    avg_text_length: float | None
    has_error: int
    error_message: dict[str, str]


class AmazonReviewsValidator(DataValidator):
    """Validator for Amazon Reviews dataset structure and quality."""

    REQUIRED_REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "title", "text", "timestamp"]

    REQUIRED_META_COLUMNS = [
        "parent_asin",
        "main_category",
        "title",
        "average_rating",
        "rating_number",
    ]

    def validate(self, data: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate Amazon Reviews data structure and quality.

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required columns
        if "rating" in data.columns:
            # Reviews data validation
            missing_cols = set(self.REQUIRED_REVIEW_COLUMNS) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing review columns: {missing_cols}")

            # Rating range validation
            if "rating" in data.columns:
                invalid_ratings = data[(data["rating"] < 1) | (data["rating"] > 5)]
                if len(invalid_ratings) > 0:
                    issues.append(
                        f"Found {len(invalid_ratings)} invalid ratings outside 1-5 range"
                    )

            # Text quality validation
            if "text" in data.columns:
                empty_text = data[data["text"].isnull() | (data["text"].str.strip() == "")]
                if len(empty_text) > len(data) * 0.1:  # More than 10% empty
                    issues.append(
                        f"High percentage of empty reviews: {len(empty_text)/len(data)*100:.1f}%"
                    )

        else:
            # Metadata validation
            missing_cols = set(self.REQUIRED_META_COLUMNS) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing metadata columns: {missing_cols}")

        # Duplicate validation
        if "user_id" in data.columns and "parent_asin" in data.columns:
            duplicates = data.duplicated(["user_id", "parent_asin"])
            if duplicates.sum() > 0:
                issues.append(f"Found {duplicates.sum()} duplicate user-item interactions")

        return len(issues) == 0, issues


class AmazonReviewsLoader(DataLoader):
    """
    Loader for Amazon Reviews 2023 dataset with caching and validation.

    Handles downloading from HuggingFace Hub, local caching for efficiency,
    and comprehensive data validation.
    """

    def __init__(
        self,
        config: DatasetConfig,
        cache_dir: Path | None = None,
        trust_remote_code: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.cache_dir = cache_dir or Path("data/cache")
        self.trust_remote_code = trust_remote_code
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.validator = AmazonReviewsValidator()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, category: str, data_type: str) -> Path:
        """Generate cache file path for category and data type."""
        cache_filename = f"{category}_{data_type}.parquet"
        return self.cache_dir / cache_filename

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached file exists and is not too old."""
        if not cache_path.exists():
            return False

        file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        return file_age_hours < max_age_hours

    def _download_category_data(self, category: str, data_type: str = "reviews") -> pd.DataFrame:
        """
        Download data for specific category from HuggingFace Hub.

        Args:
            category: Product category (e.g., 'All_Beauty')
            data_type: Type of data ('reviews' or 'meta')

        Returns:
            DataFrame with downloaded data
        """
        self.logger.info(f"Downloading {data_type} data for category: {category}")

        try:
            dataset_name = f"raw_{data_type}_{category}"

            dataset = load_dataset(
                self.config.source, dataset_name, trust_remote_code=self.trust_remote_code
            )

            # Convert to pandas DataFrame
            df: pd.DataFrame = dataset["full"].to_pandas()

            # Validate downloaded data
            is_valid, issues = self.validator.validate(df)
            if not is_valid:
                self.logger.warning(f"Data validation issues for {category}: {issues}")

            self.logger.info(
                f"Successfully downloaded {len(df)} records for {category} {data_type}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Failed to download {category} {data_type}: {str(e)}")
            raise

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save DataFrame to cache with compression."""
        try:
            data.to_parquet(cache_path, compression="snappy", index=False)
            self.logger.debug(f"Cached data to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {str(e)}")

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame:
        """Load DataFrame from cache."""
        try:
            df = pd.read_parquet(cache_path)
            self.logger.debug(f"Loaded data from cache: {cache_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load from cache {cache_path}: {str(e)}")
            raise

    def load_category(
        self,
        category: str,
        data_type: str = "reviews",
        use_cache: bool = True,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Load data for a specific category.

        Args:
            category: Product category to load
            data_type: Type of data ('reviews' or 'meta')
            use_cache: Whether to use cached data if available
            sample_size: Optional sample size for development/testing

        Returns:
            DataFrame with category data
        """
        cache_path = self._get_cache_path(category, data_type)

        # Check cache first
        if use_cache and self._is_cache_valid(cache_path):
            df = self._load_from_cache(cache_path)
        else:
            # Download fresh data
            df = self._download_category_data(category, data_type)

            # Cache the downloaded data
            if use_cache:
                self._save_to_cache(df, cache_path)

        # Apply sampling if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            self.logger.info(f"Sampled {sample_size} records from {len(df)} total")

        return df

    def load_multiple_categories(
        self,
        categories: list[str] | None = None,
        data_type: str = "reviews",
        use_cache: bool = True,
        sample_size_per_category: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load data for multiple categories.

        Args:
            categories: List of categories to load (uses config default if None)
            data_type: Type of data ('reviews' or 'meta')
            use_cache: Whether to use cached data
            sample_size_per_category: Sample size per category

        Returns:
            Dictionary mapping category names to DataFrames
        """
        if categories is None:
            categories = self.config.categories

        data_dict = {}
        failed_categories = []

        for category in categories:
            try:
                df = self.load_category(
                    category=category,
                    data_type=data_type,
                    use_cache=use_cache,
                    sample_size=sample_size_per_category,
                )
                data_dict[category] = df

            except Exception as e:
                self.logger.error(f"Failed to load {category}: {str(e)}")
                failed_categories.append(category)
                continue

        if failed_categories:
            self.logger.warning(f"Failed to load categories: {failed_categories}")

        self.logger.info(
            f"Successfully loaded {len(data_dict)} out of {len(categories)} categories"
        )

        return data_dict

    def load(self, source: str | Path) -> pd.DataFrame:
        """
        Load data from source (implements abstract method).

        For Amazon Reviews loader, source should be a category name.

        Args:
            source: Category name to load

        Returns:
            DataFrame with loaded data
        """
        if isinstance(source, Path):
            source = str(source)

        return self.load_category(category=source)

    def get_available_categories(self) -> list[str]:
        """
        Get list of available categories from HuggingFace Hub.

        Returns:
            List of available category names
        """
        try:
            api = HfApi()
            dataset_info = api.dataset_info(self.config.source)

            # Extract category names from dataset configs
            available_configs = getattr(dataset_info, "config_names", []) or []

            # Filter for review configs and extract category names
            review_configs = [
                config.replace("raw_review_", "")
                for config in available_configs
                if config.startswith("raw_review_")
            ]

            return sorted(review_configs)

        except Exception as e:
            self.logger.error(f"Failed to get available categories: {str(e)}")
            return self.config.categories  # Fallback to config categories

    def get_dataset_statistics(
        self, categories: list[str] | None = None
    ) -> dict[str, CategoryStats]:
        """
        Get basic statistics for dataset categories.

        Args:
            categories: Categories to analyze (uses config default if None)

        Returns:
            Dictionary with statistics for each category
        """
        if categories is None:
            categories = self.config.categories[:5]  # Limit for efficiency

        stats = {}

        for category in categories:
            try:
                df = self.load_category(category, sample_size=10000)  # Sample for stats

                category_stats = {
                    "total_records": len(df),
                    "unique_users": (df["user_id"].nunique() if "user_id" in df.columns else 0),
                    "unique_items": (
                        df["parent_asin"].nunique() if "parent_asin" in df.columns else 0
                    ),
                    "avg_rating": df["rating"].mean() if "rating" in df.columns else None,
                    "rating_distribution": (
                        df["rating"].value_counts().to_dict() if "rating" in df.columns else {}
                    ),
                    "avg_text_length": (
                        df["text"].str.len().mean() if "text" in df.columns else None
                    ),
                }

                stats[category] = cast(CategoryStats, category_stats)

            except Exception as e:
                self.logger.error(f"Failed to get stats for {category}: {str(e)}")
                # Create a stats entry for error case
                stats[category] = cast(
                    CategoryStats,
                    {
                        "total_records": 0,
                        "unique_users": 0,
                        "unique_items": 0,
                        "avg_rating": None,
                        "rating_distribution": {},
                        "avg_text_length": None,
                        "has_error": 1,  # Using 1 instead of True for type consistency
                        "error_message": {
                            "text": str(e)
                        },  # Convert to dict to match expected types
                    },
                )

        return stats
