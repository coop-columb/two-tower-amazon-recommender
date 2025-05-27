"""
Advanced data preprocessing pipeline for Amazon Reviews dataset.

This module implements sophisticated preprocessing including text cleaning,
feature engineering, interaction filtering, and train/validation/test splitting
with proper temporal considerations.
"""

import html
import logging
import re
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .base import DataProcessor, DatasetConfig


class TextProcessor:
    """Advanced text processing utilities for review data."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2000,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_special_chars: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        stem_words: bool = False,
    ):
        """Initialize text processor with configuration options.
        
        Args:
            min_length: Minimum allowed text length
            max_length: Maximum allowed text length
            remove_html: Whether to remove HTML tags and entities
            remove_urls: Whether to remove URLs from text
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert text to lowercase
            remove_stopwords: Whether to remove common stopwords
            stem_words: Whether to apply word stemming
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words

        # Initialize NLTK components if needed
        if self.remove_stopwords or self.stem_words:
            self._setup_nltk()

    def _setup_nltk(self) -> None:
        """Download required NLTK data."""
        # Initialize attributes that might be used later
        self.stop_words = set()
        self.stemmer = None

        try:
            # Always try to get the punkt tokenizer
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            # Get stopwords if needed
            if self.remove_stopwords:
                try:
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                self.stop_words = set(stopwords.words("english"))

            # Initialize stemmer if needed
            if self.stem_words:
                self.stemmer = PorterStemmer()
        except Exception as e:
            # Log the error but continue without failing
            logging.warning(
                f"NLTK setup error: {str(e)}. Some text processing features may be unavailable."
            )

    def clean_text(self, text: str | None | Any) -> str:
        """
        Clean and normalize text data.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove HTML entities and tags
        if self.remove_html:
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        if self.remove_urls:
            # URL pattern broken into parts for readability
            url_pattern = (
                r"http[s]?://"  # Protocol part
                r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|"  # Valid characters
                r"(?:%[0-9a-fA-F][0-9a-fA-F])+)"  # Hex-encoded characters
            )
            text = re.sub(url_pattern, "", text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove special characters but keep basic punctuation
        if self.remove_special_chars:
            text = re.sub(r"[^\w\s\.\!\?\,\;\:]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Advanced processing if enabled
        if self.remove_stopwords or self.stem_words:
            tokens = word_tokenize(text)

            if self.remove_stopwords:
                tokens = [token for token in tokens if token.lower() not in self.stop_words]

            if self.stem_words and self.stemmer is not None:
                tokens = [self.stemmer.stem(token) for token in tokens]

            text = " ".join(tokens)

        return text

    def validate_text_length(self, text: str) -> bool:
        """Check if text meets length requirements."""
        return self.min_length <= len(text) <= self.max_length


class InteractionFilter:
    """Filter user-item interactions based on frequency and quality."""

    def __init__(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        """Initialize interaction filter with filtering criteria.
        
        Args:
            min_user_interactions: Minimum interactions required per user
            min_item_interactions: Minimum interactions required per item
            min_rating: Minimum rating value to include
            max_rating: Maximum rating value to include
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.min_rating = min_rating
        self.max_rating = max_rating

    def filter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter interactions based on frequency and quality criteria.

        Args:
            df: DataFrame with user-item interactions

        Returns:
            Filtered DataFrame
        """
        logger = logging.getLogger(self.__class__.__name__)
        initial_size = len(df)

        # Filter by rating range
        df = df[(df["rating"] >= self.min_rating) & (df["rating"] <= self.max_rating)]
        logger.info(f"After rating filter: {len(df)} rows ({len(df)/initial_size*100:.1f}%)")

        # Iterative filtering for user and item frequency
        prev_size = 0
        iteration = 0
        max_iterations = 10

        while len(df) != prev_size and iteration < max_iterations:
            prev_size = len(df)
            iteration += 1

            # Filter users with insufficient interactions
            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            df = df[df["user_id"].isin(valid_users)]

            # Filter items with insufficient interactions
            item_counts = df["parent_asin"].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            df = df[df["parent_asin"].isin(valid_items)]

            logger.debug(f"Iteration {iteration}: {len(df)} rows remaining")

        logger.info(
            f"After interaction filtering: {len(df)} rows "
            f"({len(df)/initial_size*100:.1f}% retention)"
        )

        return df.reset_index(drop=True)


class FeatureEngineer:
    """Feature engineering for user-item interactions."""

    def __init__(self, text_processor: TextProcessor | None = None):
        """Initialize feature engineer with optional text processor.
        
        Args:
            text_processor: TextProcessor instance for text feature creation
        """
        self.text_processor = text_processor or TextProcessor()

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp data."""
        if "timestamp" not in df.columns:
            return df

        df = df.copy()

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        # Extract temporal features
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["hour"] = df["datetime"].dt.hour
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Create relative time features
        min_date = df["datetime"].min()
        df["days_since_start"] = (df["datetime"] - min_date).dt.days

        return df

    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text data."""
        df = df.copy()

        # Process review text
        if "text" in df.columns:
            df["text_clean"] = df["text"].apply(self.text_processor.clean_text)
            df["text_length"] = df["text_clean"].str.len()
            df["word_count"] = df["text_clean"].str.split().str.len()
            df["exclamation_count"] = df["text"].str.count("!")
            df["question_count"] = df["text"].str.count(r"\?")
            df["caps_ratio"] = df["text"].str.count(r"[A-Z]") / df["text"].str.len().replace(0, 1)

        # Process review title
        if "title" in df.columns:
            df["title_clean"] = df["title"].apply(self.text_processor.clean_text)
            df["title_length"] = df["title_clean"].str.len()
            df["title_word_count"] = df["title_clean"].str.split().str.len()

        return df

    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-level aggregated features."""
        df = df.copy()

        # User rating statistics
        user_stats = (
            df.groupby("user_id")
            .agg(
                {
                    "rating": ["count", "mean", "std", "min", "max"],
                    "text_length": ["mean", "std"],
                    "word_count": ["mean", "std"],
                }
            )
            .round(3)
        )

        # Flatten column names
        user_stats.columns = pd.Index(
            [
                f"user_{col[0]}_{col[1]}" if col[1] else f"user_{col[0]}"
                for col in user_stats.columns
            ]
        )
        user_stats = user_stats.reset_index()

        # Merge back to main dataframe
        df = df.merge(user_stats, on="user_id", how="left")

        return df

    def create_item_features(
        self, df: pd.DataFrame, meta_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Create item-level aggregated features."""
        df = df.copy()

        # Item rating statistics
        item_stats = (
            df.groupby("parent_asin")
            .agg(
                {
                    "rating": ["count", "mean", "std", "min", "max"],
                    "text_length": ["mean", "std"],
                    "word_count": ["mean", "std"],
                }
            )
            .round(3)
        )

        # Flatten column names
        item_stats.columns = pd.Index(
            [
                f"item_{col[0]}_{col[1]}" if col[1] else f"item_{col[0]}"
                for col in item_stats.columns
            ]
        )
        item_stats = item_stats.reset_index()

        # Merge back to main dataframe
        df = df.merge(item_stats, on="parent_asin", how="left")

        # Add metadata features if available
        if meta_df is not None:
            meta_columns = ["parent_asin", "main_category", "average_rating", "rating_number"]
            meta_features = meta_df[meta_columns].copy()
            df = df.merge(meta_features, on="parent_asin", how="left")

        return df


class AmazonReviewsPreprocessor(DataProcessor):
    """
    Comprehensive preprocessor for Amazon Reviews dataset.

    Handles text cleaning, feature engineering, interaction filtering,
    and dataset splitting with temporal considerations.
    """

    def __init__(self, config: DatasetConfig, logger: logging.Logger | None = None):
        """Initialize Amazon Reviews preprocessor with configuration.
        
        Args:
            config: Dataset configuration object
            logger: Optional logger instance
        """
        super().__init__(config, logger)

        # Initialize components
        self.text_processor = TextProcessor(
            min_length=config.preprocessing.get("min_text_length", 10),
            max_length=config.preprocessing.get("max_text_length", 2000),
            lowercase=True,
            remove_html=True,
            remove_urls=True,
        )

        self.interaction_filter = InteractionFilter(
            min_user_interactions=config.preprocessing["min_interactions_per_user"],
            min_item_interactions=config.preprocessing["min_interactions_per_item"],
            min_rating=config.preprocessing.get("min_rating", 1.0),
            max_rating=config.preprocessing.get("max_rating", 5.0),
        )

        self.feature_engineer = FeatureEngineer(self.text_processor)

        # Label encoders for categorical variables
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def process(self, data: pd.DataFrame, meta_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Process Amazon Reviews data through complete pipeline.

        Args:
            data: Reviews DataFrame
            meta_data: Optional metadata DataFrame

        Returns:
            Processed DataFrame ready for model training
        """
        self.logger.info("Starting Amazon Reviews preprocessing pipeline")

        # Validate input data
        required_columns = ["user_id", "parent_asin", "rating", "text", "timestamp"]
        self.validate_input(data, required_columns)

        df = data.copy()

        # Step 1: Basic data cleaning
        self.logger.info("Step 1: Basic data cleaning")
        df = self._basic_cleaning(df)

        # Step 2: Text processing
        self.logger.info("Step 2: Text processing")
        df = self._process_text_data(df)

        # Step 3: Filter interactions
        self.logger.info("Step 3: Filtering interactions")
        df = self.interaction_filter.filter_interactions(df)

        # Step 4: Feature engineering
        self.logger.info("Step 4: Feature engineering")
        df = self._engineer_features(df, meta_data)

        # Step 5: Encode categorical variables
        self.logger.info("Step 5: Encoding categorical variables")
        df = self._encode_categories(df)

        # Log final statistics
        self.log_processing_stats(data, df)
        self._log_dataset_statistics(df)

        return df

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning operations."""
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(["user_id", "parent_asin"])

        if len(df) < initial_size:
            self.logger.info(f"Removed {initial_size - len(df)} duplicate interactions")

        # Handle missing values
        df = df.dropna(subset=["user_id", "parent_asin", "rating"])
        df["text"] = df["text"].fillna("")
        df["title"] = df["title"].fillna("")

        return df

    def _process_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate text data."""
        # Clean text fields
        if "text" in df.columns:
            df["text_clean"] = df["text"].apply(self.text_processor.clean_text)

            # Filter by text length
            valid_text = df["text_clean"].apply(self.text_processor.validate_text_length)
            df = df[valid_text]

            self.logger.info(f"Filtered {(~valid_text).sum()} reviews with invalid text length")

        if "title" in df.columns:
            df["title_clean"] = df["title"].apply(self.text_processor.clean_text)

        return df

    def _engineer_features(self, df: pd.DataFrame, meta_data: pd.DataFrame | None) -> pd.DataFrame:
        """Apply feature engineering pipeline."""
        # Create temporal features
        df = self.feature_engineer.create_temporal_features(df)

        # Create text features
        df = self.feature_engineer.create_text_features(df)

        # Create user and item features
        df = self.feature_engineer.create_user_features(df)
        df = self.feature_engineer.create_item_features(df, meta_data)

        return df

    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables to numerical format."""
        # Encode user and item IDs
        df["user_id_encoded"] = self.user_encoder.fit_transform(df["user_id"])
        df["item_id_encoded"] = self.item_encoder.fit_transform(df["parent_asin"])

        # Encode categorical features if present
        if "main_category" in df.columns:
            category_encoder = LabelEncoder()
            df["category_encoded"] = category_encoder.fit_transform(
                df["main_category"].fillna("Unknown")
            )

        return df

    def _log_dataset_statistics(self, df: pd.DataFrame) -> None:
        """Log comprehensive dataset statistics."""
        stats = {
            "total_interactions": len(df),
            "unique_users": df["user_id"].nunique(),
            "unique_items": df["parent_asin"].nunique(),
            "sparsity": 1 - (len(df) / (df["user_id"].nunique() * df["parent_asin"].nunique())),
            "rating_distribution": df["rating"].value_counts().to_dict(),
            "avg_rating": df["rating"].mean(),
            "avg_text_length": df["text_length"].mean() if "text_length" in df.columns else None,
            "date_range": (
                (df["datetime"].min(), df["datetime"].max()) if "datetime" in df.columns else None
            ),
        }

        self.logger.info(f"Dataset statistics: {stats}")

    def split_temporal(
        self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset temporally for realistic evaluation.

        Args:
            df: Processed DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if "datetime" not in df.columns:
            raise ValueError("Temporal splitting requires datetime column")

        # Sort by timestamp
        df_sorted = df.sort_values("datetime").reset_index(drop=True)

        # Calculate split indices
        n_total = len(df_sorted)
        train_idx = int(n_total * train_ratio)
        val_idx = int(n_total * (train_ratio + val_ratio))

        train_df = df_sorted[:train_idx].copy()
        val_df = df_sorted[train_idx:val_idx].copy()
        test_df = df_sorted[val_idx:].copy()

        self.logger.info(
            f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def split_random(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset randomly for development purposes.

        Args:
            df: Processed DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        test_ratio = 1 - train_ratio - val_ratio

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=df["rating"] if len(df["rating"].unique()) > 1 else None,
        )

        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state,
            stratify=temp_df["rating"] if len(temp_df["rating"].unique()) > 1 else None,
        )

        self.logger.info(
            f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        return train_df, val_df, test_df
