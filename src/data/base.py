"""
Abstract base classes for data processing pipeline components.

This module defines the foundational interfaces for all data processing
operations, ensuring consistent implementation patterns across the pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration container for dataset parameters."""

    name: str
    source: str
    categories: list[str]
    preprocessing: dict[str, Any]
    model: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        required_preprocessing_keys = ["min_interactions_per_user", "min_interactions_per_item"]
        for key in required_preprocessing_keys:
            if key not in self.preprocessing:
                raise ValueError(f"Missing required preprocessing parameter: {key}")


class DataProcessor(ABC):
    """Abstract base class for all data processing components."""

    def __init__(self, config: DatasetConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process input data and return transformed data.

        Args:
            data: Input DataFrame to process

        Returns:
            Transformed DataFrame
        """
        pass

    def validate_input(self, data: pd.DataFrame, required_columns: list[str]) -> None:
        """
        Validate that input data contains required columns.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def log_processing_stats(self, input_data: pd.DataFrame, output_data: pd.DataFrame) -> None:
        """Log processing statistics."""
        self.logger.info(
            f"Processing complete: {len(input_data)} -> {len(output_data)} rows "
            f"({len(output_data)/len(input_data)*100:.1f}% retention)"
        )


class DataValidator(ABC):
    """Abstract base class for data validation components."""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate data quality and return validation results.

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass


class DataLoader(ABC):
    """Abstract base class for data loading components."""

    @abstractmethod
    def load(self, source: str | Path) -> pd.DataFrame:
        """
        Load data from specified source.

        Args:
            source: Data source path or identifier

        Returns:
            Loaded DataFrame
        """
        pass


class DataSaver(ABC):
    """Abstract base class for data saving components."""

    @abstractmethod
    def save(self, data: pd.DataFrame, destination: str | Path) -> None:
        """
        Save data to specified destination.

        Args:
            data: DataFrame to save
            destination: Save destination path
        """
        pass
