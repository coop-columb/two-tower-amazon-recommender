"""Unit tests for base data pipeline components."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data.base import DataLoader, DataProcessor, DataSaver, DatasetConfig, DataValidator


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_dataset_config_creation(self):
        """Test creating a DatasetConfig instance."""
        config = DatasetConfig(
            name="test_dataset",
            source="test_source",
            categories=["cat1", "cat2"],
            preprocessing={
                "min_interactions_per_user": 5,
                "min_interactions_per_item": 5,
                "param1": "value1",
            },
            model={"param2": "value2"},
        )

        assert config.name == "test_dataset"
        assert config.source == "test_source"
        assert config.categories == ["cat1", "cat2"]
        assert config.preprocessing["min_interactions_per_user"] == 5
        assert config.preprocessing["min_interactions_per_item"] == 5
        assert config.model == {"param2": "value2"}

    def test_dataset_config_default_values(self):
        """Test DatasetConfig with minimal required fields."""
        config = DatasetConfig(
            name="test",
            source="source",
            categories=[],
            preprocessing={
                "min_interactions_per_user": 1,
                "min_interactions_per_item": 1,
            },
            model={},
        )

        assert config.name == "test"
        assert config.source == "source"
        assert config.categories == []
        assert config.preprocessing["min_interactions_per_user"] == 1
        assert config.preprocessing["min_interactions_per_item"] == 1
        assert config.model == {}

    def test_dataset_config_missing_required_params(self):
        """Test DatasetConfig raises error when missing required preprocessing params."""
        with pytest.raises(ValueError, match="Missing required preprocessing parameter"):
            DatasetConfig(
                name="test",
                source="source",
                categories=[],
                preprocessing={},  # Missing required parameters
                model={},
            )


class ConcreteDataLoader(DataLoader):
    """Concrete implementation of DataLoader for testing."""

    def load(self, path: Path) -> pd.DataFrame:
        """Mock load implementation."""
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


class ConcreteDataProcessor(DataProcessor):
    """Concrete implementation of DataProcessor for testing."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock process implementation."""
        return data.copy()


class ConcreteDataValidator(DataValidator):
    """Concrete implementation of DataValidator for testing."""

    def validate(self, data: pd.DataFrame) -> tuple[bool, list[str]]:
        """Mock validate implementation."""
        if data.empty:
            return False, ["Data is empty"]
        return True, []


class ConcreteDataSaver(DataSaver):
    """Concrete implementation of DataSaver for testing."""

    def save(self, data: pd.DataFrame, path: Path) -> None:
        """Mock save implementation."""
        pass


class TestDataLoader:
    """Test DataLoader abstract base class."""

    def test_data_loader_interface(self):
        """Test DataLoader can be instantiated with concrete implementation."""
        loader = ConcreteDataLoader()
        result = loader.load(Path("test.csv"))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "col1" in result.columns
        assert "col2" in result.columns

    def test_data_loader_abstract_methods(self):
        """Test DataLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataLoader()


class TestDataProcessor:
    """Test DataProcessor abstract base class."""

    def test_data_processor_interface(self):
        """Test DataProcessor can be instantiated with concrete implementation."""
        config = DatasetConfig(
            name="test",
            source="test",
            categories=[],
            preprocessing={
                "min_interactions_per_user": 1,
                "min_interactions_per_item": 1,
            },
            model={},
        )
        processor = ConcreteDataProcessor(config)
        input_df = pd.DataFrame({"col1": [1, 2, 3]})
        result = processor.process(input_df)

        assert isinstance(result, pd.DataFrame)
        assert result.equals(input_df)

    def test_data_processor_abstract_methods(self):
        """Test DataProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProcessor()


class TestDataValidator:
    """Test DataValidator abstract base class."""

    def test_data_validator_interface(self):
        """Test DataValidator can be instantiated with concrete implementation."""
        validator = ConcreteDataValidator()

        # Test with valid data
        valid_df = pd.DataFrame({"col1": [1, 2, 3]})
        is_valid, errors = validator.validate(valid_df)
        assert is_valid is True
        assert errors == []

        # Test with invalid data
        invalid_df = pd.DataFrame()
        is_valid, errors = validator.validate(invalid_df)
        assert is_valid is False
        assert len(errors) == 1
        assert errors[0] == "Data is empty"

    def test_data_validator_abstract_methods(self):
        """Test DataValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataValidator()


class TestDataSaver:
    """Test DataSaver abstract base class."""

    def test_data_saver_interface(self):
        """Test DataSaver can be instantiated with concrete implementation."""
        saver = ConcreteDataSaver()
        df = pd.DataFrame({"col1": [1, 2, 3]})

        # Should not raise any exception
        saver.save(df, Path("test.parquet"))

    def test_data_saver_abstract_methods(self):
        """Test DataSaver cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataSaver()
