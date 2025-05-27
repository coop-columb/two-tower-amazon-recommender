#!/usr/bin/env python3
"""
Download and prepare Amazon Reviews 2023 dataset.

This script downloads the dataset from HuggingFace Hub and performs
initial data validation and structure analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/data_download.log", mode="a"),
        ],
    )


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_category_data(
    category: str, output_dir: Path, sample_size: int | None = None
) -> None:
    """
    Download data for a specific category.

    Args:
        category: Product category to download
        output_dir: Directory to save processed data
        sample_size: Optional sample size for testing
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading {category} data...")

    try:
        # Download reviews
        reviews_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", trust_remote_code=True
        )

        # Download metadata
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", trust_remote_code=True
        )

        # Convert to pandas for easier manipulation
        reviews_df = reviews_dataset["full"].to_pandas()
        meta_df = meta_dataset["full"].to_pandas()

        # Sample data if specified
        if sample_size and len(reviews_df) > sample_size:
            reviews_df = reviews_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} reviews for {category}")

        # Save to parquet for efficient storage
        reviews_output = output_dir / f"{category}_reviews.parquet"
        meta_output = output_dir / f"{category}_meta.parquet"

        reviews_df.to_parquet(reviews_output, compression="snappy")
        meta_df.to_parquet(meta_output, compression="snappy")

        logger.info(f"Saved {len(reviews_df)} reviews and {len(meta_df)} items for {category}")

    except Exception as e:
        logger.error(f"Failed to download {category}: {str(e)}")
        raise


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/data_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="data/raw", help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to download (default: all from config)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for testing (downloads full dataset if not specified)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Determine categories to download
    if args.categories:
        categories = args.categories
    else:
        categories = config["dataset"]["categories"]

    logger.info(f"Starting download for {len(categories)} categories...")

    # Download each category
    for category in tqdm(categories, desc="Downloading categories"):
        try:
            download_category_data(
                category=category, output_dir=args.output_dir, sample_size=args.sample_size
            )
        except Exception as e:
            logger.error(f"Failed to process {category}: {str(e)}")
            continue

    logger.info("Download process completed!")


if __name__ == "__main__":
    main()
