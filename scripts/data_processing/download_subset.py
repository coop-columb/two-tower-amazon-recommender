#!/usr/bin/env python3
"""
Download a diverse subset of Amazon Reviews for development.

This script downloads smaller, manageable samples from multiple categories
to ensure we have representative data for testing our recommendation system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from src.data.amazon_loader import AmazonReviewsLoader, DatasetConfig


# Representative categories with different characteristics
DIVERSE_CATEGORIES = {
    "Electronics": "High-value items with technical reviews",
    "Books": "Subjective content with detailed opinions", 
    "Sports_and_Outdoors": "Functional items with usage-based reviews",
    "Grocery_and_Gourmet_Food": "Consumables with repeat purchases",
    "Toys_and_Games": "Gift items with age-specific feedback",
}


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: Path) -> DatasetConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return DatasetConfig(
        name=config_dict["dataset"]["name"],
        source=config_dict["dataset"]["source"],
        categories=list(DIVERSE_CATEGORIES.keys()),
        preprocessing=config_dict["preprocessing"],
        model=config_dict["model"]
    )


def download_category_subset(
    loader: AmazonReviewsLoader,
    category: str,
    sample_size: int,
    output_dir: Path
) -> Tuple[int, int]:
    """
    Download a subset of data for a specific category.
    
    Returns:
        Tuple of (num_reviews, num_products)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading {category}: {DIVERSE_CATEGORIES.get(category, 'General products')}")
    
    try:
        # Download with sampling
        reviews_df = loader.load_category(
            category=category,
            data_type="reviews",
            sample_size=sample_size
        )
        
        meta_df = loader.load_category(
            category=category,
            data_type="meta",
            sample_size=None  # Get all metadata for sampled products
        )
        
        # Filter metadata to only include products in our sample
        unique_products = reviews_df['parent_asin'].unique()
        meta_df = meta_df[meta_df['parent_asin'].isin(unique_products)]
        
        # Save to output directory
        reviews_output = output_dir / f"{category}_reviews.parquet"
        meta_output = output_dir / f"{category}_meta.parquet"
        
        reviews_df.to_parquet(reviews_output, compression="snappy", index=False)
        meta_df.to_parquet(meta_output, compression="snappy", index=False)
        
        logger.info(
            f"Saved {len(reviews_df):,} reviews and {len(meta_df):,} products for {category}"
        )
        
        return len(reviews_df), len(meta_df)
        
    except Exception as e:
        logger.error(f"Failed to download {category}: {str(e)}")
        raise


def main() -> None:
    """Execute the main download function."""
    parser = argparse.ArgumentParser(
        description="Download diverse subset of Amazon Reviews for development"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/data_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of reviews per category (default: 50000)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(DIVERSE_CATEGORIES.keys()),
        help="Specific categories to download"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize loader
    loader = AmazonReviewsLoader(
        config=config,
        cache_dir=Path("data/cache"),
        trust_remote_code=False  # Security first
    )
    
    # Determine categories
    categories = args.categories or list(DIVERSE_CATEGORIES.keys())
    
    logger.info(f"Starting download of {len(categories)} diverse categories")
    logger.info(f"Sample size per category: {args.sample_size:,}")
    
    # Track statistics
    total_reviews = 0
    total_products = 0
    
    # Download each category
    for category in categories:
        try:
            num_reviews, num_products = download_category_subset(
                loader=loader,
                category=category,
                sample_size=args.sample_size,
                output_dir=args.output_dir
            )
            total_reviews += num_reviews
            total_products += num_products
            
        except Exception as e:
            logger.error(f"Skipping {category} due to error: {str(e)}")
            continue
    
    # Summary statistics
    logger.info("=" * 50)
    logger.info("Download Summary:")
    logger.info(f"Total reviews: {total_reviews:,}")
    logger.info(f"Total unique products: {total_products:,}")
    logger.info(f"Categories downloaded: {len(categories)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()