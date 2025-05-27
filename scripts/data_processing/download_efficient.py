#!/usr/bin/env python3
"""
Efficiently download diverse Amazon Reviews data using preprocessed 5-core subsets.

The 5-core subsets only include users and items with at least 5 reviews each,
which dramatically reduces data size while maintaining quality for recommendations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

# Categories we want for a diverse dataset
DIVERSE_CATEGORIES = [
    "Electronics",  # Tech products - objective features
    "Books",  # Media - subjective taste
    "Sports_and_Outdoors",  # Functional items - performance focus
    "Home_and_Kitchen",  # Household - practical reviews
    "Office_Products",  # Business - professional use
]

# Map categories to their 5-core dataset configs
KCORE_CONFIGS = {
    "Electronics": "5core_timestamp_Electronics",
    "Books": "5core_timestamp_Books",
    "Sports_and_Outdoors": "5core_timestamp_Sports_and_Outdoors",
    "Home_and_Kitchen": "5core_timestamp_Home_and_Kitchen",
    "Office_Products": "5core_timestamp_Office_Products",
    "Toys_and_Games": "5core_timestamp_Toys_and_Games",
    "Cell_Phones_and_Accessories": "5core_timestamp_Cell_Phones_and_Accessories",
}


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def download_kcore_data(
    category: str, max_samples: Optional[int] = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Download 5-core preprocessed data for a category.

    Returns:
        Tuple of (DataFrame, statistics dictionary)
    """
    logger = logging.getLogger(__name__)

    config_name = KCORE_CONFIGS.get(category)
    if not config_name:
        raise ValueError(f"No 5-core config for category: {category}")

    logger.info(f"Loading 5-core data for {category}...")

    # Load the dataset - these have train/valid/test splits
    datasets = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", config_name, trust_remote_code=False
    )

    # Combine all splits
    all_data = []
    for split in ["train", "valid", "test"]:
        if split in datasets:
            all_data.append(datasets[split].to_pandas())

    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    # Convert timestamp to int if it's string
    if df["timestamp"].dtype == "object":
        df["timestamp"] = df["timestamp"].astype(int)

    # Sample if requested
    original_size = len(df)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Sampled {max_samples} from {original_size} interactions")

    # Calculate statistics
    stats = {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_items": df["parent_asin"].nunique(),
        "sparsity": 1 - (len(df) / (df["user_id"].nunique() * df["parent_asin"].nunique())),
        "avg_user_interactions": len(df) / df["user_id"].nunique(),
        "avg_item_interactions": len(df) / df["parent_asin"].nunique(),
    }

    return df, stats


def combine_categories(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple category dataframes with category labels."""
    combined_dfs = []

    for category, df in dataframes.items():
        df = df.copy()
        df["category"] = category
        combined_dfs.append(df)

    return pd.concat(combined_dfs, ignore_index=True)


def main() -> None:
    """Execute the main download function."""
    parser = argparse.ArgumentParser(
        description="Efficiently download diverse 5-core Amazon Reviews data"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(KCORE_CONFIGS.keys()),
        default=DIVERSE_CATEGORIES,
        help="Categories to download",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=200000,
        help="Maximum interactions per category (default: 200k)",
    )
    parser.add_argument("--output-dir", type=Path, default="data/raw", help="Output directory")
    parser.add_argument(
        "--combine", action="store_true", help="Combine all categories into one file"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {len(args.categories)} diverse categories")
    logger.info(f"Categories: {', '.join(args.categories)}")
    logger.info(f"Max interactions per category: {args.max_per_category:,}")

    # Download each category
    all_dataframes = {}
    all_stats = {}

    for category in args.categories:
        try:
            df, stats = download_kcore_data(category, args.max_per_category)
            all_dataframes[category] = df
            all_stats[category] = stats

            # Save individual category file
            if not args.combine:
                output_path = args.output_dir / f"{category}_5core.parquet"
                df.to_parquet(output_path, compression="snappy", index=False)
                logger.info(f"Saved {category} to {output_path}")

            # Log statistics
            logger.info(
                f"{category} stats: {stats['total_interactions']:,} interactions, "
                f"{stats['unique_users']:,} users, {stats['unique_items']:,} items"
            )

        except Exception as e:
            logger.error(f"Failed to download {category}: {str(e)}")
            continue

    # Combine if requested
    if args.combine and all_dataframes:
        logger.info("\nCombining all categories...")
        combined_df = combine_categories(all_dataframes)

        output_path = args.output_dir / "combined_5core.parquet"
        combined_df.to_parquet(output_path, compression="snappy", index=False)

        logger.info(f"Saved combined dataset to {output_path}")
        logger.info(f"Total interactions: {len(combined_df):,}")
        logger.info(f"Total unique users: {combined_df['user_id'].nunique():,}")
        logger.info(f"Total unique items: {combined_df['parent_asin'].nunique():,}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    total_interactions = sum(stats["total_interactions"] for stats in all_stats.values())
    logger.info(f"Total interactions downloaded: {total_interactions:,}")

    for category, stats in all_stats.items():
        logger.info(f"\n{category}:")
        logger.info(f"  Interactions: {stats['total_interactions']:,}")
        logger.info(f"  Users: {stats['unique_users']:,}")
        logger.info(f"  Items: {stats['unique_items']:,}")
        logger.info(f"  Sparsity: {stats['sparsity']:.2%}")
        logger.info(f"  Avg user interactions: {stats['avg_user_interactions']:.1f}")
        logger.info(f"  Avg item interactions: {stats['avg_item_interactions']:.1f}")


if __name__ == "__main__":
    main()
