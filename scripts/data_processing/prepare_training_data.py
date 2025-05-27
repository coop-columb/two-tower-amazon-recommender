#!/usr/bin/env python3
"""
Prepare diverse training data from multiple Amazon categories.

This script combines data from different product categories to create
a diverse dataset for training a robust recommendation model.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_category_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all available category data."""
    logger = logging.getLogger(__name__)

    # Find all parquet files
    parquet_files = list(data_dir.glob("*_reviews.parquet")) + list(
        data_dir.glob("*_5core.parquet")
    )

    category_data = {}

    for file_path in parquet_files:
        # Extract category name
        if "_reviews.parquet" in file_path.name:
            category = file_path.stem.replace("_reviews", "")
        else:
            category = file_path.stem.replace("_5core", "")

        logger.info(f"Loading {category} from {file_path.name}")
        df = pd.read_parquet(file_path)

        # Add category column
        df["category"] = category

        # Ensure consistent columns
        if "rating" in df.columns:
            # Full reviews data
            optional_cols = ["text", "title", "verified_purchase"]

            for col in optional_cols:
                if col not in df.columns:
                    df[col] = ""
        else:
            # 5-core data - convert rating from string if needed
            if df["rating"].dtype == "object":
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df["text"] = ""
            df["title"] = ""
            df["verified_purchase"] = None  # Use None instead of False for missing data

        category_data[category] = df
        logger.info(f"  Loaded {len(df):,} interactions for {category}")

    return category_data


def combine_and_balance_data(
    category_data: dict[str, pd.DataFrame], max_per_category: int | None = None
) -> pd.DataFrame:
    """Combine data from multiple categories with optional balancing."""
    logger = logging.getLogger(__name__)

    combined_dfs = []

    for category, df in category_data.items():
        if max_per_category and len(df) > max_per_category:
            # Sample to balance categories
            sampled_df = df.sample(n=max_per_category, random_state=42)
            logger.info(f"Sampled {max_per_category:,} from {len(df):,} for {category}")
        else:
            sampled_df = df

        combined_dfs.append(sampled_df)

    # Combine all data
    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Ensure consistent data types
    if "timestamp" in combined_df.columns:
        combined_df["timestamp"] = pd.to_numeric(combined_df["timestamp"], errors="coerce")

    # Ensure rating is numeric
    if combined_df["rating"].dtype == "object":
        combined_df["rating"] = pd.to_numeric(combined_df["rating"], errors="coerce")

    # Handle verified_purchase column - drop it if it has mixed types
    if "verified_purchase" in combined_df.columns:
        # Check if column has consistent type
        try:
            combined_df["verified_purchase"] = combined_df["verified_purchase"].astype(bool)
        except (TypeError, ValueError):
            # If conversion fails, drop the column
            combined_df = combined_df.drop("verified_purchase", axis=1)
            logger.info("Dropped verified_purchase column due to mixed types")

    return combined_df


def create_user_item_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """Create mappings from user/item IDs to integers."""
    # Get unique users and items
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["parent_asin"].unique())

    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    return user_to_idx, item_to_idx


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyze dataset statistics."""
    stats = {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_items": df["parent_asin"].nunique(),
        "categories": df["category"].unique().tolist(),
        "interactions_per_category": df["category"].value_counts().to_dict(),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
        "avg_rating": df["rating"].mean(),
        "sparsity": 1 - (len(df) / (df["user_id"].nunique() * df["parent_asin"].nunique())),
    }

    # User statistics
    user_counts = df["user_id"].value_counts()
    stats["user_stats"] = {
        "avg_interactions": user_counts.mean(),
        "min_interactions": user_counts.min(),
        "max_interactions": user_counts.max(),
        "std_interactions": user_counts.std(),
    }

    # Item statistics
    item_counts = df["parent_asin"].value_counts()
    stats["item_stats"] = {
        "avg_interactions": item_counts.mean(),
        "min_interactions": item_counts.min(),
        "max_interactions": item_counts.max(),
        "std_interactions": item_counts.std(),
    }

    return stats


def main() -> None:
    """Execute the main data preparation function."""
    parser = argparse.ArgumentParser(
        description="Prepare diverse training data from multiple categories"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/raw",
        help="Directory containing category data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=100000,
        help="Maximum interactions per category for balancing",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all category data
    logger.info("Loading category data...")
    category_data = load_category_data(args.data_dir)

    if not category_data:
        logger.error("No data files found!")
        return

    # Combine and balance
    logger.info("\nCombining and balancing data...")
    combined_df = combine_and_balance_data(category_data, args.max_per_category)

    # Create mappings
    logger.info("\nCreating user/item mappings...")
    user_to_idx, item_to_idx = create_user_item_mappings(combined_df)

    # Add encoded IDs
    combined_df["user_idx"] = combined_df["user_id"].map(user_to_idx)
    combined_df["item_idx"] = combined_df["parent_asin"].map(item_to_idx)

    # Analyze dataset
    logger.info("\nAnalyzing combined dataset...")
    stats = analyze_dataset(combined_df)

    # Save processed data
    output_path = args.output_dir / "combined_interactions.parquet"
    combined_df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"\nSaved combined data to {output_path}")

    # Save mappings
    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": {v: k for k, v in user_to_idx.items()},
        "idx_to_item": {v: k for k, v in item_to_idx.items()},
    }

    import pickle  # nosec B403 - pickle is safe here as we control the data

    mappings_path = args.output_dir / "mappings.pkl"
    with open(mappings_path, "wb") as f:
        pickle.dump(mappings, f)
    logger.info(f"Saved mappings to {mappings_path}")

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total interactions: {stats['total_interactions']:,}")
    logger.info(f"Unique users: {stats['unique_users']:,}")
    logger.info(f"Unique items: {stats['unique_items']:,}")
    logger.info(f"Categories: {', '.join(stats['categories'])}")
    logger.info(f"Sparsity: {stats['sparsity']:.2%}")
    logger.info(f"Average rating: {stats['avg_rating']:.2f}")

    logger.info("\nInteractions per category:")
    for cat, count in stats["interactions_per_category"].items():
        logger.info(f"  {cat}: {count:,}")

    logger.info("\nRating distribution:")
    for rating, count in stats["rating_distribution"].items():
        logger.info(f"  {rating}: {count:,} ({count/stats['total_interactions']*100:.1f}%)")

    logger.info("\nUser statistics:")
    logger.info(f"  Avg interactions: {stats['user_stats']['avg_interactions']:.1f}")
    min_u = stats["user_stats"]["min_interactions"]
    max_u = stats["user_stats"]["max_interactions"]
    logger.info(f"  Range: {min_u}-{max_u}")

    logger.info("\nItem statistics:")
    logger.info(f"  Avg interactions: {stats['item_stats']['avg_interactions']:.1f}")
    min_i = stats["item_stats"]["min_interactions"]
    max_i = stats["item_stats"]["max_interactions"]
    logger.info(f"  Range: {min_i}-{max_i}")


if __name__ == "__main__":
    main()
