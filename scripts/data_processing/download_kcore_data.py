#!/usr/bin/env python3
"""
Download k-core preprocessed subsets of Amazon Reviews 2023.

This uses the preprocessed 5-core versions which only include users and items
with at least 5 reviews each, making the data much more manageable while still
being representative across multiple categories.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# K-core dataset configurations available
KCORE_CONFIGS = {
    "All_Beauty": "0core_rating_only_All_Beauty",
    "Electronics": "0core_rating_only_Electronics", 
    "Books": "0core_rating_only_Books",
    "Sports_and_Outdoors": "0core_rating_only_Sports_and_Outdoors",
    "Home_and_Kitchen": "0core_rating_only_Home_and_Kitchen",
    "Cell_Phones_and_Accessories": "0core_rating_only_Cell_Phones_and_Accessories",
    "Toys_and_Games": "0core_rating_only_Toys_and_Games",
    "Office_Products": "0core_rating_only_Office_Products",
}

# For getting the full interaction data with text
FULL_CONFIGS = {
    "All_Beauty": "raw_review_All_Beauty",
    "Electronics": "raw_review_Electronics",
    "Books": "raw_review_Books", 
    "Sports_and_Outdoors": "raw_review_Sports_and_Outdoors",
    "Home_and_Kitchen": "raw_review_Home_and_Kitchen",
    "Cell_Phones_and_Accessories": "raw_review_Cell_Phones_and_Accessories",
    "Toys_and_Games": "raw_review_Toys_and_Games",
    "Office_Products": "raw_review_Office_Products",
}


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_kcore_interactions(category: str, min_interactions: int = 5) -> pd.DataFrame:
    """
    Get k-core filtered interactions for a category.
    
    This downloads just the user-item-rating tuples for users and items
    with at least min_interactions reviews.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {min_interactions}-core data for {category}")
    
    # Load the rating-only dataset (much smaller)
    config_name = KCORE_CONFIGS.get(category)
    if not config_name:
        raise ValueError(f"Category {category} not supported")
    
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        trust_remote_code=False,
        split="full"
    )
    
    # Convert to pandas
    df = dataset.to_pandas()
    
    # Apply k-core filtering
    while True:
        initial_size = len(df)
        
        # Count interactions per user and item
        user_counts = df['user_id'].value_counts()
        item_counts = df['parent_asin'].value_counts()
        
        # Keep only users and items with enough interactions
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df = df[
            df['user_id'].isin(valid_users) & 
            df['parent_asin'].isin(valid_items)
        ]
        
        # If no change, we've reached k-core
        if len(df) == initial_size:
            break
    
    logger.info(
        f"K-core filtering complete: {initial_size} -> {len(df)} interactions "
        f"({len(df)/initial_size*100:.1f}% retained)"
    )
    
    return df


def download_text_for_interactions(
    category: str,
    interaction_df: pd.DataFrame,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Download full review text for a subset of interactions.
    
    This is more efficient than downloading all reviews first.
    """
    logger = logging.getLogger(__name__)
    
    # If sampling, take a subset
    if sample_size and len(interaction_df) > sample_size:
        sampled_df = interaction_df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampling {sample_size} from {len(interaction_df)} interactions")
    else:
        sampled_df = interaction_df
    
    # Get unique user-item pairs
    unique_pairs = set(zip(sampled_df['user_id'], sampled_df['parent_asin']))
    logger.info(f"Need to fetch text for {len(unique_pairs)} unique reviews")
    
    # Load the full review dataset in streaming mode
    config_name = FULL_CONFIGS.get(category)
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        trust_remote_code=False,
        streaming=True,  # Stream to avoid loading everything
        split="full"
    )
    
    # Collect matching reviews
    reviews = []
    for review in tqdm(dataset, desc=f"Scanning {category} reviews"):
        if (review['user_id'], review['parent_asin']) in unique_pairs:
            reviews.append(review)
            
            # Stop if we've found all we need
            if len(reviews) >= len(unique_pairs):
                break
    
    return pd.DataFrame(reviews)


def main() -> None:
    """Execute the main download function."""
    parser = argparse.ArgumentParser(
        description="Download k-core filtered Amazon Reviews efficiently"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(KCORE_CONFIGS.keys()),
        default=["Electronics", "Books", "Sports_and_Outdoors", "Office_Products"],
        help="Categories to download"
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum interactions for k-core filtering (default: 5)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Maximum reviews per category (default: 100000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip downloading full text (only get ratings)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(args.categories)} categories with {args.min_interactions}-core filtering")
    
    total_reviews = 0
    
    for category in args.categories:
        try:
            logger.info(f"\nProcessing {category}...")
            
            # Get k-core interactions
            interactions_df = get_kcore_interactions(category, args.min_interactions)
            
            if args.skip_text:
                # Save just the interactions
                output_path = args.output_dir / f"{category}_interactions_{args.min_interactions}core.parquet"
                interactions_df.to_parquet(output_path, compression="snappy")
                logger.info(f"Saved {len(interactions_df)} interactions to {output_path}")
            else:
                # Get full review text for a sample
                reviews_df = download_text_for_interactions(
                    category, 
                    interactions_df,
                    args.sample_size
                )
                
                # Save the data
                output_path = args.output_dir / f"{category}_reviews_{args.min_interactions}core.parquet"
                reviews_df.to_parquet(output_path, compression="snappy")
                logger.info(f"Saved {len(reviews_df)} reviews to {output_path}")
            
            total_reviews += len(interactions_df)
            
        except Exception as e:
            logger.error(f"Failed to process {category}: {str(e)}")
            continue
    
    logger.info(f"\nTotal interactions across all categories: {total_reviews:,}")


if __name__ == "__main__":
    main()