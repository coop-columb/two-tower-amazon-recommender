#!/usr/bin/env python3
"""
Exploratory data analysis for Amazon Reviews 2023 dataset.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go


def analyze_category_data(category: str, data_dir: Path) -> Dict:
    """Analyze data for a specific category."""
    logger = logging.getLogger(__name__)
    
    # Load data
    reviews_path = data_dir / f"{category}_reviews.parquet"
    meta_path = data_dir / f"{category}_meta.parquet"
    
    if not reviews_path.exists():
        logger.warning(f"Reviews file not found for {category}")
        return {}
    
    reviews_df = pd.read_parquet(reviews_path)
    
    analysis = {
        "category": category,
        "num_reviews": len(reviews_df),
        "num_unique_users": reviews_df["user_id"].nunique(),
        "num_unique_items": reviews_df["parent_asin"].nunique(),
        "rating_distribution": reviews_df["rating"].value_counts().to_dict(),
        "avg_rating": reviews_df["rating"].mean(),
        "review_length_stats": {
            "mean": reviews_df["text"].str.len().mean(),
            "median": reviews_df["text"].str.len().median(),
            "std": reviews_df["text"].str.len().std(),
        }
    }
    
    return analysis


def create_visualizations(analysis_results: List[Dict], output_dir: Path) -> None:
    """Create comprehensive visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Category comparison
    categories = [r["category"] for r in analysis_results]
    num_reviews = [r["num_reviews"] for r in analysis_results]
    
    fig = px.bar(
        x=categories, 
        y=num_reviews,
        title="Number of Reviews by Category",
        labels={"x": "Category", "y": "Number of Reviews"}
    )
    fig.write_html(output_dir / "reviews_by_category.html")
    
    # Rating distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(analysis_results[:6]):
        if i < len(axes):
            ratings = list(result["rating_distribution"].keys())
            counts = list(result["rating_distribution"].values())
            
            axes[i].bar(ratings, counts)
            axes[i].set_title(f"{result['category']} Rating Distribution")
            axes[i].set_xlabel("Rating")
            axes[i].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Explore Amazon Reviews dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/raw",
        help="Directory containing downloaded data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="docs/research/data_analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Find all categories
    parquet_files = list(args.data_dir.glob("*_reviews.parquet"))
    categories = [f.stem.replace("_reviews", "") for f in parquet_files]
    
    logger.info(f"Found {len(categories)} categories to analyze")
    
    # Analyze each category
    analysis_results = []
    for category in categories:
        logger.info(f"Analyzing {category}...")
        result = analyze_category_data(category, args.data_dir)
        if result:
            analysis_results.append(result)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(analysis_results, args.output_dir)
    
    # Save summary report
    summary_df = pd.DataFrame(analysis_results)
    summary_df.to_csv(args.output_dir / "category_summary.csv", index=False)
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()