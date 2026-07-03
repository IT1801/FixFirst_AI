"""
CLI entrypoint to run hybrid ABSA inference over a batch of reviews and
write results into review_aspects.

Usage:
    PYTHONPATH=src python scripts/run_hybrid_inference.py --split test [--limit N]

Requires:
  - Trained model artifacts (scripts/train_aspect_category.py,
    scripts/train_aspect_sentiment.py)
  - A valid LLM_PROVIDER API key in .env (for fallback routing)
  - features_master seeded (scripts/seed_features.py)

Prints the fallback-rate breakdown at the end — this is the number for
the README ("X% of inferences required LLM fallback").
"""

import argparse
import sys

import pandas as pd

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.inference.pipeline import run_batch_hybrid_inference
from fixfirst.logging.logger import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid ABSA inference over a review split.")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test", help="Which processed split to run inference on"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N reviews")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without writing to review_aspects")
    args = parser.parse_args()

    try:
        split_path = settings.resolve_path(settings.data_processed_dir) / f"{args.split}.parquet"
        if not split_path.exists():
            raise FixFirstException(f"{split_path} not found — run scripts/run_preprocessing.py first.", sys)

        reviews_df = pd.read_parquet(split_path)
        if args.limit:
            reviews_df = reviews_df.head(args.limit)

        result = run_batch_hybrid_inference(reviews_df, write_to_db=not args.dry_run)

        stats = result["fallback_stats"]
        logging.info(
            f"Hybrid inference complete: {result['n_reviews_processed']} reviews -> "
            f"{stats['total']} aspects ({result['n_aspects_written']} written to DB)"
        )
        logging.info(
            f"Fallback breakdown: finetuned={stats['finetuned_rate']:.1%}, "
            f"llm_fallback={stats['llm_fallback_rate']:.1%}"
        )
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)