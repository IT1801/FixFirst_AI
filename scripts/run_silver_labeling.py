"""
CLI entrypoint to run zero-shot silver-labeling over the preprocessed train split.

Usage:
        PYTHONPATH=src python scripts/run_silver_labeling.py [--limit N] [--batch-size N]

Requires:
  - Postgres running with features_master seeded (scripts/seed_features.py)
  - data/processed/train.parquet present (scripts/run_preprocessing.py)
    - torch + transformers installed (run `make install-training` first)

Uses a local zero-shot classifier instead of the LLM labeling path.
"""

import argparse
import sys

import pandas as pd

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.labeling.silver_labeler import batch_label_reviews
from fixfirst.labeling.taxonomy import load_active_taxonomy
from fixfirst.logging.logger import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot silver-labeling over the train split.")
    parser.add_argument("--limit", type=int, default=None, help="Only label the first N reviews (for testing)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.zero_shot_batch_size,
        help="Reviews per zero-shot batch (5-10 is a good starting range)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore saved checkpoints and relabel from scratch")
    args = parser.parse_args()

    try:
        train_path = settings.resolve_path(settings.data_processed_dir) / "train.parquet"
        if not train_path.exists():
            raise FixFirstException(
                f"{train_path} not found — run scripts/run_preprocessing.py first.", sys
            )

        reviews_df = pd.read_parquet(train_path)
        if args.limit:
            reviews_df = reviews_df.head(args.limit)

        taxonomy = load_active_taxonomy()

        logging.info(f"Starting zero-shot silver labeling on {len(reviews_df)} reviews using {settings.zero_shot_model_name}...")
        labels_df = batch_label_reviews(
            reviews_df,
            taxonomy,
            batch_size=args.batch_size,
            write_output=True,
            resume_from_checkpoint=not args.no_resume,
        )
        logging.info(f"Silver labeling complete: {len(labels_df)} aspect labels produced.")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)