"""
CLI entrypoint to train the aspect sentiment (single-label, 3-class) classifier.

Usage:
    PYTHONPATH=src python scripts/train_aspect_sentiment.py [--limit N]

Requires:
  - data/silver_labels/silver_labels.parquet (scripts/run_silver_labeling.py)
  - MLflow tracking server reachable at settings.mlflow_tracking_uri
    (docker compose up -d mlflow)
  - torch + transformers installed (pip install -r requirements-training.txt)

Use --limit while sanity-checking the pipeline runs end-to-end before
committing to a full training run.
"""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.models.aspect_sentiment.train import train_aspect_sentiment_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the aspect sentiment classifier.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N silver-label rows (smoke test)")
    args = parser.parse_args()

    try:
        metrics = train_aspect_sentiment_model(limit=args.limit)
        logging.info(f"Training complete. Final validation metrics: {metrics}")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)