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
from fixfirst.ml._training.aspect_sentiment.train import AspectSentimentTrainer


def main() -> int:
    """Run aspect sentiment training."""
    parser = argparse.ArgumentParser(description="Train 3-class sentiment classifier.")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to train on")
    args = parser.parse_args()

    try:
        trainer = AspectSentimentTrainer(limit=args.limit)
        trainer.train()
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
