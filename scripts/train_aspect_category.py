"""
CLI entrypoint to train the aspect category (multi-label) classifier.

Usage:
    PYTHONPATH=src python scripts/train_aspect_category.py [--limit N]

Requires:
  - data/processed/train.parquet (scripts/run_preprocessing.py)
  - data/silver_labels/{silver_labels,labeling_failures}.parquet
    (scripts/run_silver_labeling.py)
  - MLflow tracking server reachable at settings.mlflow_tracking_uri
    (docker compose up -d mlflow)
  - torch + transformers installed (not in requirements.txt by default —
    see requirements-training.txt, since they're heavy and only needed
    on a training machine, not the API/dashboard serving path)

Use --limit while sanity-checking the pipeline runs end-to-end before
committing to a full training run.
"""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.models.aspect_category.train import train_aspect_category_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the aspect category multi-label classifier.")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N attempted reviews (smoke test)")
    args = parser.parse_args()

    try:
        metrics = train_aspect_category_model(limit=args.limit)
        logging.info(f"Training complete. Final validation metrics: {metrics}")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)