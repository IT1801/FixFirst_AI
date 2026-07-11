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
from fixfirst.ml._training.aspect_category.train import AspectCategoryTrainer


def main() -> int:
    """Run aspect category training."""
    parser = argparse.ArgumentParser(description="Train multi-label category classifier.")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to train on")
    args = parser.parse_args()

    try:
        trainer = AspectCategoryTrainer(limit=args.limit)
        metrics = trainer.train()
        logging.info(f"Final metrics: {metrics}")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())