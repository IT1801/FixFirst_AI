"""
CLI entrypoint to run both fine-tuned models against AWARE gold labels.

Usage:
    PYTHONPATH=src python scripts/run_gold_eval.py

Requires:
  - data/processed/test.parquet (scripts/run_preprocessing.py)
  - Trained model artifacts at artifacts/models/aspect_category/final/
    and artifacts/models/aspect_sentiment/final/
    (scripts/train_aspect_category.py, scripts/train_aspect_sentiment.py)

BEFORE RUNNING: inspect actual AWARE aspect_category values and verify/
edit src/fixfirst/ml/_evaluation/category_mapping.py — an incomplete mapping
silently shrinks the gold eval set rather than crashing. Check the
"report_mapping_coverage" log line printed on each run.
"""

import json
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._evaluation.gold_eval import GoldEvaluator
from fixfirst.logging.logger import logging


def main() -> int:
    """Run gold evaluation."""
    try:
        evaluator = GoldEvaluator()
        metrics = evaluator.evaluate()
        logging.info("Gold evaluation complete.")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
