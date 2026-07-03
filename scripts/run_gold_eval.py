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
edit src/fixfirst/evaluation/category_mapping.py — an incomplete mapping
silently shrinks the gold eval set rather than crashing. Check the
"report_mapping_coverage" log line printed on each run.
"""

import json
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.evaluation.gold_eval import run_gold_evaluation
from fixfirst.logging.logger import logging

if __name__ == "__main__":
    try:
        results = run_gold_evaluation()
        logging.info("Gold evaluation complete.")
        logging.info(json.dumps(results, indent=2))
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)