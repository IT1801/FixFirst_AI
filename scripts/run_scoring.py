"""
CLI entrypoint to compute and persist criticality scores.

Usage:
    PYTHONPATH=src python scripts/run_scoring.py [--half-life-days N]

Requires review_aspects to be populated (scripts/run_hybrid_inference.py).
"""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.scoring.criticality import DEFAULT_HALF_LIFE_DAYS
from fixfirst.scoring.pipeline import run_scoring_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and persist criticality scores.")
    parser.add_argument(
        "--half-life-days",
        type=int,
        default=DEFAULT_HALF_LIFE_DAYS,
        help="Recency decay half-life in days (default: 90)",
    )
    args = parser.parse_args()

    try:
        n_written = run_scoring_pipeline(half_life_days=args.half_life_days)
        logging.info(f"Scoring complete: {n_written} rows written to criticality_scores.")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)