"""
CLI entrypoint to run the full preprocessing pipeline.

Usage:
    PYTHONPATH=src python scripts/run_preprocessing.py

Requires raw_reviews to already be populated (see scripts/ingest_aware.py).
"""

import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.preprocessing.pipeline import run_preprocessing_pipeline

if __name__ == "__main__":
    try:
        result = run_preprocessing_pipeline(write_output=True)
        for split_name, split_df in result.items():
            logging.info(f"{split_name}: {len(split_df)} rows")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)