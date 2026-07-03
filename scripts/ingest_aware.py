"""
CLI entrypoint to ingest an AWARE CSV file into raw_reviews.

Usage:
    PYTHONPATH=src python scripts/ingest_aware.py --csv data/raw/aware_reviews.csv

Requires the `db` service to be running and the schema already created
(see scripts/init_db.py).
"""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ingestion.aware_loader import ingest_aware_csv
from fixfirst.logging.logger import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest AWARE dataset CSV into raw_reviews.")
    parser.add_argument("--csv", required=True, help="Path to the AWARE CSV file")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    args = parser.parse_args()

    try:
        count = ingest_aware_csv(args.csv, batch_size=args.batch_size)
        logging.info(f"Successfully ingested {count} reviews from {args.csv}")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)