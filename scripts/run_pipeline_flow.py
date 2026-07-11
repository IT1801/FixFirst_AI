"""
CLI entrypoint to run the full Prefect pipeline flow.

Usage:
    PYTHONPATH=src python scripts/run_pipeline_flow.py [options]

    # First-time bootstrap (ingests AWARE, seeds features, preprocesses,
    # runs hybrid inference, scores):
    PYTHONPATH=src python scripts/run_pipeline_flow.py \\
        --run-ingestion --aware-csv data/raw/aware_reviews.csv

    # Recurring run (skip ingestion, re-run inference + scoring on new data):
    PYTHONPATH=src python scripts/run_pipeline_flow.py --inference-split test
"""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.orchestration.flows import fixfirst_pipeline_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full FixFirst AI pipeline flow.")
    parser.add_argument("--run-ingestion", action="store_true", help="Also run AWARE ingestion (one-time bootstrap)")
    parser.add_argument("--aware-csv", type=str, default=None, help="Path to AWARE CSV (required if --run-ingestion)")
    parser.add_argument("--inference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--inference-limit", type=int, default=None)
    parser.add_argument("--half-life-days", type=int, default=90)
    args = parser.parse_args()

    try:
        summary = fixfirst_pipeline_flow(
            run_ingestion=args.run_ingestion,
            aware_csv_path=args.aware_csv,
            inference_split=args.inference_split,
            inference_limit=args.inference_limit,
            half_life_days=args.half_life_days,
        )
        logging.info(f"Pipeline flow complete: {summary}")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)
