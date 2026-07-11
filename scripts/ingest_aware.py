"""CLI entrypoint to ingest an AWARE CSV file into raw_reviews."""

import argparse
import sys

from fixfirst.data_pipeline.ingestion import AWAREIngestor
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def main() -> int:
    """Run the AWARE ingestion CLI."""
    parser = argparse.ArgumentParser(description="Ingest AWARE dataset CSV into raw_reviews.")
    parser.add_argument("--csv", required=True, help="Path to the AWARE CSV file")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    args = parser.parse_args()

    try:
        ingestor = AWAREIngestor(csv_path=args.csv)
        # Using the wrapper property directly from the parsed args isn't strictly necessary since we can just use the config
        count = ingestor.load(ingestor.transform(ingestor.extract()), batch_size=args.batch_size)
        logging.info(f"Successfully ingested {count} reviews from {args.csv}")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
