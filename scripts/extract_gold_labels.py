"""CLI entrypoint to extract dataset gold labels for training."""

import argparse
import sys

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._labeling.extractor import GoldLabelExtractor

def main() -> int:
    """Extract gold labels from the processed train split."""
    parser = argparse.ArgumentParser(description="Extract gold labels from the train split.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N reviews (for testing)")
    # accept batch-size just so Makefile doesn't break if it passes it, but ignore it
    parser.add_argument("--batch-size", type=int, default=None, help="Ignored")
    args = parser.parse_args()

    try:
        extractor = GoldLabelExtractor(limit=args.limit)
        extractor.run()
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1

if __name__ == "__main__":
    sys.exit(main())
