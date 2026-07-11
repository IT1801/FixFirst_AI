"""CLI entrypoint to compute and persist criticality scores."""

import argparse
import sys

from fixfirst.business._scoring.pipeline import CriticalityScorer
from fixfirst.constants import DEFAULT_HALF_LIFE_DAYS
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def main() -> int:
    """Run the criticality scoring pipeline."""
    parser = argparse.ArgumentParser(description="Compute and persist criticality scores.")
    parser.add_argument(
        "--half-life-days",
        type=int,
        default=DEFAULT_HALF_LIFE_DAYS,
        help="Recency decay half-life in days (default: 90)",
    )
    args = parser.parse_args()

    try:
        scorer = CriticalityScorer(half_life_days=args.half_life_days)
        n_written = scorer.run()
        logging.info(f"Scoring complete: {n_written} rows written to criticality_scores.")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
