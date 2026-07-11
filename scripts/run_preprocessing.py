"""CLI entrypoint to run the full preprocessing pipeline."""

import sys

from fixfirst.data_pipeline._preprocessing.pipeline import PreprocessingPipeline
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def main() -> int:
    """Run preprocessing and log the resulting split sizes."""
    try:
        pipeline = PreprocessingPipeline()
        result = pipeline.run(write_output=True)
        for split_name, split_df in result.items():
            logging.info(f"{split_name}: {len(split_df)} rows")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
