"""
Preprocessing pipeline for FixFirst AI.

Orchestrates: load raw_reviews from Postgres -> clean -> deduplicate ->
filter to English -> split into train/val/test -> write Parquet.
"""

import sys
from typing import Dict

import pandas as pd

from fixfirst.config.configuration import ConfigurationManager
from fixfirst.constants import TRAIN_FILENAME, VAL_FILENAME, TEST_FILENAME
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.data_pipeline._preprocessing.text_cleaning import clean_dataframe
from fixfirst.data_pipeline._preprocessing.dedup import deduplicate_reviews
from fixfirst.data_pipeline._preprocessing.split import split_dataset


class PreprocessingPipeline:
    """Object-oriented preprocessing pipeline."""

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.settings = self.config_manager.get_settings()
        self.split_config = self.config_manager.get_split_config()

    def load_raw_reviews_df(self) -> pd.DataFrame:
        """Loads all raw_reviews rows from Postgres into a DataFrame."""
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import RawReview

        try:
            with get_db() as db:
                rows = db.query(RawReview).all()
                records = [
                    {
                        "id": str(r.id),
                        "source": r.source,
                        "app_id": r.app_id,
                        "review_text": r.review_text,
                        "rating": r.rating,
                        "review_date": r.review_date,
                        "raw_metadata": r.raw_metadata,
                    }
                    for r in rows
                ]
            df = pd.DataFrame(records)
            logging.info(f"load_raw_reviews_df: loaded {len(df)} rows from raw_reviews")
            return df
        except Exception as e:
            raise FixFirstException(e, sys) from e

    def run(self, write_output: bool = True) -> Dict[str, pd.DataFrame]:
        """Runs the full preprocessing pipeline."""
        try:
            df = self.load_raw_reviews_df()
            if df.empty:
                raise FixFirstException(
                    "raw_reviews is empty — run scripts/ingest_aware.py before preprocessing.", sys
                )

            df = clean_dataframe(df, text_col="review_text")
            df = deduplicate_reviews(df, text_col="review_text", app_col="app_id")

            train_df, val_df, test_df = split_dataset(
                df,
                config=self.split_config,
            )

            if write_output:
                out_dir = self.settings.resolve_path(self.settings.data_processed_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                train_df.to_parquet(out_dir / TRAIN_FILENAME, index=False)
                val_df.to_parquet(out_dir / VAL_FILENAME, index=False)
                test_df.to_parquet(out_dir / TEST_FILENAME, index=False)

                logging.info(f"run_preprocessing_pipeline: wrote train/val/test Parquet files to {out_dir}")

            return {"train": train_df, "val": val_df, "test": test_df}
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e


# Backward compatibility wrapper
def run_preprocessing_pipeline(write_output: bool = True) -> Dict[str, pd.DataFrame]:
    """Legacy function for backward compatibility."""
    return PreprocessingPipeline().run(write_output=write_output)