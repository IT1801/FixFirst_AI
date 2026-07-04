"""
Preprocessing pipeline for FixFirst AI.

Orchestrates: load raw_reviews from Postgres -> clean -> deduplicate ->
filter to English -> split into train/val/test -> write Parquet.

Outputs Parquet (not CSV/xlsx) for dtype preservation and efficient
downstream loading, consistent with the customer-segmentation-retention
project's convention for the same reasons (large row counts, dtype safety).

Usage:
    PYTHONPATH=src python scripts/run_preprocessing.py
"""

import sys

import pandas as pd

from fixfirst.core.config import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.data_pipeline._preprocessing.text_cleaning import clean_dataframe
from fixfirst.data_pipeline._preprocessing.dedup import deduplicate_reviews
from fixfirst.data_pipeline._preprocessing.split import split_dataset


def load_raw_reviews_df() -> pd.DataFrame:
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
        raise FixFirstException(e, sys)


def run_preprocessing_pipeline(write_output: bool = True) -> dict:
    """
    Runs the full preprocessing pipeline. Returns a dict of the resulting
    train/val/test DataFrames (and writes them to Parquet if write_output).
    """
    try:
        df = load_raw_reviews_df()
        if df.empty:
            raise FixFirstException(
                "raw_reviews is empty — run scripts/ingest_aware.py before preprocessing.", sys
            )

        df = clean_dataframe(df, text_col="review_text")
        df = deduplicate_reviews(df, text_col="review_text", app_col="app_id")

        if write_output:
            out_dir = settings.resolve_path(settings.data_processed_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            train_df.to_parquet(out_dir / "train.parquet", index=False)
            val_df.to_parquet(out_dir / "val.parquet", index=False)
            test_df.to_parquet(out_dir / "test.parquet", index=False)

            logging.info(f"run_preprocessing_pipeline: wrote train/val/test Parquet files to {out_dir}")

        return {"train": train_df, "val": val_df, "test": test_df}
    except Exception as e:
        raise FixFirstException(e, sys)