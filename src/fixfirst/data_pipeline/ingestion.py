"""AWARE dataset loader for FixFirst AI."""

import sys
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd

from fixfirst.config.configuration import AWAREIngestionConfig, ConfigurationManager
from fixfirst.constants import RAW_METADATA_AWARE, RAW_DATA_FILE, SOURCE_AWARE
from fixfirst.core.db import RawReview, get_db
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def extract(
    config: AWAREIngestionConfig,
    csv_path: Path = RAW_DATA_FILE,
) -> pd.DataFrame:
    """Load and validate the raw AWARE CSV.
    """
    try:
        resolved_path = Path(csv_path)

        logging.info(f"Loading AWARE CSV from {resolved_path}")

        if not resolved_path.exists():
            raise FileNotFoundError(f"AWARE CSV not found at {resolved_path}")

        dataframe = pd.read_csv(resolved_path, keep_default_na=False)

        logging.info(f"Rows loaded: {len(dataframe):,}")
        logging.info(f"Columns: {dataframe.columns.tolist()}")

        _validate_raw(dataframe, config)

        return dataframe
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def _validate_raw(df: pd.DataFrame, config: AWAREIngestionConfig) -> None:
    """Validate the raw AWARE dataframe."""
    try:
        expected_columns = [
            config.domain_column,
            config.app_column,
            config.sentence_column,
            config.category_column,
            config.term_column,
            config.sentiment_column,
        ]

        missing_columns = [column for column in expected_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(
                "AWARE CSV is missing expected columns: "
                f"{missing_columns}. Found columns: {df.columns.tolist()}."
            )

        if df.empty:
            raise ValueError("Extracted AWARE dataframe is empty.")

        logging.info("AWARE raw data validation passed")
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def transform_to_raw_reviews(df: pd.DataFrame, config: AWAREIngestionConfig) -> list[dict]:
    """Transform raw AWARE rows into raw_reviews-shaped records."""
    try:
        app_col = config.app_column
        domain_col = config.domain_column
        sentence_col = config.sentence_column
        term_col = config.term_column
        category_col = config.category_column
        sentiment_col = config.sentiment_column
        from_col = config.from_column
        to_col = config.to_column

        logging.info("Transforming AWARE rows into raw_reviews records")

        group_keys = [app_col, domain_col, sentence_col]
        has_span_columns = (
            from_col in df.columns and
            to_col in df.columns
        )
        if not has_span_columns:
            logging.info("Span columns are not present; annotations will omit offsets")

        records: list[dict] = []

        for (app_name, domain, sentence), group in df.groupby(group_keys, dropna=False):
            annotations = []

            for _, row in group.iterrows():
                aspect = row[term_col]
                category = row[category_col]
                polarity = row[sentiment_col]

                if pd.isna(aspect) and pd.isna(category) and pd.isna(polarity):
                    continue

                annotation = {
                    "aspect_term": None if pd.isna(aspect) else str(aspect),
                    "aspect_category": None if pd.isna(category) else str(category),
                    "polarity": None if pd.isna(polarity) else str(polarity),
                }
                if has_span_columns:
                    annotation["from"] = None if pd.isna(row[from_col]) else int(row[from_col])
                    annotation["to"] = None if pd.isna(row[to_col]) else int(row[to_col])

                annotations.append(annotation)

            records.append(
                {
                    "id": uuid.uuid4(),
                    "source": SOURCE_AWARE,
                    "app_id": str(app_name),
                    "review_text": str(sentence),
                    "rating": None,
                    "review_date": None,
                    "raw_metadata": {"domain": str(domain), RAW_METADATA_AWARE: annotations},
                }
            )

        logging.info(f"Rows processed: {len(df):,}")
        logging.info(f"Reviews created: {len(records):,}")
        logging.info("Transformation completed successfully")
        return records
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def ingest_aware_csv(
    csv_path: str | Path = RAW_DATA_FILE,
    config: Optional[AWAREIngestionConfig] = None,
    batch_size: Optional[int] = None,
) -> int:
    """
    Ingest AWARE CSV rows into the raw_reviews table.
    """
    try:
        if config is None:
            config = ConfigurationManager().get_aware_ingestion_config()
        effective_batch_size = batch_size or config.batch_size

        dataframe = extract(config=config, csv_path=csv_path)
        records = transform_to_raw_reviews(dataframe, config)

        inserted = 0
        with get_db() as database:
            logging.info("Inserting raw_reviews records")
            for start in range(0, len(records), effective_batch_size):
                batch = records[start : start + effective_batch_size]
                database.bulk_insert_mappings(RawReview, batch)
                inserted += len(batch)
                logging.info(f"Inserted batch: {inserted}/{len(records)} raw_reviews rows")

        logging.info(f"AWARE ingestion complete: {inserted} rows inserted into raw_reviews")
        return inserted
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc