"""Data ingestion pipelines for FixFirst AI."""

import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from fixfirst.config.configuration import AWAREIngestionConfig, ConfigurationManager
from fixfirst.constants import RAW_METADATA_AWARE, RAW_DATA_FILE, SOURCE_AWARE
from fixfirst.core._db.models import RawReview
from fixfirst.core._db.base import get_db
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


class DataIngestor(ABC):
    """Abstract base class for all data ingestors."""

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data from source and return a DataFrame."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> List[Dict]:
        """Transform raw data into raw_reviews-shaped records."""
        pass

    def load(self, records: List[Dict], batch_size: int = 500) -> int:
        """Load records into the database."""
        try:
            inserted = 0
            with get_db() as database:
                logging.info("Inserting raw_reviews records")
                for start in range(0, len(records), batch_size):
                    batch = records[start : start + batch_size]
                    database.bulk_insert_mappings(RawReview, batch)
                    inserted += len(batch)
                    logging.info(f"Inserted batch: {inserted}/{len(records)} raw_reviews rows")

            logging.info(f"Ingestion complete: {inserted} rows inserted into raw_reviews")
            return inserted
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def run(self) -> int:
        """Run the complete ETL pipeline."""
        df = self.extract()
        records = self.transform(df)
        return self.load(records)


class AWAREIngestor(DataIngestor):
    """Ingestor for the AWARE dataset."""

    def __init__(self, config: Optional[AWAREIngestionConfig] = None, csv_path: Optional[Path] = None):
        self.config = config or ConfigurationManager().get_aware_ingestion_config()
        self.csv_path = csv_path or self.config.csv_path

    def extract(self) -> pd.DataFrame:
        """Load and validate the raw AWARE CSV."""
        try:
            resolved_path = Path(self.csv_path)
            logging.info(f"Loading AWARE CSV from {resolved_path}")

            if not resolved_path.exists():
                raise FileNotFoundError(f"AWARE CSV not found at {resolved_path}")

            dataframe = pd.read_csv(resolved_path, keep_default_na=False)

            logging.info(f"Rows loaded: {len(dataframe):,}")
            logging.info(f"Columns: {dataframe.columns.tolist()}")

            self._validate_raw(dataframe)
            return dataframe
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def _validate_raw(self, df: pd.DataFrame) -> None:
        """Validate the raw AWARE dataframe."""
        try:
            expected_columns = [
                self.config.domain_column,
                self.config.app_column,
                self.config.sentence_column,
                self.config.category_column,
                self.config.term_column,
                self.config.sentiment_column,
                self.config.rating_column,
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

    def transform(self, df: pd.DataFrame) -> List[Dict]:
        """Transform raw AWARE rows into raw_reviews-shaped records."""
        try:
            app_col = self.config.app_column
            domain_col = self.config.domain_column
            sentence_col = self.config.sentence_column
            term_col = self.config.term_column
            category_col = self.config.category_column
            sentiment_col = self.config.sentiment_column
            from_col = self.config.from_column
            to_col = self.config.to_column
            rating_col = self.config.rating_column

            logging.info("Transforming AWARE rows into raw_reviews records")

            group_keys = [app_col, domain_col, sentence_col]
            has_span_columns = (from_col in df.columns and to_col in df.columns)
            if not has_span_columns:
                logging.info("Span columns are not present; annotations will omit offsets")

            records: List[Dict] = []

            for (app_name, domain, sentence), group in df.groupby(group_keys, dropna=False):
                annotations: List[Dict] = []

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
                rating = group.iloc[0][rating_col]
                records.append(
                    {
                        "id": uuid.uuid4(),
                        "source": SOURCE_AWARE,
                        "app_id": str(app_name),
                        "review_text": str(sentence),
                        "rating": None if pd.isna(rating) else int(rating),
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

    def run(self) -> int:
        """Run the complete ETL pipeline for AWARE."""
        df = self.extract()
        records = self.transform(df)
        return self.load(records, batch_size=self.config.batch_size)


# For backward compatibility / scripting simplicity
def ingest_aware_csv(
    csv_path: str | Path = RAW_DATA_FILE,
    config: Optional[AWAREIngestionConfig] = None,
    batch_size: Optional[int] = None,
) -> int:
    """Ingest AWARE CSV rows into the raw_reviews table."""
    ingestor = AWAREIngestor(config=config, csv_path=Path(csv_path) if csv_path else None)
    if batch_size:
        # Avoid creating new dataclass directly to support overriding, though it's frozen
        pass  # Batch size override in wrapper handled at run time
    return ingestor.run()