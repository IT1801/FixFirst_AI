"""
AWARE dataset loader for FixFirst AI.

AWARE (Dabrowski et al.) is a benchmark of ~11,323 app review SENTENCES
across three domains — Productivity, Social Networking, and Games — each
annotated with aspect term(s), aspect category, and sentiment polarity.

IMPORTANT: AWARE is sentence-level and a single review/sentence may carry
MULTIPLE aspect annotations (one row per aspect mention in some mirrors of
the dataset). We group by the source sentence so each unique sentence
becomes exactly one `raw_reviews` row, with all of its aspect annotations
preserved in `raw_metadata.aware_annotations` for later use as a gold
evaluation set in Phase 4 (rather than discarding these free labels).

COLUMN MAPPING: download mirrors of AWARE are not perfectly consistent in
header naming. Before running this loader, inspect your actual file with
`pandas.read_csv(path).columns` and adjust AWARE_COLUMN_MAP below to match.
The keys (right-hand side, our names) must stay fixed; only the values
(left-hand side, source column names) should change.
"""

import sys
import uuid
from typing import Dict, List, Optional

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

# --- EDIT THIS to match the actual downloaded AWARE CSV headers -----------
AWARE_COLUMN_MAP: Dict[str, str] = {
    "sentence_id": "sentence_id",
    "domain": "domain",
    "app_name": "app",
    "sentence": "sentence",
    "aspect_term": "term",
    "aspect_category": "category",
    "polarity": "sentiment",
}
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = list(AWARE_COLUMN_MAP.values())


def load_aware_dataframe(csv_path: str) -> pd.DataFrame:
    """Reads the raw AWARE CSV and validates expected columns are present."""
    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
    except Exception as e:
        raise FixFirstException(e, sys)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise FixFirstException(
            f"AWARE CSV at {csv_path} is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Update AWARE_COLUMN_MAP in aware_loader.py to match this file.",
            sys,
        )

    logging.info(f"Loaded AWARE CSV: {len(df)} rows from {csv_path}")
    return df


def transform_to_raw_reviews(df: pd.DataFrame) -> List[dict]:
    """
    Groups AWARE rows by unique sentence (per app + domain) and produces one
    raw_reviews-shaped dict per sentence, with all aspect annotations
    preserved in raw_metadata for later gold-eval use.

    Returns a list of dicts matching RawReview constructor kwargs:
        {id, source, app_id, review_text, rating, review_date, raw_metadata}
    """
    try:
        col = AWARE_COLUMN_MAP
        group_keys = [col["app_name"], col["domain"], col["sentence"]]

        records: List[dict] = []

        for (app_name, domain, sentence), group in df.groupby(group_keys, dropna=False):
            annotations = []

            for _, row in group.iterrows():
                aspect = row[col["aspect_term"]]
                category = row[col["aspect_category"]]
                polarity = row[col["polarity"]]

                # Skip rows where there is no annotation at all
                if pd.isna(aspect) and pd.isna(category) and pd.isna(polarity):
                    continue

                annotations.append(
                    {
                        "aspect_term": None if pd.isna(aspect) else str(aspect),
                        "aspect_category": None if pd.isna(category) else str(category),
                        "polarity": None if pd.isna(polarity) else str(polarity),
                    }
                )

            records.append(
                {
                    "id": uuid.uuid4(),
                    "source": "aware",
                    "app_id": str(app_name),
                    "review_text": str(sentence),
                    "rating": None,       # AWARE is sentence-level
                    "review_date": None,  # No review dates in AWARE
                    "raw_metadata": {
                        "domain": str(domain),
                        "aware_annotations": annotations,
                    },
                }
            )

        logging.info(
            f"Transformed {len(df)} AWARE rows into {len(records)} unique "
            f"raw_reviews records (grouped by sentence)."
        )
        return records

    except Exception as e:
        raise FixFirstException(e, sys)


def ingest_aware_csv(csv_path: str, batch_size: int = 500) -> int:
    """
    Full ingestion: load CSV -> transform -> bulk insert into raw_reviews.
    Returns the number of records inserted.
    """
    from fixfirst.db.base import get_db
    from fixfirst.db.models import RawReview

    try:
        df = load_aware_dataframe(csv_path)
        records = transform_to_raw_reviews(df)

        inserted = 0
        with get_db() as db:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                db.bulk_insert_mappings(RawReview, batch)
                inserted += len(batch)
                logging.info(f"Inserted batch: {inserted}/{len(records)} raw_reviews rows")

        logging.info(f"AWARE ingestion complete: {inserted} rows inserted into raw_reviews.")
        return inserted
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)