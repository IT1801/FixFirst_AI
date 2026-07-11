"""Extractor for gold labels from AWARE format to simulate labeling progress."""

import sys
from typing import Dict, List, Optional
import pandas as pd

from fixfirst.config.configuration import ConfigurationManager
from fixfirst.constants import RAW_METADATA_AWARE, EXTRACTED_LABELS_FILENAME, EXTRACTED_PROGRESS_FILENAME
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

class GoldLabelExtractor:
    """Object-oriented extractor for gold labels."""

    def __init__(self, limit: Optional[int] = None):
        self.config_manager = ConfigurationManager()
        self.settings = self.config_manager.get_settings()
        self.limit = limit

    def run(self):
        try:
            train_path = self.settings.resolve_path(self.settings.data_processed_dir) / "train.parquet"
            if not train_path.exists():
                raise FixFirstException("train.parquet not found. Run preprocessing first.", sys)

            df = pd.read_parquet(train_path)
            if self.limit:
                df = df.head(self.limit)

            from fixfirst.core._db.base import get_db
            from fixfirst.core._db.models import FeatureMaster
            with get_db() as db:
                taxonomy = db.query(FeatureMaster.feature_key, FeatureMaster.display_name).filter(FeatureMaster.is_active.is_(True)).all()
                feature_names = {row.feature_key: row.display_name for row in taxonomy}

            records = []
            valid_sentiments = {"positive", "negative", "neutral"}

            for _, row in df.iterrows():
                metadata = row.get("raw_metadata", {})
                annotations = metadata.get(RAW_METADATA_AWARE, [])

                for ann in annotations:
                    cat = ann.get("aspect_category")
                    pol = ann.get("polarity")

                    if isinstance(pol, str):
                        pol_lower = pol.lower()
                    else:
                        continue

                    if cat in feature_names and pol_lower in valid_sentiments:
                        records.append({
                            "review_id": row["id"],
                            "feature_key": cat,
                            "sentiment": pol_lower,
                            "confidence": 1.0,
                            "source": "gold",
                            "review_text": row["review_text"],
                        })

            labels_df = pd.DataFrame(records)
            if not labels_df.empty:
                progress_df = labels_df[["review_id", "review_text"]].drop_duplicates().copy()
                progress_df["status"] = "labeled"
            else:
                labels_df = pd.DataFrame(columns=["review_id", "feature_key", "sentiment", "confidence", "source", "review_text"])
                progress_df = pd.DataFrame(columns=["review_id", "review_text", "status"])

            out_dir = self.settings.resolve_path(self.settings.data_extracted_labels_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            labels_df.to_parquet(out_dir / EXTRACTED_LABELS_FILENAME)
            progress_df.to_parquet(out_dir / EXTRACTED_PROGRESS_FILENAME)
            
            logging.info(f"GoldLabelExtractor: Extracted {len(labels_df)} labels from {len(progress_df)} reviews.")
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc
