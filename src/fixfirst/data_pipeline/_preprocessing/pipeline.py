"""
Preprocessing pipeline for FixFirst AI.

Orchestrates: load raw_reviews from Postgres -> clean -> deduplicate ->
filter to English -> split into train/val/test -> write Parquet.
"""

import json
import sys
from pathlib import Path
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

    def _build_training_format(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        Convert the three parquet splits into three JSONL files per split:
          - master.jsonl          : review_id + text + aspects list
          - aspect_category.jsonl : text + multi-label category list
          - aspect_sentiment.jsonl: aspect + text + sentiment label (one row per aspect)

        Written to  <data_training_format_dir>/<split>/  so every downstream
        trainer can load data without touching the database or extracted labels.
        """
        LABEL_VOCAB = sorted([
            "aesthetics", "compatibility", "cost", "effectiveness",
            "efficiency", "enjoyability", "general", "learnability",
            "reliability", "safety", "security", "usability",
        ])
        LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABEL_VOCAB)}
        INVALID = {"N/A", "NA", "", None}

        def _parse_aspects(raw_metadata):
            annots = raw_metadata.get("aware_annotations", []) if isinstance(raw_metadata, dict) else []
            seen, aspects = set(), []
            for a in annots:
                if not isinstance(a, dict):
                    continue
                cat = a.get("aspect_category")
                pol = a.get("polarity")
                if cat in INVALID or cat not in LABEL_TO_IDX:
                    continue
                pol = None if pol in INVALID else pol
                key = (cat, pol)
                if key not in seen:
                    seen.add(key)
                    aspects.append({"category": cat, "sentiment": pol})
            return aspects

        out_root = self.settings.resolve_path(self.settings.data_training_format_dir)

        for split_name, df in splits.items():
            split_dir = out_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            master_recs, cat_recs, sent_recs = [], [], []
            for _, row in df.iterrows():
                text = str(row["review_text"]).strip()
                rid  = str(row["id"])
                if not text:
                    continue
                aspects = _parse_aspects(row["raw_metadata"])
                master_recs.append({"review_id": rid, "text": text, "aspects": aspects})
                labels = sorted({a["category"] for a in aspects})
                cat_recs.append({"review_id": rid, "text": text, "labels": labels})
                for asp in aspects:
                    if asp["sentiment"] is None:
                        continue
                    sent_recs.append({"review_id": rid, "aspect": asp["category"],
                                      "text": text, "label": asp["sentiment"]})

            for filename, records in [
                ("master.jsonl",           master_recs),
                ("aspect_category.jsonl",  cat_recs),
                ("aspect_sentiment.jsonl", sent_recs),
            ]:
                path = split_dir / filename
                with path.open("w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logging.info(f"_build_training_format: wrote {len(records):,} records → {path}")

        # Write shared vocab
        vocab_path = out_root / "label_vocab.json"
        with vocab_path.open("w") as f:
            json.dump({"label_vocab": LABEL_VOCAB, "label_to_idx": LABEL_TO_IDX}, f, indent=2)
        logging.info(f"_build_training_format: vocab written → {vocab_path}")

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

            splits = {"train": train_df, "val": val_df, "test": test_df}

            if write_output:
                out_dir = self.settings.resolve_path(self.settings.data_processed_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                train_df.to_parquet(out_dir / TRAIN_FILENAME, index=False)
                val_df.to_parquet(out_dir / VAL_FILENAME, index=False)
                test_df.to_parquet(out_dir / TEST_FILENAME, index=False)
                logging.info(f"run: wrote train/val/test Parquet files to {out_dir}")

                # Build the three JSONL training-format files from the fresh splits
                self._build_training_format(splits)

            return splits
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e


# Backward compatibility wrapper
def run_preprocessing_pipeline(write_output: bool = True) -> Dict[str, pd.DataFrame]:
    """Legacy function for backward compatibility."""
    return PreprocessingPipeline().run(write_output=write_output)