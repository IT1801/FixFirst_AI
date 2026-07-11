"""Helpers for constructing gold-evaluation labels from AWARE reviews."""

import sys
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._evaluation.category_mapping import map_aware_category, report_mapping_coverage
from fixfirst.ml._training.common import build_label_index


def _normalize_raw_metadata(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        return {}
    if isinstance(value, (list, tuple, np.ndarray)):
        return {}
    if pd.isna(value):
        return {}
    return dict(value)


def _normalize_annotations(value) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, str):
        return []
    if pd.isna(value):
        return []
    return list(value)


def _iter_aware_annotations(row: pd.Series) -> Iterable[dict]:
    raw_metadata = _normalize_raw_metadata(row.get("raw_metadata"))
    annotations = _normalize_annotations(raw_metadata.get("aware_annotations"))
    for annotation in annotations:
        if isinstance(annotation, dict):
            yield annotation


def extract_gold_category_labels(test_df: pd.DataFrame, feature_keys: List[str]) -> pd.DataFrame:
    """Build one gold-label row per review for the category evaluator."""
    try:
        if test_df.empty:
            raise FixFirstException("test_df is empty — no gold labels available for evaluation.", sys)

        feature_index = build_label_index(feature_keys)
        raw_categories = set()
        rows = []

        for row in test_df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            review_text = row_series.get("review_text")
            review_id = row_series.get("id")
            gold_labels = np.zeros(len(feature_index), dtype=np.int64)

            for annotation in _iter_aware_annotations(row_series):
                raw_category = annotation.get("aspect_category")
                raw_categories.add(str(raw_category) if raw_category is not None else "")
                mapped_category = map_aware_category(raw_category)
                if mapped_category is None or mapped_category not in feature_index:
                    continue
                gold_labels[feature_index[mapped_category]] = 1

            rows.append(
                {
                    "review_id": review_id,
                    "review_text": review_text,
                    "gold_labels": gold_labels,
                }
            )

        report_mapping_coverage({category for category in raw_categories if category})
        return pd.DataFrame(rows)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def extract_gold_sentiment_pairs(test_df: pd.DataFrame, feature_display_names: Dict[str, str]) -> pd.DataFrame:
    """Build one sentiment-classification row per mapped AWARE annotation."""
    try:
        if test_df.empty:
            raise FixFirstException("test_df is empty — no gold labels available for evaluation.", sys)

        rows = []
        invalid_sentiments = set()

        for row in test_df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            review_text = row_series.get("review_text")
            review_id = row_series.get("id")

            for annotation in _iter_aware_annotations(row_series):
                raw_category = annotation.get("aspect_category")
                feature_key = map_aware_category(raw_category)
                if feature_key is None:
                    continue

                if feature_key not in feature_display_names:
                    logging.warning(
                        f"extract_gold_sentiment_pairs: feature_key {feature_key!r} has no display name mapping; skipping"
                    )
                    continue

                sentiment = annotation.get("polarity")
                if sentiment is None:
                    continue

                normalized_sentiment = str(sentiment).strip().lower()
                if normalized_sentiment in {"n/a", "na", "none"}:
                    normalized_sentiment = "neutral"
                    
                if normalized_sentiment not in {"negative", "neutral", "positive"}:
                    invalid_sentiments.add(normalized_sentiment)
                    continue

                rows.append(
                    {
                        "review_id": review_id,
                        "feature_key": feature_key,
                        "text_a": review_text,
                        "text_b": feature_display_names[feature_key],
                        "gold_sentiment": normalized_sentiment,
                    }
                )

        if invalid_sentiments:
            logging.warning(
                f"extract_gold_sentiment_pairs: dropped invalid sentiment values {sorted(invalid_sentiments)}"
            )

        if not rows:
            raise FixFirstException(
                "No valid gold sentiment pairs were found after filtering unmapped categories and invalid sentiments.",
                sys,
            )

        return pd.DataFrame(rows)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc