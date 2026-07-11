"""
Windowed aggregation for criticality scoring.

Buckets review_aspects rows into calendar-month windows per feature, runs
compute_criticality_score per (feature, window) group, and computes
trend_delta = this window's score minus the immediately preceding dated
window's score for the same feature (null for a feature's first dated
window, or for the undated bucket, which has no "previous" in time).

Rows with no review_date (e.g. AWARE, which carries no dates — see
ingestion/aware_loader.py) are bucketed separately under window_key=None
("undated") rather than forced into an arbitrary calendar window. This
bucket gets a score like any other, just no trend_delta, since trend
analysis is inherently about change over time and undated data has no
time axis to trend along.
"""

import sys
from datetime import date
from typing import Optional

import pandas as pd

from fixfirst.business._scoring.criticality import compute_criticality_score
from fixfirst.constants import DEFAULT_HALF_LIFE_DAYS
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

REQUIRED_COLUMNS = {"feature_id", "sentiment", "review_date"}


def _month_window(d: Optional[date]) -> Optional[tuple]:
    """Returns (window_start, window_end) for the calendar month containing
    d, or None if d is None (undated bucket)."""
    try:
        if d is None:
            return None

        window_start = date(d.year, d.month, 1)
        if d.month == 12:
            next_month_start = date(d.year + 1, 1, 1)
        else:
            next_month_start = date(d.year, d.month + 1, 1)

        window_end = next_month_start - pd.Timedelta(days=1)
        return (window_start, window_end.date() if hasattr(window_end, "date") else window_end)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def build_windowed_scores(
    aspects_df: pd.DataFrame, half_life_days: int = DEFAULT_HALF_LIFE_DAYS
) -> pd.DataFrame:
    """
    aspects_df must have columns: feature_id, sentiment, review_date
    (review_date may contain None/NaT values).

    Returns a DataFrame with columns:
        feature_id, window_start, window_end, score, mention_count,
        negative_ratio, undated_ratio, trend_delta
    One row per (feature_id, window) — the undated bucket, if present, has
    window_start=window_end=None and trend_delta=None.
    """
    try:
        missing = REQUIRED_COLUMNS - set(aspects_df.columns)
        if missing:
            raise FixFirstException(f"aspects_df missing required columns: {missing}", sys)

        if aspects_df.empty:
            raise FixFirstException("aspects_df is empty — nothing to score.", sys)

        df = aspects_df.copy()
        is_dated = df["review_date"].notna()

        rows = []

        # --- Dated rows: bucket into (feature_id, calendar_month) groups ---
        dated_df = df[is_dated].copy()
        if not dated_df.empty:
            windows = dated_df["review_date"].apply(_month_window)
            dated_df["_window_start"] = windows.apply(lambda w: w[0])
            dated_df["_window_end"] = windows.apply(lambda w: w[1])

            for (feature_id, window_start, window_end), group in dated_df.groupby(
                ["feature_id", "_window_start", "_window_end"]
            ):
                metrics = compute_criticality_score(
                    group["sentiment"].tolist(),
                    group["review_date"].tolist(),
                    window_end,
                    half_life_days=half_life_days,
                )
                rows.append(
                    {
                        "feature_id": feature_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        **metrics,
                    }
                )

        # --- Undated rows: one bucket per feature_id, window_start/end = None ---
        undated_df = df[~is_dated]
        if not undated_df.empty:
            for feature_id, group in undated_df.groupby("feature_id"):
                metrics = compute_criticality_score(
                    group["sentiment"].tolist(),
                    group["review_date"].tolist(),
                    date.today(),  # harmless placeholder; every date is None -> recency_weight=1.0 regardless
                    half_life_days=half_life_days,
                )
                rows.append(
                    {
                        "feature_id": feature_id,
                        "window_start": None,
                        "window_end": None,
                        **metrics,
                    }
                )

        scores_df = pd.DataFrame(rows)
        scores_df["trend_delta"] = None

        for feature_id, group in scores_df.groupby("feature_id"):
            dated = group[group["window_start"].notna()].sort_values("window_start")
            prev_score = None
            for idx in dated.index:
                current_score = scores_df.loc[idx, "score"]
                if prev_score is not None:
                    scores_df.loc[idx, "trend_delta"] = current_score - prev_score
                prev_score = current_score

        n_undated_features = scores_df[scores_df["window_start"].isna()]["feature_id"].nunique()
        if n_undated_features:
            logging.info(
                f"build_windowed_scores: {n_undated_features} feature(s) have an undated bucket "
                f"(no trend_delta available for that portion of their data)"
            )

        logging.info(f"build_windowed_scores: computed {len(scores_df)} (feature, window) score rows")
        return scores_df
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc