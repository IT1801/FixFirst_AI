"""
Scoring pipeline: reads review_aspects (joined with raw_reviews for
review_date) from Postgres, computes windowed criticality scores, and
writes them into criticality_scores.

Usage:
    PYTHONPATH=src python scripts/run_scoring.py
"""

import sys

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.scoring.aggregation import build_windowed_scores
from fixfirst.scoring.criticality import DEFAULT_HALF_LIFE_DAYS


def load_aspects_with_dates() -> pd.DataFrame:
    """Loads review_aspects joined with raw_reviews.review_date."""
    from fixfirst.db.base import get_db
    from fixfirst.db.models import ReviewAspect, RawReview

    try:
        with get_db() as db:
            rows = (
                db.query(
                    ReviewAspect.feature_id,
                    ReviewAspect.sentiment,
                    RawReview.review_date,
                )
                .join(RawReview, ReviewAspect.review_id == RawReview.id)
                .all()
            )

        df = pd.DataFrame(
            [{"feature_id": r.feature_id, "sentiment": r.sentiment.value, "review_date": r.review_date} for r in rows]
        )
        logging.info(f"load_aspects_with_dates: loaded {len(df)} review_aspects rows joined with review_date")
        return df
    except Exception as e:
        raise FixFirstException(e, sys)


def write_criticality_scores(scores_df: pd.DataFrame) -> int:
    """Replaces the contents of criticality_scores with the newly computed
    scores. Full replace (not incremental upsert) — this pipeline is
    intended to be re-run in full each time (e.g. via the Prefect flow in
    Phase 6), which is simpler and less error-prone than reconciling
    partial updates for a scoring table this size."""
    from fixfirst.db.base import get_db
    from fixfirst.db.models import CriticalityScore

    try:
        rows = []
        for _, row in scores_df.iterrows():
            if row["window_start"] is None:
                # Undated bucket has no calendar window; criticality_scores
                # requires non-null window_start/window_end (NOT NULL columns),
                # so we skip persisting undated-bucket rows to the DB table —
                # they're still visible via the pipeline's return value/logs,
                # just not written to a table whose schema assumes a real window.
                continue

            rows.append(
                {
                    "feature_id": row["feature_id"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "score": row["score"],
                    "mention_count": row["mention_count"],
                    "negative_ratio": row["negative_ratio"],
                    "trend_delta": row["trend_delta"],
                }
            )

        with get_db() as db:
            db.query(CriticalityScore).delete()
            db.bulk_insert_mappings(CriticalityScore, rows)

        logging.info(f"write_criticality_scores: wrote {len(rows)} rows to criticality_scores")
        return len(rows)
    except Exception as e:
        raise FixFirstException(e, sys)


def run_scoring_pipeline(half_life_days: int = DEFAULT_HALF_LIFE_DAYS) -> int:
    try:
        aspects_df = load_aspects_with_dates()
        if aspects_df.empty:
            raise FixFirstException(
                "review_aspects is empty — run scripts/run_hybrid_inference.py first.", sys
            )

        scores_df = build_windowed_scores(aspects_df, half_life_days=half_life_days)
        n_written = write_criticality_scores(scores_df)

        n_undated = (scores_df["window_start"].isna()).sum()
        if n_undated:
            logging.warning(
                f"run_scoring_pipeline: {n_undated} undated-bucket score row(s) were computed but "
                f"NOT written to criticality_scores (no calendar window to store them under)."
            )

        return n_written
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)