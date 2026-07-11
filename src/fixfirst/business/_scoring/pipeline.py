"""Scoring pipeline."""

import sys
import pandas as pd

from fixfirst.business._scoring.aggregation import build_windowed_scores
from fixfirst.constants import DEFAULT_HALF_LIFE_DAYS
from fixfirst.core.db import CriticalityScore, RawReview, ReviewAspect, get_db
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


class CriticalityScorer:
    """Object-oriented scorer for business criticality metrics."""

    def __init__(self, half_life_days: int = DEFAULT_HALF_LIFE_DAYS):
        self.half_life_days = half_life_days

    def load_aspects_with_dates(self) -> pd.DataFrame:
        """Loads review_aspects joined with raw_reviews.review_date."""
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
            logging.info(f"CriticalityScorer: loaded {len(df)} review_aspects rows joined with review_date")
            return df
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def write_criticality_scores(self, scores_df: pd.DataFrame) -> int:
        """Replaces the contents of criticality_scores with the newly computed scores."""
        try:
            rows = []
            for _, row in scores_df.iterrows():
                if row["window_start"] is None:
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

            logging.info(f"CriticalityScorer: wrote {len(rows)} rows to criticality_scores")
            return len(rows)
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def run(self) -> int:
        """Run the full scoring pipeline."""
        try:
            aspects_df = self.load_aspects_with_dates()
            if aspects_df.empty:
                raise FixFirstException(
                    "review_aspects is empty — run scripts/run_hybrid_inference.py first.", sys
                )

            scores_df = build_windowed_scores(aspects_df, half_life_days=self.half_life_days)
            n_written = self.write_criticality_scores(scores_df)

            n_undated = (scores_df["window_start"].isna()).sum()
            if n_undated:
                logging.warning(
                    f"CriticalityScorer: {n_undated} undated-bucket score row(s) were computed but "
                    f"NOT written to criticality_scores (no calendar window to store them under)."
                )

            return n_written
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc


def run_scoring_pipeline(half_life_days: int = DEFAULT_HALF_LIFE_DAYS) -> int:
    """Backward compatibility wrapper."""
    return CriticalityScorer(half_life_days=half_life_days).run()

def load_aspects_with_dates() -> pd.DataFrame:
    """Backward compatibility wrapper."""
    return CriticalityScorer().load_aspects_with_dates()

def write_criticality_scores(scores_df: pd.DataFrame) -> int:
    """Backward compatibility wrapper."""
    return CriticalityScorer().write_criticality_scores(scores_df)