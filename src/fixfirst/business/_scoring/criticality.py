"""
Criticality score formula for FixFirst AI.

    score = negative_ratio * log(1 + mention_count) * mean(recency_weight)

Where:
  negative_ratio   = negative_mentions / total_mentions in the window.
                      The core "how bad is it" signal.
  log(1 + mention_count) = frequency term, log-dampened so a feature with
                      500 mentions doesn't linearly dominate one with 50 —
                      both are "worth developer attention," but 10x the
                      volume shouldn't mean 10x the score.
  recency_weight    = exponential decay per review: 0.5 ** (age_days / HALF_LIFE_DAYS).
                      A review from today counts fully; one from 6 months
                      ago (with the default 90-day half life) counts for
                      ~6%. Reviews with no review_date (e.g. AWARE, which
                      carries no dates) get weight=1.0 — treated as
                      "always current" rather than penalized for missing
                      metadata, and this is logged so it's visible in
                      aggregate stats rather than silently skewing scores.

HONEST LIMITATION: the original design sketch included a `severity_weight`
term (crash > "could be better") derived from intensity classification.
That intensity signal was never built — our sentiment scheme is 3-class
polarity only (negative/neutral/positive), with no severity/intensity
label. Rather than fake that term, it's omitted here. A real severity
signal (e.g. an LLM-tagged intensity score, or using AspectSource/
confidence as a rough proxy) is a documented extension point, not a
silently-dropped promise.
"""

import sys
from datetime import date
from typing import List, Optional

import numpy as np

from fixfirst.constants import DEFAULT_HALF_LIFE_DAYS
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def compute_negative_ratio(sentiments: List[str]) -> float:
    """Return the fraction of negative sentiments in a bucket."""
    try:
        if not sentiments:
            return 0.0

        negative_count = sum(1 for sentiment in sentiments if sentiment == "negative")
        return negative_count / len(sentiments)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def compute_recency_weight(
    review_date: Optional[date], window_end: date, half_life_days: int = DEFAULT_HALF_LIFE_DAYS
) -> float:
    """
    Returns 1.0 for reviews with no date (undated sources like AWARE are
    treated as always-current rather than decayed toward zero). Otherwise
    exponential decay based on age relative to window_end.
    """
    try:
        if review_date is None:
            return 1.0

        age_days = (window_end - review_date).days
        if age_days < 0:
            age_days = 0

        return float(0.5 ** (age_days / half_life_days))
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def compute_criticality_score(
    sentiments: List[str],
    review_dates: List[Optional[date]],
    window_end: date,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
) -> dict:
    """
    Computes {score, mention_count, negative_ratio, undated_ratio} for one
    (feature, window) group. sentiments and review_dates must be
    parallel lists (same length, same order).
    """
    try:
        if len(sentiments) != len(review_dates):
            raise FixFirstException(
                f"sentiments and review_dates must be the same length, got "
                f"{len(sentiments)} and {len(review_dates)}",
                sys,
            )

        mention_count = len(sentiments)
        if mention_count == 0:
            return {"score": 0.0, "mention_count": 0, "negative_ratio": 0.0, "undated_ratio": 0.0}

        negative_ratio = compute_negative_ratio(sentiments)
        recency_weights = [compute_recency_weight(d, window_end, half_life_days) for d in review_dates]
        mean_recency_weight = float(np.mean(recency_weights))
        undated_ratio = sum(1 for d in review_dates if d is None) / mention_count

        score = negative_ratio * np.log1p(mention_count) * mean_recency_weight

        if undated_ratio > 0.5:
            logging.warning(
                f"compute_criticality_score: {undated_ratio:.0%} of {mention_count} mentions in this "
                f"window have no review_date — recency weighting is effectively disabled for most of "
                f"this feature's score."
            )

        return {
            "score": float(score),
            "mention_count": mention_count,
            "negative_ratio": float(negative_ratio),
            "undated_ratio": float(undated_ratio),
        }
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc