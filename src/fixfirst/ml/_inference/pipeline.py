"""
Batch hybrid inference pipeline.

Runs route_review() over a set of reviews, writes results into the real
review_aspects table (unlike Phase 3's silver labeling, which only wrote
to Parquet for QA before promotion — this IS the production write path),
and reports the fallback-rate metric: what fraction of inferences needed
the LLM vs. the fine-tuned model, the number that headlines the README.

Usage:
    PYTHONPATH=src python scripts/run_hybrid_inference.py [--limit N]
"""

import sys
from typing import Dict, List

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._inference.confidence import compute_fallback_rate_stats
from fixfirst.ml._inference.router import route_review
from fixfirst.logging.logger import logging


def run_batch_hybrid_inference(reviews_df: pd.DataFrame, write_to_db: bool = True) -> Dict:
    """
    reviews_df must have columns [id, review_text].

    Returns {"fallback_stats": {...}, "n_reviews_processed": int,
             "n_aspects_written": int}.
    """
    from fixfirst.ml._inference.model_inference import predict_category_probs, predict_sentiment_probs
    from fixfirst.ml._labeling.labeling import label_review
    from fixfirst.ml._labeling.taxonomy import load_active_taxonomy

    try:
        taxonomy = load_active_taxonomy()
        source_counts = {"finetuned": 0, "llm_fallback": 0}
        all_rows: List[Dict] = []

        total = len(reviews_df)
        for i, row in enumerate(reviews_df.itertuples(index=False), start=1):
            aspects = route_review(
                row.review_text,
                taxonomy,
                predict_category_probs,
                predict_sentiment_probs,
                label_review,
            )
            for aspect in aspects:
                source_counts[aspect["source"]] = source_counts.get(aspect["source"], 0) + 1
                all_rows.append(
                    {
                        "review_id": row.id,
                        "feature_key": aspect["feature_key"],
                        "sentiment": aspect["sentiment"],
                        "confidence": aspect["confidence"],
                        "source": aspect["source"],
                    }
                )

            if i % 25 == 0 or i == total:
                logging.info(f"run_batch_hybrid_inference: processed {i}/{total} reviews")

        fallback_stats = compute_fallback_rate_stats(source_counts)
        logging.info(
            f"run_batch_hybrid_inference: {fallback_stats['total']} aspects produced, "
            f"llm_fallback_rate={fallback_stats['llm_fallback_rate']:.1%}"
        )

        if write_to_db:
            n_written = _write_review_aspects(all_rows)
        else:
            n_written = 0

        return {
            "fallback_stats": fallback_stats,
            "n_reviews_processed": total,
            "n_aspects_written": n_written,
        }
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)


def _write_review_aspects(rows: List[Dict]) -> int:
    """
    Writes routed aspect predictions into review_aspects. LLM-fallback rows
    have confidence=None (no calibrated probability from the LLM) — stored
    as 1.0 in the DB (NOT NULL confidence column) with source clearly
    marked llm_fallback, so downstream scoring can distinguish "confidently
    finetuned" from "LLM-adjudicated" without needing a nullable column.
    """
    from fixfirst.core._db.base import get_db
    from fixfirst.core._db.models import ReviewAspect, SentimentLabel, AspectSource
    from fixfirst.ml._labeling.taxonomy import load_active_taxonomy

    try:
        taxonomy = load_active_taxonomy()
        feature_id_by_key = _load_feature_id_map(taxonomy)

        mapped_rows = []
        for row in rows:
            feature_id = feature_id_by_key.get(row["feature_key"])
            if feature_id is None:
                logging.warning(f"_write_review_aspects: unknown feature_key {row['feature_key']!r}, skipping")
                continue

            mapped_rows.append(
                {
                    "review_id": row["review_id"],
                    "feature_id": feature_id,
                    "sentiment": SentimentLabel(row["sentiment"]),
                    "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                    "source": AspectSource(row["source"]),
                }
            )

        with get_db() as db:
            db.bulk_insert_mappings(ReviewAspect, mapped_rows)

        logging.info(f"_write_review_aspects: wrote {len(mapped_rows)} rows to review_aspects")
        return len(mapped_rows)
    except Exception as e:
        raise FixFirstException(e, sys)


def _load_feature_id_map(taxonomy: List[Dict]) -> Dict[str, object]:
    from fixfirst.core._db.base import get_db
    from fixfirst.core._db.models import FeatureMaster

    with get_db() as db:
        rows = db.query(FeatureMaster.feature_key, FeatureMaster.id).all()
        return {key: fid for key, fid in rows}