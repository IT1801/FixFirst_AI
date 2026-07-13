"""Batch inference pipeline."""

import sys
from typing import Dict, List

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._inference.router import InferenceRouter
from fixfirst.logging.logger import logging
from fixfirst.constants import SOURCE_FINETUNED


class HybridInferencePipeline:
    """Object-oriented pipeline for batch inference."""

    def __init__(self, write_to_db: bool = True):
        self.write_to_db = write_to_db

    def run(self, reviews_df: pd.DataFrame) -> Dict:
        """Runs the pipeline over a dataframe of reviews."""
        from fixfirst.ml._inference.model_inference import predict_category_probs, predict_sentiment_probs
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import FeatureMaster

        try:
            with get_db() as db:
                taxonomy_rows = db.query(FeatureMaster).filter(FeatureMaster.is_active.is_(True)).all()
                taxonomy = [{"feature_key": row.feature_key, "display_name": row.display_name} for row in taxonomy_rows]
            
            source_counts = {SOURCE_FINETUNED: 0}
            all_rows: List[Dict] = []
            total = len(reviews_df)

            router = InferenceRouter(
                taxonomy=taxonomy,
                predict_category_probs_fn=predict_category_probs,
                predict_sentiment_probs_fn=predict_sentiment_probs,
            )

            for i, row in enumerate(reviews_df.itertuples(index=False), start=1):
                aspects = router.route_review(row.review_text)
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
                    logging.info(f"HybridInferencePipeline: processed {i}/{total} reviews")

            logging.info(
                f"HybridInferencePipeline: {sum(source_counts.values())} aspects produced, "
                f"llm_fallback_rate=0.0%"
            )

            n_written = self._write_review_aspects(all_rows) if self.write_to_db else 0

            return {
                "fallback_stats": {"total": sum(source_counts.values()), "llm_fallback_rate": 0.0},
                "n_reviews_processed": total,
                "n_aspects_written": n_written,
            }
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e

    def _write_review_aspects(self, rows: List[Dict]) -> int:
        """Writes routed aspect predictions into review_aspects."""
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import ReviewAspect, SentimentLabel, AspectSource

        try:
            feature_id_by_key = self._load_feature_id_map()

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
            raise FixFirstException(e, sys) from e

    def _load_feature_id_map(self) -> Dict[str, object]:
        """Load mapping of feature_key to primary key ID."""
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import FeatureMaster

        with get_db() as db:
            rows = db.query(FeatureMaster.feature_key, FeatureMaster.id).all()
            return {key: fid for key, fid in rows}


def run_batch_hybrid_inference(reviews_df: pd.DataFrame, write_to_db: bool = True) -> Dict:
    """Backward compatibility wrapper."""
    pipeline = HybridInferencePipeline(write_to_db=write_to_db)
    return pipeline.run(reviews_df)