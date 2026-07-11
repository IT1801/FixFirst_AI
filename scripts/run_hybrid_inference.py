"""Batch hybrid inference pipeline."""

import argparse
import sys
import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._inference.pipeline import HybridInferencePipeline
from fixfirst.logging.logger import logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch hybrid ABSA inference.")
    parser.add_argument("--limit", type=int, default=2500, help="Max reviews to process")
    parser.add_argument("--split", type=str, help="Dataset split to evaluate (e.g., test)")
    parser.add_argument("--no-db", action="store_true", help="Skip writing results to the database")
    args = parser.parse_args()

    try:
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import RawReview

        logging.info(f"Starting batch inference (limit={args.limit})...")
        
        with get_db() as db:
            reviews = db.query(RawReview.id, RawReview.review_text).limit(args.limit).all()
            
        if not reviews:
            logging.info("No reviews found in the database.")
            return 0

        reviews_df = pd.DataFrame(reviews, columns=["id", "review_text"])
        
        pipeline = HybridInferencePipeline(write_to_db=not args.no_db)
        stats = pipeline.run(reviews_df)
        
        print("\n" + "="*40)
        print("🚀 BATCH INFERENCE COMPLETE")
        print("="*40)
        print(f"Reviews Processed: {stats['n_reviews_processed']}")
        print(f"Aspects Written:   {stats['n_aspects_written']}")
        print(f"LLM Fallback Rate: {stats['fallback_stats']['llm_fallback_rate']:.1%}")
        print("="*40 + "\n")
        return 0
        
    except FixFirstException as exc:
        logging.error(f"Fatal error during batch inference: {exc}")
        return 1
    except Exception as exc:
        logging.error(f"Fatal error during batch inference: {exc}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
