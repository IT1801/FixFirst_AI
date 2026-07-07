"""
Silver-labeling orchestration for FixFirst AI.

"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import pandas as pd

from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._labeling.zero_shot import classify_review_batch

from fixfirst.constants import (
    LABELS_FILENAME,
    FAILURES_FILENAME,
    PROGRESS_FILENAME,
    LABEL_COLUMNS,
    FAILURE_COLUMNS,
    PROGRESS_COLUMNS,
    MAX_RETRIES,
    DEFAULT_BATCH_SIZE,
    RATE_LIMIT_SECONDS,
)


def _empty_dataframe(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_review_id(review_id) -> str:
    return str(review_id)


def _chunk_reviews(reviews: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [reviews[i : i + batch_size] for i in range(0, len(reviews), batch_size)]


def _load_parquet_if_exists(path: Path, columns: List[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return _empty_dataframe(columns)


def _collect_processed_ids(progress_df: pd.DataFrame, labels_df: pd.DataFrame, failures_df: pd.DataFrame) -> Set[str]:
    processed_ids = set()

    if not progress_df.empty and "review_id" in progress_df.columns:
        processed_ids.update(_normalize_review_id(review_id) for review_id in progress_df["review_id"].dropna())

    if not labels_df.empty and "review_id" in labels_df.columns:
        processed_ids.update(_normalize_review_id(review_id) for review_id in labels_df["review_id"].dropna())

    if not failures_df.empty and "review_id" in failures_df.columns:
        processed_ids.update(_normalize_review_id(review_id) for review_id in failures_df["review_id"].dropna())

    return processed_ids


def _write_checkpoint(
    out_dir: Path,
    labels_records: List[Dict],
    failures_records: List[Dict],
    progress_records: List[Dict],
) -> None:
    pd.DataFrame(labels_records, columns=LABEL_COLUMNS).to_parquet(out_dir / LABELS_FILENAME, index=False)
    pd.DataFrame(failures_records, columns=FAILURE_COLUMNS).to_parquet(out_dir / FAILURES_FILENAME, index=False)
    pd.DataFrame(progress_records, columns=PROGRESS_COLUMNS).to_parquet(out_dir / PROGRESS_FILENAME, index=False)


def _is_valid_for_absa(review_text: str, min_words: int = 4) -> bool:
    """
    Filters out reviews that are too short to contain meaningful aspect/sentiment context.
    """
    if not isinstance(review_text, str) or not review_text.strip():
        return False
    return len(review_text.split()) >= min_words


def _apply_heuristic_overrides(review_text: str, ai_category: str, ai_sentiment: str) -> Tuple[str, str]:
    """
    Overrides the AI's predictions if undeniable keyword triggers are found in the text.
    """
    text_lower = review_text.lower()
    
    final_category = ai_category
    final_sentiment = ai_sentiment

    # 1. Billing & Refunds
    billing_keywords = ["refund", "money back", "charged", "stole my money", "unauthorized", "billed", "subscription"]
    if any(kw in text_lower for kw in billing_keywords):
        final_category = "billing_subscription"

    # 2. Crashes & Stability
    crash_keywords = ["crash", "crashed", "crashes", "keeps closing", "force close", "won't open", "black screen"]
    if any(kw in text_lower for kw in crash_keywords):
        final_category = "crashes_stability"
        final_sentiment = "negative"  # Crashes are always negative

    # 3. Passwords & Login
    login_keywords = ["password", "log in", "login", "can't access account", "locked out"]
    if any(kw in text_lower for kw in login_keywords):
        final_category = "login_auth"

    # 4. Sentiment Overrides
    defect_keywords = ["bug", "glitch", "broken", "worst", "unusable", "terrible", "fix this", "hate"]
    if any(kw in text_lower for kw in defect_keywords):
        final_sentiment = "negative"

    return final_category, final_sentiment


def _build_review_items(reviews_df: pd.DataFrame, text_col: str, id_col: str, processed_ids: Set[str]) -> List[Dict[str, str]]:
    review_items: List[Dict[str, str]] = []
    for row in reviews_df.itertuples(index=False):
        review_id = getattr(row, id_col)
        review_id_str = _normalize_review_id(review_id)
        if review_id_str in processed_ids:
            continue
            
        review_text = getattr(row, text_col)
        
        # PRE-PROCESSING: Drop noisy/short reviews before they hit the model
        if not _is_valid_for_absa(review_text):
            continue
            
        review_items.append({"review_id": review_id_str, "review_text": review_text})
    return review_items


def label_review(
    review_text: str,
    taxonomy: List[Dict[str, str]],
    max_retries: int = MAX_RETRIES,
) -> Optional[List[Dict[str, str]]]:
    """Labels a single review with zero-shot classification."""
    if not _is_valid_for_absa(review_text):
        return None
        
    review_items = [{"review_id": "single", "review_text": review_text}]
    results, failures, _ = classify_review_batch(
        review_items,
        taxonomy,
        category_threshold=settings.zero_shot_category_threshold,
        fallback_threshold=settings.zero_shot_category_fallback_threshold,
        max_aspects_per_review=settings.zero_shot_max_aspects_per_review,
        batch_size=1,
    )
    if failures:
        return None
        
    final_results = []
    for item in results:
        # Apply overrides to single review processing
        cat, sent = _apply_heuristic_overrides(review_text, item["feature_key"], item["sentiment"])
        final_results.append({"feature_key": cat, "sentiment": sent})
        
    return final_results


def batch_label_reviews(
    reviews_df: pd.DataFrame,
    taxonomy: List[Dict[str, str]],
    text_col: str = "review_text",
    id_col: str = "id",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = 1,
    batch_pause_seconds: float = RATE_LIMIT_SECONDS,
    category_threshold: float = None,
    fallback_threshold: float = None,
    max_aspects_per_review: int = None,
    write_output: bool = True,
    resume_from_checkpoint: bool = True,
) -> pd.DataFrame:
    """
    Labels every review in reviews_df. Returns a long-format DataFrame with
    one row per (review_id, feature_key, sentiment) triple, plus a separate
    Parquet file of review_ids that failed labeling entirely.
    """
    try:
        total_reviews = len(reviews_df)
        out_dir = settings.resolve_path(settings.data_silver_labels_dir)
        if write_output:
            out_dir.mkdir(parents=True, exist_ok=True)

        labels_path = out_dir / LABELS_FILENAME
        failures_path = out_dir / FAILURES_FILENAME
        progress_path = out_dir / PROGRESS_FILENAME

        if resume_from_checkpoint:
            existing_labels_df = _load_parquet_if_exists(labels_path, LABEL_COLUMNS)
            existing_failures_df = _load_parquet_if_exists(failures_path, FAILURE_COLUMNS)
            existing_progress_df = _load_parquet_if_exists(progress_path, PROGRESS_COLUMNS)
        else:
            existing_labels_df = _empty_dataframe(LABEL_COLUMNS)
            existing_failures_df = _empty_dataframe(FAILURE_COLUMNS)
            existing_progress_df = _empty_dataframe(PROGRESS_COLUMNS)

        processed_ids = _collect_processed_ids(existing_progress_df, existing_labels_df, existing_failures_df)
        results: List[Dict] = existing_labels_df.to_dict(orient="records")
        failures: List[Dict] = existing_failures_df.to_dict(orient="records")
        progress_records: List[Dict] = existing_progress_df.to_dict(orient="records")
        initial_label_count = len(results)
        initial_failure_count = len(failures)

        review_items = _build_review_items(reviews_df, text_col, id_col, processed_ids)
        skipped_reviews = total_reviews - len(review_items)
        if skipped_reviews:
            logging.info(f"batch_label_reviews: skipping {skipped_reviews} already-processed reviews (or invalid length)")

        if batch_size < 1:
            raise FixFirstException("batch_size must be at least 1", sys)

        category_threshold = settings.zero_shot_category_threshold if category_threshold is None else category_threshold
        fallback_threshold = (
            settings.zero_shot_category_fallback_threshold if fallback_threshold is None else fallback_threshold
        )
        max_aspects_per_review = (
            settings.zero_shot_max_aspects_per_review if max_aspects_per_review is None else max_aspects_per_review
        )

        batches = _chunk_reviews(review_items, batch_size)
        completed_reviews = 0

        for batch_index, batch_reviews in enumerate(batches, start=1):
            batch_results, batch_failures, batch_progress = classify_review_batch(
                batch_reviews,
                taxonomy,
                category_threshold=category_threshold,
                fallback_threshold=fallback_threshold,
                max_aspects_per_review=max_aspects_per_review,
                batch_size=batch_size,
            )
            
            # Create text lookup dictionary to pair with incoming AI results
            text_lookup = {item["review_id"]: item["review_text"] for item in batch_reviews}
            
            # POST-PROCESSING: Apply heuristic overrides to the AI's predictions
            for res in batch_results:
                review_id = res.get("review_id")
                original_text = text_lookup.get(review_id, "")
                if original_text:
                    new_category, new_sentiment = _apply_heuristic_overrides(
                        review_text=original_text,
                        ai_category=res["feature_key"],
                        ai_sentiment=res["sentiment"]
                    )
                    res["feature_key"] = new_category
                    res["sentiment"] = new_sentiment

            results.extend(batch_results)
            failures.extend(batch_failures)
            progress_records.extend(batch_progress)
            completed_reviews += len(batch_reviews)

            if completed_reviews % 50 == 0 or completed_reviews == len(review_items):
                logging.info(
                    f"batch_label_reviews: processed {completed_reviews}/{len(review_items)} new reviews "
                    f"({batch_index}/{len(batches)} batches)"
                )

            if write_output:
                _write_checkpoint(out_dir, results, failures, progress_records)

        labels_df = pd.DataFrame(results, columns=LABEL_COLUMNS)
        failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
        new_label_count = len(results) - initial_label_count
        new_failure_count = len(failures) - initial_failure_count
        attempted_reviews = len(review_items)

        logging.info(
            f"batch_label_reviews: processed {attempted_reviews} new reviews -> "
            f"{new_label_count} new silver aspect labels, {new_failure_count} new failures "
            f"({new_failure_count/max(attempted_reviews,1):.1%}); checkpoint now contains "
            f"{len(labels_df)} labels"
        )

        if write_output:
            _write_checkpoint(out_dir, results, failures, progress_records)
            logging.info(f"batch_label_reviews: wrote outputs to {out_dir}")

        return labels_df
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)