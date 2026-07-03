"""
Silver-labeling orchestration for FixFirst AI.

This version uses a local transformers zero-shot classifier instead of an
LLM. It still keeps checkpointed parquet outputs so interrupted runs can
resume from the last completed review.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Set

import pandas as pd

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.labeling.zero_shot import classify_review_batch

MAX_RETRIES = 1
DEFAULT_BATCH_SIZE = 8
RATE_LIMIT_SECONDS = 0.0
LABELS_FILENAME = "silver_labels.parquet"
FAILURES_FILENAME = "labeling_failures.parquet"
PROGRESS_FILENAME = "silver_labeling_progress.parquet"
LABEL_COLUMNS = ["review_id", "review_text", "feature_key", "sentiment"]
FAILURE_COLUMNS = ["review_id", "review_text"]
PROGRESS_COLUMNS = ["review_id", "review_text", "status"]


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


def _build_review_items(reviews_df: pd.DataFrame, text_col: str, id_col: str, processed_ids: Set[str]) -> List[Dict[str, str]]:
    review_items: List[Dict[str, str]] = []
    for row in reviews_df.itertuples(index=False):
        review_id = getattr(row, id_col)
        review_id_str = _normalize_review_id(review_id)
        if review_id_str in processed_ids:
            continue
        review_items.append({"review_id": review_id_str, "review_text": getattr(row, text_col)})
    return review_items


def label_review(
    review_text: str,
    taxonomy: List[Dict[str, str]],
    max_retries: int = MAX_RETRIES,
) -> Optional[List[Dict[str, str]]]:
    """Labels a single review with zero-shot classification."""
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
    return [
        {"feature_key": item["feature_key"], "sentiment": item["sentiment"]}
        for item in results
    ]


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

    Long format (rather than one row per review with a nested list) makes
    the QA notebook and later promotion-to-review_aspects step simpler:
    each row maps 1:1 onto a future review_aspects row.

    When resume_from_checkpoint is enabled, the function reloads the last
    saved parquet checkpoint in data/silver_labels/ and skips any review
    IDs already recorded there. That allows an interrupted run to continue
    from the next unlabeled review instead of replaying completed ones.

    Batching is the main efficiency optimization; the zero-shot classifier
    runs locally, so max_workers and batch_pause_seconds are kept only for
    backward compatibility with the old LLM CLI and are ignored.
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

        review_items = _build_review_items(reviews_df, text_col, id_col, processed_ids)
        skipped_reviews = total_reviews - len(review_items)
        if skipped_reviews:
            logging.info(f"batch_label_reviews: skipping {skipped_reviews} already-processed reviews")

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

        logging.info(
            f"batch_label_reviews: {total_reviews} reviews -> {len(labels_df)} silver aspect labels, "
            f"{len(failures_df)} reviews failed entirely ({len(failures_df)/max(total_reviews,1):.1%})"
        )

        if write_output:
            _write_checkpoint(out_dir, results, failures, progress_records)
            logging.info(f"batch_label_reviews: wrote outputs to {out_dir}")

        return labels_df
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)