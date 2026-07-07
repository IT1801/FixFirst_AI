"""
Zero-shot classification helpers for FixFirst AI silver labeling.
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

from fixfirst.constants import (
    ASPECT_HYPOTHESIS_TEMPLATE,
    SENTIMENT_HYPOTHESIS_TEMPLATE,
    SENTIMENT_LABELS,
)

_classifier_cache = {}


def _load_zero_shot_classifier():
    """Load and cache the Hugging Face zero-shot classifier."""
    if "classifier" in _classifier_cache:
        return _classifier_cache["classifier"]

    try:
        from fixfirst.config.configuration import settings
        import torch
        from transformers import pipeline

        if torch.cuda.is_available():
            device = 0
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            device_name = "mps"
        else:
            device = -1
            device_name = "cpu"

        pipeline_kwargs = {}
        if device_name == "mps":
            pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        classifier = pipeline(
            "zero-shot-classification",
            model=settings.zero_shot_model_name,
            device=device,
            **pipeline_kwargs,
        )
        _classifier_cache["classifier"] = classifier
        _classifier_cache["device"] = device
        logging.info(
            f"_load_zero_shot_classifier: loaded {settings.zero_shot_model_name} on "
            f"{device_name}"
        )
        return classifier
    except Exception as e:
        raise FixFirstException(
            "Zero-shot labeling requires torch and transformers. "
            "Run `make install-training` before `make label`.",
            sys,
        ) from e


def _clear_accelerator_cache() -> None:
    """Release cached accelerator allocations between inference stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


def _inference_batch_size(requested_batch_size: int) -> int:
    """Use conservative micro-batches on MPS, which shares RAM with macOS."""
    if _classifier_cache.get("device") == "mps":
        return 1
    return requested_batch_size


def _as_result_list(result_or_results):
    if isinstance(result_or_results, list):
        return result_or_results
    return [result_or_results]


def _select_aspect_labels(
    category_result: Dict[str, object],
    label_to_feature_key: Dict[str, str],
    label_to_display_name: Dict[str, str],  # <-- Added mapping parameter
    threshold: float,
    fallback_threshold: float,
    max_aspects_per_review: int,
) -> List[Dict[str, object]]:
    labels = category_result["labels"]
    scores = category_result["scores"]

    ranked = [
        {
            "display_name": label_to_display_name[label],  # <-- Resolve clear string here
            "feature_key": label_to_feature_key[label], 
            "score": float(score)
        }
        for label, score in sorted(zip(labels, scores), key=lambda pair: pair[1], reverse=True)
    ]

    selected = [item for item in ranked if item["score"] >= threshold]
    if not selected and ranked and ranked[0]["score"] >= fallback_threshold:
        selected = [ranked[0]]

    return selected[:max_aspects_per_review]


def _select_sentiment(sentiment_result: Dict[str, object]) -> str:
    labels = sentiment_result["labels"]
    scores = sentiment_result["scores"]
    if not labels:
        return "neutral"
    best_index = max(range(len(labels)), key=lambda idx: scores[idx])
    return str(labels[best_index])


def classify_review_batch(
    review_items: List[Dict[str, str]],
    taxonomy: List[Dict[str, str]],
    category_threshold: float,
    fallback_threshold: float,
    max_aspects_per_review: int,
    batch_size: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Classify a batch of reviews with a zero-shot model."""
    if not review_items:
        return [], [], []
    
    classifier = _load_zero_shot_classifier()
    inference_batch_size = _inference_batch_size(batch_size)

    feature_labels = [feature["hypothesis"] for feature in taxonomy]
    hypothesis_to_feature_key = {feature["hypothesis"]: feature["feature_key"] for feature in taxonomy}
    hypothesis_to_display_name = {feature["hypothesis"]: feature["display_name"] for feature in taxonomy}

    try:
        category_results = classifier(
            [item["review_text"] for item in review_items],
            candidate_labels=feature_labels,
            hypothesis_template=ASPECT_HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=inference_batch_size,
        )
        category_results = _as_result_list(category_results)

        aspect_jobs: List[Dict[str, str]] = []

        for review_item, category_result in zip(review_items, category_results):
            selected = _select_aspect_labels(
                category_result,
                hypothesis_to_feature_key,
                hypothesis_to_display_name,  # <-- Pass mapping cleanly
                threshold=category_threshold,
                fallback_threshold=fallback_threshold,
                max_aspects_per_review=max_aspects_per_review,
            )
            review_id = review_item["review_id"]

            for aspect in selected:
                aspect_jobs.append(
                    {
                        "review_id": review_id,
                        "review_text": review_item["review_text"],
                        "feature_key": aspect["feature_key"],
                        "feature_display_name": aspect["display_name"],
                    }
                )

        sentiment_results = []
        if aspect_jobs:
            del category_results
            _clear_accelerator_cache()
            sentiment_inputs = [
                f"Review: {job['review_text']}\nFeature: {job['feature_display_name']}"
                for job in aspect_jobs
            ]
            sentiment_results = classifier(
                sentiment_inputs,
                candidate_labels=SENTIMENT_LABELS,
                hypothesis_template=SENTIMENT_HYPOTHESIS_TEMPLATE,
                multi_label=False,
                batch_size=inference_batch_size,
            )
            sentiment_results = _as_result_list(sentiment_results)

        results: List[Dict] = []
        failures: List[Dict] = []
        progress_records: List[Dict] = []

        for review_item in review_items:
            progress_records.append({"review_id": review_item["review_id"], "review_text": review_item["review_text"], "status": "labeled"})

        for job, sentiment_result in zip(aspect_jobs, sentiment_results):
            sentiment = _select_sentiment(sentiment_result)
            results.append(
                {
                    "review_id": job["review_id"],
                    "review_text": job["review_text"],
                    "feature_key": job["feature_key"],
                    "sentiment": sentiment,
                }
            )

        return results, failures, progress_records
    except Exception as e:
        logging.warning(f"classify_review_batch: zero-shot batch inference failed, falling back to singles: {e}")
        _clear_accelerator_cache()
        return classify_reviews_fallback(
            review_items,
            taxonomy,
            category_threshold=category_threshold,
            fallback_threshold=fallback_threshold,
            max_aspects_per_review=max_aspects_per_review,
            batch_size=batch_size,
        )


def classify_reviews_fallback(
    review_items: List[Dict[str, str]],
    taxonomy: List[Dict[str, str]],
    category_threshold: float,
    fallback_threshold: float,
    max_aspects_per_review: int,
    batch_size: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fallback path brought in line with hypothesis configurations."""
    classifier = _load_zero_shot_classifier()
    
    # Mirror the hypothesis layout inside fallback route as well!
    feature_labels = [feature["hypothesis"] for feature in taxonomy]
    hypothesis_to_feature_key = {feature["hypothesis"]: feature["feature_key"] for feature in taxonomy}
    hypothesis_to_display_name = {feature["hypothesis"]: feature["display_name"] for feature in taxonomy}

    results: List[Dict] = []
    failures: List[Dict] = []
    progress_records: List[Dict] = []

    for review_item in review_items:
        review_id = review_item["review_id"]
        review_text = review_item["review_text"]
        try:
            category_result = classifier(
                review_text,
                candidate_labels=feature_labels,
                hypothesis_template=ASPECT_HYPOTHESIS_TEMPLATE,
                multi_label=True,
            )
            selected = _select_aspect_labels(
                category_result,
                hypothesis_to_feature_key,
                hypothesis_to_display_name,
                threshold=category_threshold,
                fallback_threshold=fallback_threshold,
                max_aspects_per_review=max_aspects_per_review,
            )

            if not selected:
                progress_records.append({"review_id": review_id, "review_text": review_text, "status": "labeled"})
                continue

            for aspect in selected:
                sentiment_result = classifier(
                    f"Review: {review_text}\nFeature: {aspect['display_name']}",
                    candidate_labels=SENTIMENT_LABELS,
                    hypothesis_template=SENTIMENT_HYPOTHESIS_TEMPLATE,
                    multi_label=False,
                )
                sentiment = _select_sentiment(sentiment_result)
                results.append(
                    {
                        "review_id": review_id,
                        "review_text": review_text,
                        "feature_key": aspect["feature_key"],
                        "sentiment": sentiment,
                    }
                )

            progress_records.append({"review_id": review_id, "review_text": review_text, "status": "labeled"})
        except Exception as e:
            logging.warning(f"classify_reviews_fallback: failed to classify review {review_id}: {e}")
            failures.append({"review_id": review_id, "review_text": review_text})
            progress_records.append({"review_id": review_id, "review_text": review_text, "status": "failed"})
        finally:
            _clear_accelerator_cache()

    return results, failures, progress_records