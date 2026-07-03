"""
Confidence-gated routing logic for the hybrid ABSA pipeline.

Pure math/decision logic with NO model-loading dependency, so it's fully
unit-testable independent of torch/transformers being installed.

Routing decisions:

  CATEGORY level: per-label decision confidence is how sure the model is
  about whichever side of the 0.5 boundary it landed on — a label
  predicted positive at prob=0.98 is confident; a label predicted
  negative at prob=0.52 (i.e. 0.48 positive prob) is NOT confident, even
  though "negative" was still the argmax decision. If the model's LEAST
  confident label decision for a review falls below CATEGORY_CONFIDENCE_
  THRESHOLD, the whole review's aspect set is deferred to the LLM rather
  than trusting a partially-uncertain multi-label vector.

  SENTIMENT level: decision confidence is simply the softmax probability
  of the predicted class. Below SENTIMENT_CONFIDENCE_THRESHOLD, that
  single (review, feature) pair is deferred to the LLM — sentiment
  routing is per-pair, not per-review, since one review's features can
  vary widely in how clear-cut their sentiment is.
"""

import sys
from typing import Dict, List

import numpy as np

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException

CATEGORY_CONFIDENCE_THRESHOLD = settings.llm_fallback_threshold
SENTIMENT_CONFIDENCE_THRESHOLD = settings.llm_fallback_threshold


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def category_decision_confidence(probs: np.ndarray) -> np.ndarray:
    """
    probs: shape (n_labels,), sigmoid outputs in [0, 1].
    Returns per-label confidence in whichever decision (positive/negative)
    the probability implies: prob if >=0.5, else (1 - prob). Always in
    [0.5, 1.0] by construction.
    """
    return np.where(probs >= 0.5, probs, 1.0 - probs)


def category_needs_llm_fallback(
    probs: np.ndarray, threshold: float = CATEGORY_CONFIDENCE_THRESHOLD
) -> bool:
    """
    Returns True if the review's aspect-category prediction should be
    deferred to the LLM (i.e., the model's least-confident label decision
    falls below threshold).
    """
    try:
        confidences = category_decision_confidence(probs)
        return bool(np.min(confidences) < threshold)
    except Exception as e:
        raise FixFirstException(e, sys)


def category_predicted_labels(probs: np.ndarray, label_names: List[str]) -> List[str]:
    """Returns the feature_key names predicted positive (prob >= 0.5)."""
    return [name for name, p in zip(label_names, probs) if p >= 0.5]


def sentiment_decision_confidence(probs: np.ndarray) -> float:
    """probs: shape (3,), softmax output. Returns max(probs)."""
    return float(np.max(probs))


def sentiment_needs_llm_fallback(
    probs: np.ndarray, threshold: float = SENTIMENT_CONFIDENCE_THRESHOLD
) -> bool:
    return sentiment_decision_confidence(probs) < threshold


def sentiment_predicted_label(probs: np.ndarray, label_names: List[str]) -> str:
    return label_names[int(np.argmax(probs))]


def compute_fallback_rate_stats(source_counts: Dict[str, int]) -> Dict[str, float]:
    """
    source_counts: e.g. {"finetuned": 342, "llm_fallback": 58}
    Returns {"total": ..., "finetuned_rate": ..., "llm_fallback_rate": ...}
    — the metric that goes in the README ("X% of inferences required LLM
    fallback").
    """
    try:
        total = sum(source_counts.values())
        if total == 0:
            return {"total": 0, "finetuned_rate": 0.0, "llm_fallback_rate": 0.0}

        return {
            "total": total,
            "finetuned_rate": source_counts.get("finetuned", 0) / total,
            "llm_fallback_rate": source_counts.get("llm_fallback", 0) / total,
        }
    except Exception as e:
        raise FixFirstException(e, sys)