"""
Hybrid ABSA inference router.

For a single review, produces the final list of (feature_key, sentiment,
confidence, source) predictions by combining the fine-tuned models with
LLM fallback per confidence.py's routing rules:

  1. Run the fine-tuned CATEGORY model.
     - If confident: use its predicted feature_keys directly.
     - If NOT confident: defer the whole review to the LLM (via the
       already-tested labeling.silver_labeler.label_review), which
       returns both aspects AND sentiment in one call — no need for a
       second round trip.

  2. For features that came from the fine-tuned category model, run the
     fine-tuned SENTIMENT model per feature.
     - If confident: use its predicted sentiment directly.
     - If NOT confident: ask the LLM to re-analyze the whole review; if
       the LLM's own aspect extraction corroborates that feature, use the
       LLM's sentiment for it. If the LLM disagrees the feature is even
       discussed, keep the fine-tuned sentiment prediction (source stays
       "finetuned") rather than discarding a real prediction because a
       second, differently-scoped model didn't happen to surface it too.

Every returned aspect is tagged with source: "finetuned" or "llm_fallback",
which is exactly the AspectSource enum column on review_aspects — this
router's output maps 1:1 onto that table.
"""

import sys
from typing import Dict, List

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.inference.confidence import (
    category_needs_llm_fallback,
    category_predicted_labels,
    category_decision_confidence,
    sentiment_needs_llm_fallback,
    sentiment_decision_confidence,
    sentiment_predicted_label,
)
from fixfirst.logging.logger import logging

CATEGORY_SOURCE_FINETUNED = "finetuned"
CATEGORY_SOURCE_LLM = "llm_fallback"


def route_review(
    review_text: str,
    taxonomy: List[Dict[str, str]],
    predict_category_probs_fn,
    predict_sentiment_probs_fn,
    label_review_fn,
) -> List[Dict]:
    """
    Full hybrid routing for a single review. Dependency-injected model/LLM
    call functions (predict_category_probs_fn, predict_sentiment_probs_fn,
    label_review_fn) so this orchestration logic is unit-testable by
    mocking those three functions, without needing torch or a live LLM.

    Returns a list of dicts:
        {feature_key, sentiment, confidence, source}
    """
    try:
        feature_display_names = {t["feature_key"]: t["display_name"] for t in taxonomy}
        results: List[Dict] = []

        cat_probs, cat_label_names = predict_category_probs_fn(review_text)

        if category_needs_llm_fallback(cat_probs):
            llm_aspects = label_review_fn(review_text, taxonomy)
            if llm_aspects is None:
                logging.error("route_review: category LLM fallback failed to produce output; returning no aspects")
                return []

            for aspect in llm_aspects:
                results.append(
                    {
                        "feature_key": aspect["feature_key"],
                        "sentiment": aspect["sentiment"],
                        "confidence": None,  # LLM fallback has no calibrated probability
                        "source": CATEGORY_SOURCE_LLM,
                    }
                )
            return results

        # Category model was confident -> use its predicted feature set,
        # then resolve sentiment per feature (fine-tuned or LLM-gated).
        predicted_features = category_predicted_labels(cat_probs, cat_label_names)
        cat_confidences = dict(zip(cat_label_names, category_decision_confidence(cat_probs)))

        for feature_key in predicted_features:
            display_name = feature_display_names.get(feature_key, feature_key)
            sent_probs, sent_label_names = predict_sentiment_probs_fn(review_text, display_name)

            if sentiment_needs_llm_fallback(sent_probs):
                llm_aspects = label_review_fn(review_text, taxonomy)
                llm_match = None
                if llm_aspects:
                    llm_match = next((a for a in llm_aspects if a["feature_key"] == feature_key), None)

                if llm_match is not None:
                    results.append(
                        {
                            "feature_key": feature_key,
                            "sentiment": llm_match["sentiment"],
                            "confidence": None,
                            "source": CATEGORY_SOURCE_LLM,
                        }
                    )
                else:
                    # LLM didn't corroborate this feature; keep the fine-tuned
                    # sentiment prediction rather than dropping it entirely.
                    logging.warning(
                        f"route_review: LLM sentiment fallback for feature_key={feature_key!r} did not "
                        f"corroborate the aspect; retaining fine-tuned prediction."
                    )
                    results.append(
                        {
                            "feature_key": feature_key,
                            "sentiment": sentiment_predicted_label(sent_probs, sent_label_names),
                            "confidence": sentiment_decision_confidence(sent_probs),
                            "source": CATEGORY_SOURCE_FINETUNED,
                        }
                    )
            else:
                results.append(
                    {
                        "feature_key": feature_key,
                        "sentiment": sentiment_predicted_label(sent_probs, sent_label_names),
                        "confidence": sentiment_decision_confidence(sent_probs),
                        "source": CATEGORY_SOURCE_FINETUNED,
                    }
                )

        return results
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)