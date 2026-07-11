"""Inference router."""

import sys
from typing import Dict, List, Callable

from fixfirst.constants import SOURCE_FINETUNED
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._inference.confidence import (
    category_predicted_labels,
    sentiment_predicted_label,
    sentiment_decision_confidence,
)


class InferenceRouter:
    """Object-oriented router for inference."""

    def __init__(
        self,
        taxonomy: List[Dict[str, str]],
        predict_category_probs_fn: Callable,
        predict_sentiment_probs_fn: Callable,
    ):
        self.taxonomy = taxonomy
        self.feature_display_names = {t["feature_key"]: t["display_name"] for t in taxonomy}
        self.predict_category_probs_fn = predict_category_probs_fn
        self.predict_sentiment_probs_fn = predict_sentiment_probs_fn

    def route_review(self, review_text: str) -> List[Dict]:
        """
        Routing for a single review.
        Returns a list of dicts: {feature_key, sentiment, confidence, source}
        """
        try:
            results: List[Dict] = []
            cat_probs, cat_label_names, cat_threshold = self.predict_category_probs_fn(review_text)

            predicted_features = category_predicted_labels(cat_probs, cat_label_names, decision_threshold=cat_threshold)

            for feature_key in predicted_features:
                display_name = self.feature_display_names.get(feature_key, feature_key)
                sent_probs, sent_label_names = self.predict_sentiment_probs_fn(review_text, display_name)

                results.append(
                    {
                        "feature_key": feature_key,
                        "sentiment": sentiment_predicted_label(sent_probs, sent_label_names),
                        "confidence": sentiment_decision_confidence(sent_probs),
                        "source": SOURCE_FINETUNED,
                    }
                )

            return results
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e


def route_review(
    review_text: str,
    taxonomy: List[Dict[str, str]],
    predict_category_probs_fn,
    predict_sentiment_probs_fn,
    label_review_fn=None, # unused, kept for signature compatibility
) -> List[Dict]:
    """Backward compatibility wrapper."""
    router = InferenceRouter(
        taxonomy=taxonomy,
        predict_category_probs_fn=predict_category_probs_fn,
        predict_sentiment_probs_fn=predict_sentiment_probs_fn,
    )
    return router.route_review(review_text)
