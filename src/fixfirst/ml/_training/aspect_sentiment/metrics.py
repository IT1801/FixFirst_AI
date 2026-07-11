"""
Metrics for the aspect SENTIMENT classifier (single-label, 3-class).

Separated from train.py for the same reason as the category classifier's
metrics module: pure numpy/sklearn logic, unit-testable without torch.
"""

import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._training.aspect_sentiment.dataset import SENTIMENT_LABELS


def compute_sentiment_metrics(logits: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    logits: shape (n_examples, 3), raw model outputs (pre-softmax)
    true_labels: shape (n_examples,), integer class indices matching
                 SENTIMENT_LABELS order
    """
    try:
        preds = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(true_labels, preds)
        labels_to_eval = [i for i in range(len(SENTIMENT_LABELS)) if np.sum(true_labels == i) > 0]
        if not labels_to_eval:
            labels_to_eval = list(range(len(SENTIMENT_LABELS)))

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, preds, average="macro", zero_division=0, labels=labels_to_eval
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
        }

        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            true_labels, preds, average=None, zero_division=0, labels=list(range(len(SENTIMENT_LABELS)))
        )
        for i, label_name in enumerate(SENTIMENT_LABELS):
            metrics[f"f1_{label_name}"] = float(per_class_f1[i])
            metrics[f"support_{label_name}"] = float(per_class_support[i])

        return metrics
    except Exception as e:
        raise FixFirstException(e, sys)