"""Metrics for multi-label aspect category classification."""

import sys
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from fixfirst.exceptions.exception import FixFirstException


def find_optimal_threshold(logits: np.ndarray, true_labels: np.ndarray) -> float:
    """Finds the global threshold in [0.1, 0.9] that maximizes f1_macro."""
    best_thresh = 0.5
    best_f1 = -1.0
    
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    true_labels = np.asarray(true_labels).astype(int)
    
    for t in np.arange(0.1, 0.95, 0.05):
        predictions = (probabilities >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresh = float(t)
            
    return round(best_thresh, 2)


def compute_metrics_from_logits(
    logits: np.ndarray,
    true_labels: np.ndarray,
    label_names: Sequence[str],
    threshold: float = None,
) -> Dict[str, float]:
    """Convert logits with sigmoid and compute aggregate/per-label metrics."""
    try:
        if threshold is None:
            threshold = find_optimal_threshold(logits, true_labels)

        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        predictions = (probabilities >= threshold).astype(int)
        true_labels = np.asarray(true_labels).astype(int)

        labels_to_eval = [i for i in range(len(label_names)) if np.sum(true_labels[:, i]) > 0]
        if not labels_to_eval:
            labels_to_eval = list(range(len(label_names)))

        metrics: Dict[str, float] = {"optimal_threshold": threshold}
        for average in ("micro", "macro"):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels,
                predictions,
                average=average,
                zero_division=0,
                labels=labels_to_eval
            )
            metrics[f"precision_{average}"] = float(precision)
            metrics[f"recall_{average}"] = float(recall)
            metrics[f"f1_{average}"] = float(f1)

        _, _, per_label_f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            zero_division=0,
        )
        for index, label_name in enumerate(label_names):
            metrics[f"f1_{label_name}"] = float(per_label_f1[index])
            metrics[f"support_{label_name}"] = float(support[index])
        return metrics
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc
