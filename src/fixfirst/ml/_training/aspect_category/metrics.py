"""Metrics for multi-label aspect category classification."""

import sys
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from fixfirst.exceptions.exception import FixFirstException


def compute_metrics_from_logits(
    logits: np.ndarray,
    true_labels: np.ndarray,
    label_names: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Convert logits with sigmoid and compute aggregate/per-label metrics."""
    try:
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        predictions = (probabilities >= threshold).astype(int)
        true_labels = np.asarray(true_labels).astype(int)

        metrics: Dict[str, float] = {}
        for average in ("micro", "macro"):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels,
                predictions,
                average=average,
                zero_division=0,
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
