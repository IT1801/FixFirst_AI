"""Shared utilities for classifier training."""

from typing import Dict, Iterable


def build_label_index(labels: Iterable[str]) -> Dict[str, int]:
    """Return a deterministic alphabetical label-to-index mapping."""
    return {label: index for index, label in enumerate(sorted(set(labels)))}
