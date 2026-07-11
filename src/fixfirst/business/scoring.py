"""Feature criticality scoring and aggregation."""

from fixfirst.business._scoring.aggregation import build_windowed_scores
from fixfirst.business._scoring.criticality import (
    DEFAULT_HALF_LIFE_DAYS,
    compute_criticality_score,
    compute_negative_ratio,
    compute_recency_weight,
)
from fixfirst.business._scoring.pipeline import (
    load_aspects_with_dates,
    run_scoring_pipeline,
    write_criticality_scores,
)

__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "build_windowed_scores",
    "compute_criticality_score",
    "compute_negative_ratio",
    "compute_recency_weight",
    "load_aspects_with_dates",
    "run_scoring_pipeline",
    "write_criticality_scores",
]
