"""
DB query functions for the FixFirst AI API.

Kept separate from route handlers (routes/*.py) so route handlers stay
thin (parse request -> call query fn -> serialize response) and these
functions can be tested or swapped independently — e.g. route tests can
mock these directly without needing a live Postgres connection.
"""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session, joinedload

from fixfirst.db.models import CriticalityScore, FeatureMaster, RawReview, ReviewAspect


def list_features(db: Session, active_only: bool = True) -> List[FeatureMaster]:
    query = db.query(FeatureMaster)
    if active_only:
        query = query.filter(FeatureMaster.is_active.is_(True))
    return query.order_by(FeatureMaster.display_name).all()


def list_reviews(
    db: Session,
    feature_key: Optional[str] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[RawReview], int]:
    """Returns (rows, total_count_before_pagination)."""
    query = db.query(RawReview).options(joinedload(RawReview.aspects).joinedload(ReviewAspect.feature))

    if feature_key or sentiment or source:
        query = query.join(RawReview.aspects).join(ReviewAspect.feature)
        if feature_key:
            query = query.filter(FeatureMaster.feature_key == feature_key)
        if sentiment:
            query = query.filter(ReviewAspect.sentiment == sentiment)
        if source:
            query = query.filter(RawReview.source == source)
        query = query.distinct()

    total = query.count()
    rows = query.order_by(RawReview.ingested_at.desc()).limit(limit).offset(offset).all()
    return rows, total


def list_criticality_scores(
    db: Session,
    priority: Optional[str] = None,  # "high" | "low" | None
    limit: int = 50,
) -> List[Tuple[CriticalityScore, FeatureMaster]]:
    """
    Returns the MOST RECENT window's score per feature (not every
    historical window — that's what /trends is for), ordered by score
    descending for "high" priority or ascending for "low" priority.
    """
    # Subquery: latest window_end per feature_id
    from sqlalchemy import func

    latest_window = (
        db.query(
            CriticalityScore.feature_id,
            func.max(CriticalityScore.window_end).label("latest_window_end"),
        )
        .group_by(CriticalityScore.feature_id)
        .subquery()
    )

    query = (
        db.query(CriticalityScore, FeatureMaster)
        .join(FeatureMaster, CriticalityScore.feature_id == FeatureMaster.id)
        .join(
            latest_window,
            (CriticalityScore.feature_id == latest_window.c.feature_id)
            & (CriticalityScore.window_end == latest_window.c.latest_window_end),
        )
    )

    if priority == "high":
        query = query.order_by(CriticalityScore.score.desc())
    elif priority == "low":
        query = query.order_by(CriticalityScore.score.asc())
    else:
        query = query.order_by(CriticalityScore.score.desc())

    return query.limit(limit).all()


def get_feature_trend(db: Session, feature_key: str) -> Optional[Tuple[FeatureMaster, List[CriticalityScore]]]:
    """Returns (feature, all_windows_sorted_by_window_start) or None if
    feature_key doesn't exist."""
    feature = db.query(FeatureMaster).filter(FeatureMaster.feature_key == feature_key).first()
    if feature is None:
        return None

    scores = (
        db.query(CriticalityScore)
        .filter(CriticalityScore.feature_id == feature.id)
        .order_by(CriticalityScore.window_start.asc())
        .all()
    )
    return feature, scores