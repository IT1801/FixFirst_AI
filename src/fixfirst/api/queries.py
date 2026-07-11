"""DB query functions for the FixFirst AI API."""

import sys
from typing import List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from fixfirst.core.db import CriticalityScore, FeatureMaster, RawReview, ReviewAspect
from fixfirst.exceptions.exception import FixFirstException


def list_features(db: Session, active_only: bool = True) -> List[FeatureMaster]:
    """Return features ordered by display name."""
    try:
        query = db.query(FeatureMaster)
        if active_only:
            query = query.filter(FeatureMaster.is_active.is_(True))
        return query.order_by(FeatureMaster.display_name).all()
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def list_reviews(
    db: Session,
    feature_key: Optional[str] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[RawReview], int]:
    """Return paginated reviews and the total matching count."""
    try:
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
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def list_criticality_scores(
    db: Session,
    priority: Optional[str] = None,
    limit: int = 50,
) -> List[Tuple[CriticalityScore, FeatureMaster]]:
    """Return the latest score per feature ordered by priority."""
    try:
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
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def get_feature_trend(db: Session, feature_key: str) -> Optional[Tuple[FeatureMaster, List[CriticalityScore]]]:
    """Return a feature and its windowed scores, or None when missing."""
    try:
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
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc